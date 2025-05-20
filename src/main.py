import torch
from torch import nn, optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, ValueOperator, NormalParamExtractor, TanhNormal
from tensordict.nn import TensorDictModule

from env import PatrollingEnv  # Adjust import if needed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transform to flatten nested observation ---
class ObservationExtractor(TensorDictModule):
    def __init__(self):
        super().__init__(
            in_keys=[("agents", "patrollers", "observation")],
            out_keys=["observation"],
        )

    def forward(self, tensordict):
        obs = tensordict.get(("agents", "patrollers", "observation"))
        tensordict.set("observation", obs)
        return tensordict

def main():
    # Initialize environment
    env = PatrollingEnv(num_patrollers=3, num_intruders=2, device=DEVICE)
    env = TransformedEnv(env)
    env.set_seed(42)

    # Wrap environment with observation extractor
    obs_extractor = ObservationExtractor()
    env = TransformedEnv(env, transform=obs_extractor)

    # Check observation shape
    example_td = env.reset()
    obs_shape = example_td["observation"].shape[-1]
    n_agents = example_td["observation"].shape[-2]

    # Define policy network: input obs, output mean and std for actions
    # Action space: discrete with 5 actions per agent, so use Categorical or TanhNormal distribution accordingly.
    # Here we assume discrete actions, so use Categorical distribution.

    # For discrete actions, ProbabilisticActor expects logits output
    policy_net = nn.Sequential(
        nn.Linear(obs_shape, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 5),  # 5 discrete actions per agent
    )

    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["observation"],
        out_keys=["logits"],
    )

    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    # Value network for critic
    value_net = nn.Sequential(
        nn.Linear(obs_shape, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    value_module = TensorDictModule(
        module=value_net,
        in_keys=["observation"],
        out_keys=["state_value"],
    )
    value_operator = ValueOperator(value_module, in_keys=["observation"], out_keys=["value"])

    # GAE for advantage estimation
    gae = GAE(gamma=0.99, lmbda=0.95, value_network=value_operator)
    gae.set_keys(value="value", advantage="advantage", value_target="value_target")

    # PPO loss
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=value_operator,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        gae_lambda=0.95,
        gamma=0.99,
    )

    # Optimizer
    optimizer = optim.Adam(list(actor.parameters()) + list(value_operator.parameters()), lr=3e-4)

    # Replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=5000),
        sampler=SamplerWithoutReplacement(),
        batch_size=64,
    )

    # Data collector
    collector = SyncDataCollector(env, policy=actor, total_frames=10000, frames_per_batch=1000, device=DEVICE)

    # Training loop
    for batch_idx, tensordict_data in enumerate(collector):
        # Compute advantage
        gae(tensordict_data)

        # Flatten batch for training
        tensordict_data = tensordict_data.reshape(-1)

        # Store in replay buffer
        replay_buffer.extend(tensordict_data.cpu())

        # Sample mini-batches and optimize
        for _ in range(10):
            batch = replay_buffer.sample(64).to(DEVICE)
            loss_dict = loss_module(batch)
            optimizer.zero_grad()
            loss_dict["loss_objective"].backward()
            optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: reward mean {tensordict_data['next', 'reward'].mean().item():.3f}")

        if batch_idx >= 100:
            break

    # Save or evaluate the trained policy as needed

if __name__ == "__main__":
    main()
