import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

# Use full names from torchrl.data for clarity, especially in mock specs
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec
)
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator

import sys
import os
# Use abspath for robustness
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.env import PatrollingEnv # For type hinting and Mock constants

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda")
else:
    DEFAULT_DEVICE = torch.device("cpu")
print(f"Agent using device: {DEFAULT_DEVICE}")


# --- Actor Network Definition ---
# Expects a flat observation tensor: [Batch, FlatObsDim]
# Outputs logits tensor: [Batch, NumPatrollers, NumActionsPerAgent]
class ActorNet(nn.Module):
    def __init__(self, flat_obs_dim: int, num_patrollers: int, num_actions_per_agent: int, hidden_dim: int = 64):
        super().__init__()
        self.num_patrollers = num_patrollers
        self.num_actions_per_agent = num_actions_per_agent
        self.mlp = MLP(
            in_features=flat_obs_dim,
            out_features=num_patrollers * num_actions_per_agent, # Output flat, then reshape in forward
            num_cells=[hidden_dim, hidden_dim],
            activation_class=nn.Tanh,
            activate_last_layer=False
        )

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        logits_flat = self.mlp(obs_flat)
        # Reshape logits to [Batch (if any), NumPatrollers, NumActionsPerAgent]
        if logits_flat.ndim > 1: # Batched input like [B, Np*Na]
            return logits_flat.unflatten(-1, (self.num_patrollers, self.num_actions_per_agent))
        else: # Non-batched input like [Np*Na] for test
            return logits_flat.reshape(self.num_patrollers, self.num_actions_per_agent)


# --- Critic Network Definition ---
# Expects a flat observation tensor: [Batch, FlatObsDim]
# Outputs a single value estimate: [Batch, 1]
class CriticNet(nn.Module):
    def __init__(self, flat_obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = MLP(
            in_features=flat_obs_dim,
            out_features=1,
            num_cells=[hidden_dim, hidden_dim],
            activation_class=nn.Tanh,
            activate_last_layer=False
        )
    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs_flat)


# --- Function to Create Models (for FLATTENED interface) ---
def create_ppo_models(
    env, # MockTransformedEnvSpecs (or object with similar .observation_spec, .action_spec, .num_patrollers)
    cfg, # AlgoConfig instance
    device: torch.device = DEFAULT_DEVICE,
    ):
    hidden_dim = getattr(cfg, "hidden_dim", 64) # Use getattr for SimpleConfig

    # Keys policy/value modules will operate on (at root of TensorDict)
    flat_obs_key = "observation_flat"
    action_key_root = "action"
    log_prob_key_root = "sample_log_prob"
    logits_key_root = "logits"
    value_key_root = "state_value"

    # --- Determine Dimensions from TRANSFORMED Specs ---
    try:
        obs_spec_transformed = env.observation_spec[flat_obs_key]
        flat_obs_dim = obs_spec_transformed.shape[-1]

        # Action spec for policy is now root due to RemapActionToNested transform_action_spec
        if isinstance(env.action_spec, CompositeSpec) and action_key_root in env.action_spec.keys():
            action_spec_transformed_leaf = env.action_spec[action_key_root]
        elif hasattr(env.action_spec, 'space') and hasattr(env.action_spec.space, 'n'):
            print("Info: env.action_spec appears to be the leaf spec directly for create_ppo_models.")
            action_spec_transformed_leaf = env.action_spec
        else:
            raise ValueError(f"Cannot determine action spec leaf for create_ppo_models. ActionSpec: {env.action_spec}")

        num_actions_per_agent = action_spec_transformed_leaf.space.n
        num_patrollers = env.num_patrollers # Get from mock env

    except Exception as e:
         raise ValueError(f"Could not get transformed specs in create_ppo_models.\n"
                          f"ObsSpec: {getattr(env, 'observation_spec', 'MISSING')}\n"
                          f"ActionSpec: {getattr(env, 'action_spec', 'MISSING')}") from e

    print(f"Agent create_ppo_models: Flat Obs Dim: {flat_obs_dim}, Num Patrollers: {num_patrollers}, Actions/Agent: {num_actions_per_agent}")

    # --- Create Actor Network ---
    # ActorNet needs Np and N_actions_per_agent for correct logit reshaping
    actor_net = ActorNet(flat_obs_dim, num_patrollers, num_actions_per_agent, hidden_dim)

    # --- Create Policy Module (Operates on FLAT obs, outputs ROOT action/log_prob) ---
    policy_module = ProbabilisticActor(
        module=TensorDictModule(
            module=actor_net,
            in_keys=[flat_obs_key], # Reads "observation_flat"
            out_keys=[logits_key_root]  # Outputs "logits" (reshaped by ActorNet to [B,Np,N_actions])
        ),
        spec=action_spec_transformed_leaf, # Spec for root "action" (shape [B, Np])
        in_keys=[logits_key_root],        # Reads root "logits"
        out_keys=[action_key_root],       # Writes root "action" (shape [B, Np])
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        log_prob_key=log_prob_key_root # Writes root "sample_log_prob" (shape [B, Np])
    ).to(device)

    # --- Create Critic Network ---
    critic_net = CriticNet(flat_obs_dim, hidden_dim)

    # --- Create Value Module (Operates on FLAT obs, outputs ROOT value) ---
    value_module = ValueOperator(
        module=critic_net,
        in_keys=[flat_obs_key],  # Reads flat observation
        out_keys=[value_key_root], # Writes root state_value (shape [B, 1])
    ).to(device)

    print("Policy Module Created (expects flat_obs, outputs root action/log_prob).")
    print("Value Module Created (expects flat_obs, outputs root value).")
    return policy_module, value_module

# --- Basic Test (Adjusted for FLATTENED Interface) ---
if __name__ == "__main__":
    print("Testing Agent/Model Definitions (Flattened Interface)...")

    # Constants for mock environment setup
    N_PATROLLERS_MOCK = 3
    N_INTRUDERS_MOCK = 2 # For calculating obs_per_agent for PatrollingEnv internals
    _base_env_temp_for_calc = PatrollingEnv(num_patrollers=N_PATROLLERS_MOCK, num_intruders=N_INTRUDERS_MOCK)
    OBS_PER_AGENT_MOCK = _base_env_temp_for_calc.obs_dim_per_patroller
    _base_env_temp_for_calc.close()
    FLAT_OBS_DIM_MOCK = N_PATROLLERS_MOCK * OBS_PER_AGENT_MOCK
    N_ACTIONS_PER_AGENT_MOCK = 5

    class MockTransformedEnvSpecs:
        observation_spec = CompositeSpec({
            "observation_flat": UnboundedContinuousTensorSpec(
                shape=torch.Size([FLAT_OBS_DIM_MOCK]), dtype=torch.float32, device=DEFAULT_DEVICE)
        }, shape=torch.Size([]))
        action_spec = CompositeSpec({
             "action": DiscreteTensorSpec(
                 n=N_ACTIONS_PER_AGENT_MOCK,
                 shape=torch.Size([N_PATROLLERS_MOCK]),
                 dtype=torch.int64, device=DEFAULT_DEVICE)
        }, shape=torch.Size([]))
        batch_size = torch.Size([])
        num_patrollers = N_PATROLLERS_MOCK # Policy needs this

    print(f"Mock Specs: Obs Flat Shape: {MockTransformedEnvSpecs.observation_spec['observation_flat'].shape}, "
          f"Action Root Shape: {MockTransformedEnvSpecs.action_spec['action'].shape}, "
          f"NumPatrollers for Mock: {MockTransformedEnvSpecs.num_patrollers}")

    test_cfg_flat = {"hidden_dim": 32} # For AlgoConfig access
    policy, value_func = create_ppo_models(
        env=MockTransformedEnvSpecs(),
        cfg=type('AlgoConfigMock', (), test_cfg_flat)(), # Mock the AlgoConfig object
        device=DEFAULT_DEVICE
    )

    sample_flat_obs = torch.randn(FLAT_OBS_DIM_MOCK, device=DEFAULT_DEVICE)
    td_input_flat = TensorDict({"observation_flat": sample_flat_obs}, batch_size=[])

    print("\nTesting Policy Module Forward Pass (Flat Interface)...")
    td_policy_output = policy(td_input_flat.clone())
    print(f"Policy Output TD Keys (Root): {td_policy_output.keys()}")
    assert "action" in td_policy_output.keys()
    assert "sample_log_prob" in td_policy_output.keys()
    assert "logits" in td_policy_output.keys()
    print("Policy output keys checked (root action, root sample_log_prob, root logits).")
    print(f"Sampled Action Shape: {td_policy_output['action'].shape}")
    assert td_policy_output["action"].shape == torch.Size([N_PATROLLERS_MOCK])
    print(f"Sampled LogProb Shape: {td_policy_output['sample_log_prob'].shape}")
    assert td_policy_output['sample_log_prob'].shape == torch.Size([N_PATROLLERS_MOCK])

    print("\nTesting Value Module Forward Pass (Flat Interface)...")
    td_value_output = value_func(td_input_flat.clone())
    print(f"Value Output TD Keys (Root): {td_value_output.keys()}")
    assert "state_value" in td_value_output.keys()
    print("Value output key checked (root state_value).")
    print(f"State Value Shape: {td_value_output['state_value'].shape}")
    assert td_value_output['state_value'].shape == torch.Size([1])

    print("\nBasic Agent/Model Testing (Flattened Interface) Completed.")