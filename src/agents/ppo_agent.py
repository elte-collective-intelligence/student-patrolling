import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase # Added TensorDictBase import
from tensordict.nn import TensorDictModule # Removed TensorDictSequential - not used yet

# Use updated spec names directly
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec as Unbounded, DiscreteTensorSpec as Categorical
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator

import sys
import os
# Use abspath for robustness
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.env import PatrollingEnv # Assuming PatrollingEnv is defined correctly


if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda")
    print(f"Using device: {DEFAULT_DEVICE}")
else:
    DEFAULT_DEVICE = torch.device("cpu") # Default to CPU
    print(f"Using device: {DEFAULT_DEVICE}")


# --- Actor Network Definition ---
class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = MLP(
            in_features=obs_dim,
            out_features=num_actions,
            num_cells=[hidden_dim, hidden_dim],
            activation_class=nn.Tanh,
            activate_last_layer=False
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


# --- Critic Network Definition ---
class CriticNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = MLP(
            in_features=obs_dim,
            out_features=1,
            num_cells=[hidden_dim, hidden_dim],
            activation_class=nn.Tanh,
            activate_last_layer=False
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


# --- Function to Create Models ---
def create_ppo_models(
    env: PatrollingEnv,
    cfg: dict,
    device: torch.device = DEFAULT_DEVICE,
    ):
    hidden_dim = cfg.get("hidden_dim", 64)

    # Define keys based on environment structure
    obs_key = ("agents", "patrollers", "observation")
    action_key = ("agents", "patrollers", "action")
    log_prob_key = ("agents", "patrollers", "sample_log_prob")
    value_key = "state_value"
    logits_key = "logits"

    # --- Determine Dimensions ---
    try:
        # *** USE DICTIONARY ACCESS for Observation Spec ***
        patroller_obs_spec = env.observation_spec["agents"]["patrollers"]["observation"]
        obs_dim = patroller_obs_spec.shape[-1]
    except Exception as e:
         raise ValueError(f"Could not get observation spec/dim using dict access. Structure: {env.observation_spec}") from e

    try:
        # *** USE DICTIONARY ACCESS for Action Spec ***
        patroller_action_spec_leaf = env.action_spec["agents"]["patrollers"]["action"]
        num_actions = patroller_action_spec_leaf.space.n
    except Exception as e:
         # *** If nested dictionary access fails, use direct access (deprecated behaviour) ***
        try:
            print("Warning: Nested dict access for action_spec failed. Trying direct access (fallback).")
            patroller_action_spec_leaf = env.action_spec
            if not hasattr(patroller_action_spec_leaf, 'space'):
                 raise AttributeError("Direct access did not return a valid spec.")
            num_actions = patroller_action_spec_leaf.space.n
        except Exception as e2:
            raise ValueError(f"Could not get action spec/num_actions. Structure: {env.action_spec}") from e


    print(f"Detected Obs Dim: {obs_dim}, Num Actions: {num_actions}")

    # --- Create Actor ---
    actor_net = ActorNet(obs_dim, num_actions, hidden_dim).to(device)
    actor_module = TensorDictModule(
            module=actor_net, in_keys=[obs_key], out_keys=[logits_key],
        )
    policy_module = ProbabilisticActor(
        module=actor_module,
        spec=patroller_action_spec_leaf,
        in_keys=[logits_key],      # Reads root logits from module output
        # --- Only specify action key in out_keys ---
        out_keys=[action_key],     # Where to store the sampled action
        # --- End Change ---
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,    # Calculate log probability
        log_prob_key=log_prob_key  # Explicitly store log_prob under this nested key
    ).to(device)

    # --- Create Critic ---
    critic_net = CriticNet(obs_dim, hidden_dim).to(device) # Use obs_dim directly
    value_module = ValueOperator(
        module=critic_net, in_keys=[obs_key], out_keys=[value_key], # Output root value
    ).to(device)

    print("Policy Module Created (outputs nested action/log_prob).")
    print("Value Module Created (outputs root value).")
    return policy_module, value_module

# --- Basic Test ---
if __name__ == "__main__":
    print("Testing Agent/Model Definitions...")
    test_env = PatrollingEnv(num_patrollers=3, num_intruders=2)
    test_cfg = {"hidden_dim": 32}
    # create_ppo_models uses dictionary access for specs now
    policy, value_func = create_ppo_models(
        env=test_env, cfg=test_cfg, device=DEFAULT_DEVICE
    )

    td_initial = test_env.reset()
    print("\nInitial TD Structure (from env.reset):")
    print(f"  Keys: {td_initial.keys()}")
    print(f"  Obs shape: {td_initial[('agents', 'patrollers', 'observation')].shape}")

    # Test Policy Module
    print("\nTesting Policy Module Forward Pass...")
    td_policy_input = td_initial.select(("agents", "patrollers", "observation"))
    print(f"Policy Input TD Keys: {td_policy_input.keys()}")

    td_policy_output = policy(td_policy_input.clone())
    print(f"Policy Output TD Keys (Root): {td_policy_output.keys()}") # Expect 'agents', 'logits'
    nested_td_key = ("agents", "patrollers")
    patroller_td_out = td_policy_output.get(nested_td_key) # No need for default if assert passes
    print(f"Policy Output TD Keys (Nested under {nested_td_key}): {patroller_td_out.keys()}")

    # --- Corrected Assertions ---
    # Check NESTED action and NESTED log_prob
    assert "action" in patroller_td_out.keys()
    assert "sample_log_prob" in patroller_td_out.keys()
    # Check ROOT logits
    assert "logits" in td_policy_output.keys()
    print("Policy output keys checked (nested action, nested log_prob, root logits).")
    # --- End Correction ---

    # Check shapes
    action_shape = patroller_td_out["action"].shape
    log_prob_shape = patroller_td_out["sample_log_prob"].shape # Get from nested TD
    print(f"Sampled Action Shape: {action_shape}")
    print(f"Sampled LogProb Shape: {log_prob_shape}")
    assert action_shape == (*test_env.batch_size, test_env.num_patrollers)
    assert log_prob_shape == (*test_env.batch_size, test_env.num_patrollers)

    # Test Value Module (remains the same, checks root state_value)
    print("\nTesting Value Module Forward Pass...")
    td_value_input = td_initial.select(("agents", "patrollers", "observation"))
    print(f"Value Input TD Keys: {td_value_input.keys()}")
    td_value_output = value_func(td_value_input.clone())
    print(f"Value Output TD Keys (Root): {td_value_output.keys()}")
    assert "state_value" in td_value_output.keys()
    print("Value output key checked.")
    value_shape = td_value_output["state_value"].shape
    print(f"State Value Shape: {value_shape}")
    assert value_shape[-1] == 1
    assert value_shape[:-1] == (*test_env.batch_size, test_env.num_patrollers)

    print("\nBasic Agent/Model Testing Completed.")