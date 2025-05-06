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
    action_key = ("agents", "patrollers", "action") # Expected structure in env
    logits_key = "logits" # Intermediate output

    # --- Determine Dimensions ---
    try:
        # Use direct dictionary access for observation spec
        patroller_obs_spec = env.observation_spec["agents"]["patrollers"]["observation"]
        obs_dim = patroller_obs_spec.shape[-1]
    except Exception as e:
         # This shouldn't happen if env spec is correct, but raise informative error
         raise ValueError(f"Could not get observation spec/dim using dict access. Structure: {env.observation_spec}") from e

    try:
        # Use direct access for action spec based on DeprecationWarning behavior
        patroller_action_spec_leaf = env.action_spec
        # Verify it has the expected attributes before using
        if not hasattr(patroller_action_spec_leaf, 'space') or not hasattr(patroller_action_spec_leaf.space, 'n'):
             # If direct access didn't work, try dictionary access as a fallback
             print("Warning: Direct access for action_spec failed. Trying dictionary access.")
             patroller_action_spec_leaf = env.action_spec["agents"]["patrollers"]["action"]
        num_actions = patroller_action_spec_leaf.space.n
    except Exception as e:
         raise ValueError(f"Could not get action spec/num_actions. Structure: {env.action_spec}") from e


    print(f"Detected Obs Dim: {obs_dim}, Num Actions: {num_actions}")

    # --- Create Actor ---
    actor_net = ActorNet(obs_dim, num_actions, hidden_dim).to(device)

    # Inner module maps nested observation to root logits
    actor_module = TensorDictModule(
            module=actor_net,
            in_keys=[obs_key],
            out_keys=[logits_key],
        )

    # ProbabilisticActor reads root logits, WRITES NESTED action
    policy_module = ProbabilisticActor(
        module=actor_module,
        spec=patroller_action_spec_leaf, # Pass the leaf spec
        in_keys=[logits_key],
        out_keys=[action_key], # Write action to ("agents", "patrollers", "action")
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True, # Log prob goes to ("agents", "patrollers", "sample_log_prob")
    ).to(device)

    # --- Create Critic ---
    obs_dim_critic = obs_dim
    critic_in_keys = [obs_key]
    critic_net = CriticNet(obs_dim_critic, hidden_dim).to(device)
    value_module = ValueOperator(
        module=critic_net,
        in_keys=critic_in_keys,
        out_keys=["state_value"], # Keep value output at root
    ).to(device)

    print("Policy Module Created (outputs nested action).")
    print("Value Module Created (outputs root value).")
    # Return the modules - main.py will handle interaction
    return policy_module, value_module


# --- Basic Test ---
if __name__ == "__main__":
    print("Testing Agent/Model Definitions...")
    test_env = PatrollingEnv(num_patrollers=3, num_intruders=2)
    test_cfg = {"hidden_dim": 32}
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
    print(f"Policy Output TD Keys (Root): {td_policy_output.keys()}")
    nested_td_key = ("agents", "patrollers")
    if nested_td_key in td_policy_output.keys(include_nested=True):
        patroller_td_out = td_policy_output[nested_td_key]
        print(f"Policy Output TD Keys (Nested under {nested_td_key}): {patroller_td_out.keys()}")

        # Check for NESTED action
        assert "action" in patroller_td_out.keys()
        # Check for ROOT logits and sample_log_prob
        assert "logits" in td_policy_output.keys()
        assert "sample_log_prob" in td_policy_output.keys() # <<< Check ROOT
        print("Policy output keys checked (nested action, root logits, root log_prob).") # Updated message

        action_shape = patroller_td_out["action"].shape
        print(f"Sampled Action Shape: {action_shape}")
        assert action_shape == (*test_env.batch_size, test_env.num_patrollers)
    else:
        print(f"ERROR: Nested key {nested_td_key} not found in policy output!")


    # Test Value Module
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
