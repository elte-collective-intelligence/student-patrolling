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
    action_key = ("agents", "patrollers", "action") # Note: Output action is at root by default
    logits_key = "logits" # Intermediate output

    # --- Determine Dimensions ---
    try:
        # Use direct dictionary access for specs
        patroller_obs_spec = env.observation_spec["agents"]["patrollers"]["observation"]
        obs_dim = patroller_obs_spec.shape[-1]
    except Exception as e:
         raise ValueError(f"Could not get observation spec/dim from env.observation_spec. Structure: {env.observation_spec}") from e

    try:
         # Use direct dictionary access for specs
        patroller_action_spec_leaf = env.action_spec["agents"]["patrollers"]["action"]
        num_actions = patroller_action_spec_leaf.space.n
    except Exception as e:
        # Fallback for deprecated behavior (action_spec might return leaf directly)
        try:
            print("Warning: Falling back to accessing action_spec leaf directly due to potential deprecation behavior.")
            patroller_action_spec_leaf = env.action_spec
            num_actions = patroller_action_spec_leaf.space.n
        except Exception as e2:
            raise ValueError(f"Could not get action spec/num_actions from env.action_spec. Structure: {env.action_spec}") from e2


    print(f"Detected Obs Dim: {obs_dim}, Num Actions: {num_actions}")

    # --- Create Actor ---
    actor_net = ActorNet(obs_dim, num_actions, hidden_dim).to(device)

    # Wrap network to handle TensorDict input/output
    actor_module = TensorDictModule(
            module=actor_net,
            in_keys=[obs_key],    # Reads observation from this nested key
            out_keys=[logits_key], # Outputs logits to this root key
        )

    # Wrap actor_module in ProbabilisticActor
    policy_module = ProbabilisticActor(
        module=actor_module,           # The module producing distribution params
        spec=patroller_action_spec_leaf, # The spec for the action space
        in_keys=[logits_key],          # Use the logits output by actor_module
        out_keys=["action"],           # Store sampled action at root "action" key
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,          # Will calculate log_prob and store as "sample_log_prob"
    ).to(device)

    # --- Create Critic ---
    # Using local observation for now (non-centralized critic)
    obs_dim_critic = obs_dim
    critic_in_keys = [obs_key] # Critic reads same observation key

    critic_net = CriticNet(obs_dim_critic, hidden_dim).to(device)

    # ValueOperator wraps the critic network
    value_module = ValueOperator(
        module=critic_net,
        in_keys=critic_in_keys, # Reads observation
        # out_keys defaults to ["state_value"] - stores value estimate at root
    ).to(device)

    print("Policy Module Created.")
    print("Value Module Created.")
    return policy_module, value_module


# --- Basic Test ---
if __name__ == "__main__":
    print("Testing Agent/Model Definitions...")

    # Create a dummy environment instance
    test_env = PatrollingEnv(num_patrollers=3, num_intruders=2)

    # Dummy config
    test_cfg = {"hidden_dim": 32}

    # Create models
    policy, value_func = create_ppo_models(
        env=test_env,
        cfg=test_cfg,
        device=DEFAULT_DEVICE
    )

    # Get a sample observation batch from the environment reset
    td_initial = test_env.reset()
    print("\nInitial TD Structure (from env.reset):")
    print(f"  Keys: {td_initial.keys()}")
    print(f"  Obs shape: {td_initial[('agents', 'patrollers', 'observation')].shape}")


    # Test Policy Module
    print("\nTesting Policy Module Forward Pass...")
    actor_obs_key_to_select = ("agents", "patrollers", "observation") # Defined earlier conceptually
    # Input only needs the observation key specified in policy's underlying module
    td_policy_input = td_initial.select(actor_obs_key_to_select) # Select using the known key
    
    print(f"Policy Input TD Keys: {td_policy_input.keys()}")

    # Pass data through the policy module
    # This TD will contain:
    # - original input observation
    # - "logits" added by TensorDictModule
    # - "action" added by ProbabilisticActor
    # - "sample_log_prob" added by ProbabilisticActor (because return_log_prob=True)
    td_policy_output = policy(td_policy_input)
    print(f"Policy Output TD Keys: {td_policy_output.keys()}")

    # Check expected output keys at the ROOT level
    assert "action" in td_policy_output.keys()
    assert "logits" in td_policy_output.keys()
    assert "sample_log_prob" in td_policy_output.keys()
    print("Policy output keys checked (action, logits, sample_log_prob).")
    print(f"Sampled Action Shape: {td_policy_output['action'].shape}")
    assert td_policy_output["action"].shape == (*test_env.batch_size, test_env.num_patrollers)

    # Test Value Module
    print("\nTesting Value Module Forward Pass...")
    
    critic_obs_key_to_select = ("agents", "patrollers", "observation") # Defined earlier conceptually


    # Input only needs the observation key specified in value_func's in_keys
    td_value_input = td_initial.select(critic_obs_key_to_select) # Select using the known key
    
    print(f"Value Input TD Keys: {td_value_input.keys()}")

    # Pass data through the value module
    # This TD will contain:
    # - original input observation
    # - "state_value" added by ValueOperator
    td_value_output = value_func(td_value_input)
    print(f"Value Output TD Keys: {td_value_output.keys()}")

    # Check output key ("state_value") and shape
    assert "state_value" in td_value_output.keys()
    print("Value output key checked.")
    print(f"State Value Shape: {td_value_output['state_value'].shape}")
    # Check value shape aligns with [B, Np, 1] because critic takes local obs
    assert td_value_output["state_value"].shape[-1] == 1
    assert td_value_output["state_value"].shape[:-1] == td_value_input[("agents","patrollers","observation")].shape[:-1]

    print("\nBasic Agent/Model Testing Completed.")