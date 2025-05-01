import torch
import hydra # Import hydra
from omegaconf import DictConfig, OmegaConf # Import OmegaConf for config manipulation

# Add path to import agent and env modules
import sys
import os
# Add the 'src' directory (parent of this main.py) to the Python path
# Use abspath for robustness
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from envs.env import PatrollingEnv
from agents.ppo_agent import create_ppo_models

# --- Set Device ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Main script using device: {device} (CUDA)")
else:
    device = torch.device("cpu")
    print(f"Main script using device: {device} (CPU)")


# --- Hydra Entry Point ---
# config_path should be relative to THIS file (src/main.py)
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training script."""

    # Print the loaded and merged configuration
    print("-------------------- Configuration --------------------")
    print(OmegaConf.to_yaml(cfg))
    print("-------------------------------------------------------")

    # --- Seed everything ---
    seed = cfg.seed
    torch.manual_seed(seed)
    # Optional: Seed other libraries if needed
    # import numpy as np
    # import random
    # np.random.seed(seed)
    # random.seed(seed)
    print(f"Using seed: {seed}")

    # --- Environment Setup ---
    # Define a function that creates an env instance using the config
    def create_env_fn():
        env = PatrollingEnv(
            num_patrollers=cfg.env.num_patrollers,
            num_intruders=cfg.env.num_intruders,
            env_size=cfg.env.env_size,
            max_steps=cfg.env.max_steps,
            patroller_move_dist=cfg.env.patroller_move_dist,
            intruder_move_dist=cfg.env.intruder_move_dist,
            detection_radius=cfg.env.detection_radius,
            device=device # Pass the selected device
        )
        # TODO: Add transforms here later if needed (e.g., ObservationNorm)
        return env

    print("\n--- Environment Setup ---")
    # Create a temporary env to get specs (needed for model creation)
    temp_env = create_env_fn()
    print("Temporary environment created to get specs.")

    # --- Agent/Model Setup ---
    print("\n--- Agent/Model Setup ---")
    # Pass the temp env instance and algo config to create models
    policy_module, value_module = create_ppo_models(
        env=temp_env,
        cfg=cfg.algo, # Pass the 'algo' section of the config
        device=device
    )
    # No longer need the temp env
    del temp_env
    print("Policy and Value modules created.")


    # --- Data Collector Setup ---
    print("\n--- Data Collector Setup ---")
    # Uses the create_env_fn to instantiate the environment
    # Runs the policy_module to get actions
    # Collects batches of frames_per_batch size
    collector = SyncDataCollector(
        create_env_fn=create_env_fn,    # Function to create env instance(s)
        policy=policy_module,           # Policy module to use for action selection
        frames_per_batch=cfg.frames_per_batch, # Number of frames in each batch
        total_frames=cfg.total_frames,  # Total frames to collect over the whole training
        device=device,                  # Device for collector operation (can be different from env/policy)
        # Other options exist (e.g., max_frames_per_traj)
    )
    print(f"SyncDataCollector created. Collecting {cfg.frames_per_batch} frames per batch.")
    print(f"Total frames to collect: {cfg.total_frames}")

    # --- Loss Function Setup ---
    print("\n--- Loss Function Setup ---")
    # Define the Advantage module (GAE)
    # It uses the value_module (critic) to estimate advantages
    advantage_module = GAE(
        gamma=cfg.algo.gamma,
        lmbda=cfg.algo.gae_lambda,
        value_network=value_module, # Pass the instantiated value module
        average_gae=False, # Standard GAE calculation per-step
    )
    print("GAE Advantage module created.")

    # Define the PPO loss module
    # It requires the policy (actor) and value (critic) modules
    # It will use the advantage module implicitly if value_network is GAE
    loss_module = ClipPPOLoss(
        actor=policy_module,    # Pass the instantiated policy module
        critic=value_module,    # Pass the instantiated value module
        clip_epsilon=cfg.algo.clip_epsilon,
        entropy_coef=cfg.algo.entropy_coef,
        value_loss_coef=cfg.algo.value_loss_coef,
        loss_critic_type="l2", # Use Mean Squared Error for value loss (common)
        # normalize_advantage=True, # Optional: Can help stabilize training
    )
    print("ClipPPOLoss module created.")

    # --- Optimizer Setup ---
    print("\n--- Optimizer Setup ---")
    # Combine parameters from both policy and value networks
    # Use loss_module.parameters() which conveniently gathers them
    optimizer = torch.optim.Adam(
        loss_module.parameters(), # Pass parameters from the loss module (actor + critic)
        lr=cfg.algo.lr
    )
    print(f"Adam optimizer created with learning rate: {cfg.algo.lr}")

    # --- Training Loop (Placeholder) ---
    print("\n--- Training Loop ---")
    print("Placeholder: Training loop will go here.")
    # for i, data in enumerate(collector): ...

    print("\nPhase 3 Setup Complete (Placeholders in place). Ready for next steps.")


# --- Run the main function ---
if __name__ == "__main__":
    main() # This executes the Hydra application