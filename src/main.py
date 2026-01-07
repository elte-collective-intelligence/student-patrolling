import torch
import torch.optim as optim
import time
import tqdm
import traceback

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchrl.envs import TransformedEnv, Compose
from torchrl.envs.transforms import FlattenObservation as TorchRLFlattenObservation, Transform, RenameTransform
from torchrl.envs.utils import ExplorationType
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
    Categorical as TorchRLCategorical  # Add this import at the top
)
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical

from envs.env import PatrollingEnv, EnvBase
from agents.ppo_agent import create_ppo_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Main script using device: {DEVICE}")

# --- Configuration (Replaces Hydra for now) ---
class SimpleConfig:
    def __init__(self):
        self.seed = 42
        self.total_frames = 100_000
        self.frames_per_batch = 2048
        self.env = type('EnvConfig', (), {"num_patrollers": 3, "num_intruders": 2, "env_size": 10.0, "max_steps": 100, "patroller_move_dist": 0.5, "intruder_move_dist": 0.2, "detection_radius": 1.0})()
        self.algo = type('AlgoConfig', (), {"clip_epsilon": 0.2, "entropy_coef": 0.01, "value_loss_coef": 0.5, "gamma": 0.99, "gae_lambda": 0.95, "ppo_epochs": 10, "lr": 3e-4, "max_grad_norm": 0.5, "minibatch_size": 512, "hidden_dim": 64})()

# Main function
def main(cfg: SimpleConfig) -> None:
    print("Configuration...\n", vars(cfg.env), "\n", vars(cfg.algo))
    torch.manual_seed(cfg.seed)
    print(f"Seed: {cfg.seed}")

    def create_env_fn_transformed():
        base_env = PatrollingEnv(
            num_patrollers=cfg.env.num_patrollers,
            num_intruders=cfg.env.num_intruders,
            env_size=cfg.env.env_size,
            max_steps=cfg.env.max_steps,
            patroller_move_dist=cfg.env.patroller_move_dist,
            intruder_move_dist=cfg.env.intruder_move_dist,
            detection_radius=cfg.env.detection_radius,
            device=DEVICE
        )
        
        # First transform: Flatten observation
        flatten_obs = TorchRLFlattenObservation(
            in_keys=[("agents", "patrollers", "observation")],
            out_keys=["observation_flat"],
            first_dim=-2,
            last_dim=-1,
        )
        
        # Fix RenameTransform for TorchRL 0.8.1:
        action_transform = RenameTransform(
            in_keys=["action"],  # What the policy outputs (root)
            out_keys=[("agents", "patrollers", "action")],  # What the env expects (nested)
        )
        
        # Apply transforms in correct order
        transforms = Compose(
            flatten_obs,      # First flatten observation
            action_transform  # Then handle action mapping
        )
        
        transformed_env = TransformedEnv(
            base_env,
            transforms,
            device=DEVICE
        )
        
        return transformed_env

    # --- Testing TransformedEnv Interaction ---
    print("\n--- Testing TransformedEnv Interaction ---")
    try:
        transformed_env_test = create_env_fn_transformed()
        print(f"Transformed Env Test Batch Size: {transformed_env_test.batch_size}")
        print(f"Transformed Env Test Obs Spec: {transformed_env_test.observation_spec}")
        print(f"Transformed Env Test Action Spec: {transformed_env_test.action_spec}")


        print("\nTesting reset...")
        # Reset calls base_env.reset() then applies FORWARD transforms
        # Expected: FO applies, RMAN (inverse) does nothing -> Output is TD with 'observation_flat', root terminal flags
        td_from_reset = transformed_env_test.reset().to(DEVICE)
        print(f"\nFull td_from_reset AFTER transformed_env.reset():")
        print(td_from_reset)
        print(f"td_from_reset keys (incl nested): {td_from_reset.keys(True, True)}")

        assert "observation_flat" in td_from_reset.keys(), "Missing flattened observation!"
        assert ("agents", "patrollers", "observation") in td_from_reset.keys(True, True), "Missing nested observation structure!"

        # Verify shapes
        flat_obs = td_from_reset["observation_flat"]
        nested_obs = td_from_reset["agents", "patrollers", "observation"]
        assert flat_obs.shape[-1] == nested_obs.shape[-1] * nested_obs.shape[-2], "Flattened shape doesn't match nested shape"


        print("\nTesting step simulation...")
        try:
            # Get action spec from nested path since transform isn't working properly
            action_spec = transformed_env_test.action_spec["agents", "patrollers", "action"]
            mock_policy_action = action_spec.rand().to(DEVICE)
            print(f"Sampled mock action shape: {mock_policy_action.shape}")

            # Create input TD for step with root action
            td_input_to_step_test = td_from_reset.clone()
            # Set action at root level as expected by transform
            td_input_to_step_test.set("action", mock_policy_action)
            
            print("\nInput TD for step:")
            print(f"Keys (nested): {td_input_to_step_test.keys(True, True)}")
            print(f"Action shape: {td_input_to_step_test['action'].shape}")
            
            # Test step
            td_after_step = transformed_env_test.step(td_input_to_step_test)

            print(f"\n--- TD AFTER TransformedEnv.step() ---")
            print(td_after_step)
            print(f"td_after_step keys (incl nested): {td_after_step.keys(True, True)}")

            # Check for keys relevant for downstream: reward, done flags, and the 'next' state with observation_flat
            print("\nChecking TransformedEnv.step() output structure:")

            if "reward" in td_after_step.keys():
                 print(f"  Found root key 'reward'. Shape: {td_after_step['reward'].shape}")
                 # assert shape matches (*Batch, NumPatrollers)
                 # Skipping strict shape assert for now due to potential EnvBase wrapper effects, focusing on key presence
            else: print("  MISSING root key 'reward' from TransformedEnv.step() output!")

            if "done" in td_after_step.keys() and "terminated" in td_after_step.keys() and "truncated" in td_after_step.keys():
                 print(f"  Found root terminal flags ('done', 'terminated', 'truncated'). Shapes: {td_after_step['done'].shape}, {td_after_step['terminated'].shape}, {td_after_step['truncated'].shape}")
                 # assert shape matches (*Batch, 1)
            else: print("  MISSING one or more root terminal flags from EnvBase.step output root!")

            next_td_output_from_transformed = td_after_step.get("next", None)
            if next_td_output_from_transformed is not None:
                print(f"  Found root key 'next'. Keys within 'next' (incl nested): {next_td_output_from_transformed.keys(True, True)}")

                # AFTER FORWARD transforms on next state: expect 'observation_flat' inside 'next'
                if "observation_flat" in next_td_output_from_transformed.keys():
                    print(f"    SUCCESS: Found next observation at ('next', 'observation_flat'). Shape: {next_td_output_from_transformed['observation_flat'].shape}")
                    # assert shape matches (*Batch, FlatObsDim)
                elif ("agents", "patrollers", "observation") in next_td_output_from_transformed.keys(True,True):
                 print(f"    WARNING: Found next observation at ('next', 'agents', 'patrollers', 'observation'). FlattenObservation likely FAILED on next state. Shape: {next_td_output_from_transformed['agents', 'patrollers', 'observation'].shape}")
                else:
                    print("    ERROR: 'next' TD missing 'observation_flat' and original nested obs!")

                if "done" in next_td_output_from_transformed.keys() and "terminated" in next_td_output_from_transformed.keys():
                 print(f"    Found 'next' terminal flags ('next', 'done', 'next', 'terminated', 'next', 'truncated'). Shapes: {next_td_output_from_transformed['done'].shape}, {next_td_output_from_transformed['terminated'].shape}, {next_td_output_from_transformed['truncated'].shape}")
                else: print("    WARNING: 'next' TD missing terminal flags.")


            else: print("  MISSING root key 'next' from TransformedEnv.step() output!")


            transformed_env_test.close()
            print("\nTransformedEnv interaction test concluded.")


        except Exception as e:
            print(f"Step test failed: {type(e).__name__}: {e}")
            traceback.print_exc()
    except Exception as e:
         print("\n--- TransformedEnv Interaction Test Failed ---")
         traceback.print_exc()
         if 'transformed_env_test' in locals() and transformed_env_test: transformed_env_test.close()
         print("\n--- Proceeding with collector setup despite TransformedEnv test failure to gather more info ---")


    # --- Agent/Model Setup ---
    print("\n--- Agent/Model Setup ---")
    try:
        temp_env_for_specs = create_env_fn_transformed()
        
        # Get the specs after transformation
        transformed_obs_spec = temp_env_for_specs.observation_spec
        transformed_action_spec = temp_env_for_specs.action_spec
        
        class ModelSpecs:
            def __init__(self):
                # Observation spec should already be flattened
                self.observation_spec = transformed_obs_spec["observation_flat"]
                
                # Get action spec and ensure shape is preserved
                action_leaf = transformed_action_spec["agents", "patrollers", "action"]
                # Use TorchRL's Categorical instead of torch.distributions.Categorical
                self.action_spec = TorchRLCategorical(
                    n=action_leaf.space.n,
                    shape=action_leaf.shape,
                    dtype=action_leaf.dtype,
                    device=action_leaf.device
                )
                
                self.num_patrollers = temp_env_for_specs.base_env.num_patrollers
                self.batch_size = temp_env_for_specs.batch_size
    
        policy_module, value_module = create_ppo_models(env=ModelSpecs(), cfg=cfg.algo, device=DEVICE)
        temp_env_for_specs.close()

    except Exception as e:
         print(f"\nERROR during Agent/Model Setup: {type(e).__name__}: {e}")
         traceback.print_exc()
         sys.exit(1)


    # --- Replay Buffer, Loss, Optimizer Setup ---
    print("\n--- Replay Buffer, Loss, Optimizer Setup ---")
    try:
        minibatch_size = cfg.algo.minibatch_size
        replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=cfg.frames_per_batch), sampler=SamplerWithoutReplacement(), batch_size=minibatch_size)

        # Get action and value keys
        if isinstance(transformed_action_spec, CompositeSpec):
            policy_action_key = transformed_action_spec.keys(True, False)[0]
        else:
            policy_action_key = "action"
        
        policy_logprob_key = "sample_log_prob"  # This is standard for ProbabilisticActor
        value_key = "state_value"  # This is standard for ValueOperator

        # Configure loss module
        loss_module = ClipPPOLoss(
            actor=policy_module,
            critic=value_module,
            clip_epsilon=cfg.algo.clip_epsilon,
            entropy_coef=cfg.algo.entropy_coef,
            value_loss_coef=cfg.algo.value_loss_coef,
            loss_critic_type="l2"
        )
        
        # Set up GAE
        loss_module.make_value_estimator(
            ValueEstimators.GAE,
            gamma=cfg.algo.gamma,
            lmbda=cfg.algo.gae_lambda
        )
        
        # Set keys for value estimation
        loss_module.value_estimator.set_keys(value=value_key)
        
        # Set keys for PPO loss
        loss_module.set_keys(
            action=policy_action_key,
            sample_log_prob=policy_logprob_key,
            value=value_key,
            value_target="value_target",
            advantage="advantage"
        )
        
        print(f"Loss module configured with keys: action='{policy_action_key}', logprob='{policy_logprob_key}', value='{value_key}'")

        optimizer = optim.Adam(loss_module.parameters(), lr=cfg.algo.lr)
        print("Optimizer configured.")

    except Exception as e:
         print(f"\nERROR during Buffer/Loss/Optimizer Setup: {type(e).__name__}: {e}")
         traceback.print_exc()
         sys.exit(1)


    # --- Data Collector Setup ---
    print("\n--- Data Collector Setup ---")
    try:
        # The policy is provided here. Collector will use it to get actions during collection.
        # The collector expects policy input based on TransformedEnv.observation_spec (e.g. observation_flat).
        # Policy is a TensorDictModule that knows how to read its input_keys ('observation_flat').
        collector = SyncDataCollector(
            create_env_fn_transformed,
            policy=policy_module,
            frames_per_batch=cfg.frames_per_batch,
            total_frames=cfg.total_frames,
            policy_device=DEVICE,
            device=DEVICE,
            exploration_type=ExplorationType.RANDOM # Use Policy exploration type
        )
        print(f"SyncDataCollector created with policy: {policy_module}")
        print(f"Collector policy in_keys: {policy_module.in_keys}. Policy out_keys: {policy_module.out_keys}")

    except Exception as e:
         print(f"\nERROR during Data Collector Setup: {type(e).__name__}: {e}")
         traceback.print_exc()
         sys.exit(1)


    # --- Training loop ---
    print("\n--- Starting Training Loop ---")
    pbar = tqdm.tqdm(total=cfg.total_frames, desc="Training Progress")
    total_frames_processed = 0
    overall_start_time = time.time()

    try:
        for batch_data in collector:
            batch_loop_start_time = time.time()
            current_batch_frames = batch_data.numel()

            # batch_data collected includes state, action, log_prob, reward, done, next_state etc.
            # Keys depend on TransformedEnv output and Collector's handling.
            # Based on previous env test logs, collector output should contain root reward/done from step output
            # And a nested 'next' TD which includes next_obs.

            # It seems collector might ALSO add state_value at the root based on policy's value_network (if exists).
            # If policy is just actor, need to compute value here. Our policy is just actor.

            print(f"\nCollected Batch Size: {current_batch_frames}. Collected Batch Keys (incl nested): {batch_data.keys(True, True)}. Batch size: {batch_data.batch_size}")

            try:
                # Compute state_value for the batch and the next state using the value module
                # Value module reads 'observation_flat'. Collector places it at root and under 'next'.
                with torch.no_grad():
                    if value_key not in batch_data.keys():
                        print(f"Value key '{value_key}' not in batch_data. Computing value for current state.")
                        # This calls value_module(batch_data) which looks for 'observation_flat' in batch_data
                        value_module(batch_data)
                        print(f"Batch keys after current state value computation: {batch_data.keys()}")
                    else: print(f"Value key '{value_key}' already in batch_data (from policy/collector?). Skipping recomputation.")

                    # Compute value for the next state (needed for GAE bootstrapping)
                    next_td = batch_data.get("next", None)
                    if next_td is not None:
                         # Ensure next_td contains the observation key expected by value_module
                         if 'observation_flat' not in next_td.keys():
                              print(f"WARNING: next TD missing '{value_module.in_keys[0]}'. GAE next value might fail. Keys in next TD: {next_td.keys(True, True)}")

                         # Compute value for the next state (reads 'observation_flat' from next_td)
                         # Result is added to the next_td (e.g., next.state_value)
                         value_module(next_td)
                         print(f"'next' TD keys after next state value computation: {next_td.keys(True,True)}")

                    else:
                         print("WARNING: Collected batch data does not contain a 'next' TD for GAE next value computation.")


                    # Compute GAE: calculates advantages and value targets
                    # Uses: value (root), next_value (next.state_value), reward (root), done (root), etc.
                    # Expects these keys according to set_keys calls and GAE's default assumptions on structure
                    loss_module.value_estimator(batch_data)
                    print(f"Batch keys after GAE computation: {batch_data.keys()}")

                    assert "advantage" in batch_data.keys() and "value_target" in batch_data.keys(), "GAE failed to add 'advantage' or 'value_target'!"

            except Exception as e:
                 pbar.close(); print(f"\nERROR during GAE calculation: {type(e).__name__}: {e}"); traceback.print_exc(); break


            # --- PPO Optimization ---
            # Add the batch data (now with state_value, advantage, value_target) to the replay buffer
            # SamplerWithoutReplacement means we train on the entire batch, broken into minibatches
            replay_buffer.extend(batch_data.cpu()) # Clone to CPU for buffer

            batch_loss_sum = 0.0
            update_count = 0

            # Iterate through epochs
            for epoch in range(cfg.algo.ppo_epochs):
                # Iterate through minibatches from the buffer
                # SamplerWithoutReplacement drains the buffer once per epoch
                try:
                    for minibatch_data in replay_buffer: # Sampler yields minibatches
                        minibatch_data = minibatch_data.to(DEVICE)

                        # Minibatch TD has keys needed for loss at root: action, sample_log_prob, state_value, advantage, value_target
                        # The actor/critic modules will find 'observation_flat' in this minibatch TD when they are called by loss_module
                        # print(f"    Epoch {epoch+1}. Minibatch size: {minibatch_data.numel()}. Keys: {minibatch_data.keys()}")

                        try:
                            loss_dict = loss_module(minibatch_data) # Compute PPO loss components

                            valid_loss_tensors = [v for k, v in loss_dict.items() if isinstance(v, torch.Tensor) and v.requires_grad and v.numel() == 1]
                            if not valid_loss_tensors:
                                 print(f"    WARNING: No valid loss tensors requiring grad in loss_dict for minibatch. Skipping update. Loss dict: {loss_dict}")
                                 continue

                            total_loss = sum(valid_loss_tensors)

                            optimizer.zero_grad()
                            total_loss.backward()
                            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=cfg.algo.max_grad_norm)
                            optimizer.step()

                            batch_loss_sum += total_loss.item()
                            update_count += 1
                            # print(f"    Minibatch update successful. Total Loss: {total_loss.item():.4f}")

                        except Exception as e:
                             print(f"\n    ERROR during PPO minibatch update ({epoch+1}): {type(e).__name__}: {e}");
                             traceback.print_exc();
                             raise # Break epoch loop


                except StopIteration:
                     # This occurs when SamplerWithoutReplacement is exhausted
                     pass

                except Exception as e:
                     # Catch errors from epoch loop
                     print(f"ERROR during PPO epoch {epoch+1}: {type(e).__name__}: {e}");
                     traceback.print_exc();
                     raise # Break training loop


            # After epochs, clear buffer and update metrics
            if update_count == 0:
                 print("\nWARNING: No optimizer updates performed this batch cycle.")
                 avg_loss_this_batch = float('nan')
            else:
                avg_loss_this_batch = batch_loss_sum / update_count

            replay_buffer.empty() # Clear buffer for next collected batch

            # Update progress bar and log metrics
            total_frames_processed += current_batch_frames
            pbar.update(current_batch_frames)

            # Report average reward (from collected batch data)
            # Assuming reward is at ('next', 'reward') as seen in env test log structure after step.
            try:
                 avg_reward_batch_td = batch_data.get(("next", "reward"), None)
                 if avg_reward_batch_td is not None:
                      if avg_reward_batch_td.ndim > 1: avg_reward_batch_td = avg_reward_batch_td.mean(dim=[0,1])
                      elif avg_reward_batch_td.ndim == 1: avg_reward_batch_td = avg_reward_batch_td.mean(dim=0)
                      avg_reward_item = avg_reward_batch_td.item()
                 else:
                      avg_reward_item = float('nan')

            except Exception as e:
                 print(f"ERROR calculating average reward from batch: {type(e).__name__}: {e}")
                 traceback.print_exc()
                 avg_reward_item = float('nan')

            batch_elapsed = time.time() - batch_loop_start_time
            fps = current_batch_frames / max(1e-6, batch_elapsed)

            pbar.set_description(f"FPS: {fps:.0f}, Reward: {avg_reward_item:.3f}, Loss: {avg_loss_this_batch:.4f}")

        # End of `for batch_data in collector` loop
        print("\nCollector loop finished.")

    except Exception as e:
        print(f"\nERROR during Training Loop: {type(e).__name__}: {e}");
        traceback.print_exc();

    # Final Cleanup
    pbar.close()
    print("\nShutting down collector...")
    collector.shutdown()

    print(f"\nTraining completed. Total frames: {total_frames_processed}")
    print(f"Total time: {(time.time() - overall_start_time):.1f}s")

if __name__ == "__main__":
    manual_cfg = SimpleConfig()
    main(manual_cfg)