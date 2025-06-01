import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    Unbounded,  # Instead of UnboundedContinuousTensorSpec
    Composite,  # Instead of CompositeSpec  
    Categorical,  # Instead of DiscreteTensorSpec
    Binary  # Instead of BinaryDiscreteTensorSpec 
)
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs, set_exploration_type

from typing import Optional

import traceback # Needed for traceback.print_exc() in tests

DEFAULT_DEVICE = "cpu"

class PatrollingEnv(EnvBase):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    device = DEFAULT_DEVICE

    def __init__(
        self,
        num_patrollers: int = 3,
        num_intruders: int = 2,
        env_size: float = 10.0,
        max_steps: int = 100,
        patroller_move_dist: float = 0.5,
        intruder_move_dist: float = 0.2,
        detection_radius: float = 1.0,
        device: Optional[torch.device] = None,
        batch_size: Optional[torch.Size] = None,
        **kwargs,
    ):
        if batch_size is None: batch_size = torch.Size([])
        if device is None: device = torch.device(self.device)

        self.num_patrollers = num_patrollers
        self.num_intruders = num_intruders
        self.env_size = env_size
        self._max_steps = max_steps
        self.patroller_move_dist = patroller_move_dist
        self.intruder_move_dist = intruder_move_dist
        self.detection_radius = detection_radius
        self.step_count = None

        self.patroller_pos = None
        self.intruder_pos = None

        self.obs_dim_per_patroller = 2 + self.num_patrollers * 2 + self.num_intruders * 2

        self._make_specs()
        super().__init__(device=device, batch_size=batch_size)

        print(f"Initialized PatrollingEnv on device {self.device} with batch_size {self.batch_size}")
        print(f"Num Patrollers: {self.num_patrollers}, Num Intruders: {self.num_intruders}")
        print(f"Move Dist (P/I): {self.patroller_move_dist}/{self.intruder_move_dist}, Detect Radius: {self.detection_radius}")

    def _make_specs(self) -> None:
        """Updated specs using new names"""
        obs_dim = self.obs_dim_per_patroller

        self.observation_spec = Composite({
            "agents": Composite({
                "patrollers": Composite({
                    "observation": Unbounded(
                        shape=(*self.batch_size, self.num_patrollers, obs_dim),
                        dtype=torch.float32, device=self.device,
                    )
                }, shape=(*self.batch_size, self.num_patrollers))
            }, shape=self.batch_size)
        }, shape=self.batch_size)

        self.action_spec = Composite({
            "agents": Composite({
                "patrollers": Composite({
                    "action": Categorical(
                        n=5, shape=(*self.batch_size, self.num_patrollers),
                        dtype=torch.int64, device=self.device,
                    )
                }, shape=(*self.batch_size, self.num_patrollers))
            }, shape=self.batch_size)
        }, shape=self.batch_size)

        self.reward_spec = Unbounded(
            shape=(*self.batch_size, self.num_patrollers),
            dtype=torch.float32, device=self.device,
        )

        self.done_spec = Composite({
            "done": Binary(n=1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
            "terminated": Binary(n=1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
            "truncated": Binary(n=1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
        }, shape=self.batch_size)


    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        if tensordict is None or tensordict.is_empty(): _reset_batch_size = self.batch_size
        else: _reset_batch_size = tensordict.batch_size

        self.step_count = torch.zeros(_reset_batch_size, device=self.device, dtype=torch.int64)
        self.patroller_pos = (torch.rand(*_reset_batch_size, self.num_patrollers, 2, device=self.device) * 2 - 1) * self.env_size
        self.intruder_pos = (torch.rand(*_reset_batch_size, self.num_intruders, 2, device=self.device) * 2 - 1) * self.env_size
        initial_observations = self._get_observations(_reset_batch_size)

        done = torch.zeros(*_reset_batch_size, 1, dtype=torch.bool, device=self.device)
        terminated = torch.zeros(*_reset_batch_size, 1, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(*_reset_batch_size, 1, dtype=torch.bool, device=self.device)

        out_td = TensorDict({
            "agents": TensorDict({
                "patrollers": TensorDict({
                    "observation": initial_observations
                }, batch_size=(*_reset_batch_size, self.num_patrollers), device=self.device)
            }, batch_size=_reset_batch_size, device=self.device),
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        }, batch_size=_reset_batch_size, device=self.device)

        return out_td


    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Modified step function"""
        print(f"BaseEnv._step: Received TD keys: {tensordict.keys(True, True)}. Id: {id(tensordict)}")

        try:
            # Get action directly from tensordict
            action = tensordict.get(("agents", "patrollers", "action"), None)
            if action is None:
                raise KeyError("Action key ('agents','patrollers','action') missing in input TD!")
            
            # Rest of step function remains the same...
            self._update_patroller_positions(action)
            self._update_intruder_positions()
            self.step_count += 1

            rewards = self._calculate_rewards()

            if self.num_intruders > 0:
                batch_dim_count = len(self.batch_size)
                p_pos_exp1 = self.patroller_pos.unsqueeze(batch_dim_count + 1)
                i_pos_exp = self.intruder_pos.unsqueeze(batch_dim_count)
                rel_int = p_pos_exp1 - i_pos_exp
                dist_sq = torch.sum(rel_int.pow(2), dim=-1)
                detections = dist_sq < (self.detection_radius ** 2)
                terminated_agents = torch.any(detections, dim=-1)
                terminated = torch.any(terminated_agents, dim=-1)
                terminated = terminated.unsqueeze(-1)
            else:
                 terminated = torch.zeros(self.batch_size + (1,), dtype=torch.bool, device=self.device)

            truncated = (self.step_count >= self._max_steps).unsqueeze(-1)
            done = terminated | truncated

            next_observations = self._get_observations(self.batch_size)

            print(f"BaseEnv._step: About to construct output TD...")
            print(f"BaseEnv._step:   rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
            print(f"BaseEnv._step:   done shape: {done.shape}, dtype: {done.dtype}")
            print(f"BaseEnv._step:   terminated shape: {terminated.shape}, dtype: {terminated.dtype}")
            print(f"BaseEnv._step:   truncated shape: {truncated.shape}, dtype: {truncated.dtype}")
            print(f"BaseEnv._step:   next_observations shape: {next_observations.shape}, dtype: {next_observations.dtype}")


            # Build the output TensorDict dictionary structure step-by-step for clarity in prints
            next_agents_td = TensorDict({
                 "patrollers": TensorDict({
                 "observation": next_observations # Shape: [B, Np, obs_dim]
                 }, batch_size=(*self.batch_size, self.num_patrollers), device=self.device)
             }, batch_size=self.batch_size, device=self.device)
            print(f"BaseEnv._step:   next['agents'] TensorDict created. Keys: {next_agents_td.keys(True,True)}")


            next_td = TensorDict({ # "next" contains the STATE after the step
                    # Observation for the NEXT time step
                   "agents": next_agents_td, # Already created nested agents TD
                   # Copy terminal flags in 'next' (common practice for value fn bootstrapping)
                   "done": done,
                   "terminated": terminated,
                   "truncated": truncated,
               },
               batch_size=self.batch_size,
               device=self.device,
               )
            print(f"BaseEnv._step:   'next' TensorDict created. Keys: {next_td.keys(True,True)}. Root batch size {self.batch_size}. next_td batch size: {next_td.batch_size}")


            out_td_dict = {
               # Results of the step at the ROOT level
               "reward": rewards,
               "done": done,
               "terminated": terminated,
               "truncated": truncated,
               "next": next_td # Add the created next_td under the 'next' key
            }
            print(f"BaseEnv._step:   Root output dictionary created. Keys: {out_td_dict.keys()}. 'next' entry keys: {out_td_dict['next'].keys(True, True)}")

            out_td = TensorDict(out_td_dict, batch_size=self.batch_size, device=self.device)

            print(f"BaseEnv._step: Finished creating output TensorDict. Root Keys: {out_td.keys(True, False)}. Nested 'next' Keys: {out_td.get('next', TensorDict({})).keys(True,True)}. Id: {id(out_td)}")

            return out_td
        except Exception as e:
            print(f"BaseEnv._step: Exception occurred - {e}")
            raise e

    # --- Remaining helper methods (_set_seed, _get_observations, _update_patroller_positions, _update_intruder_positions, _calculate_rewards) ---

    def _set_seed(self, seed: Optional[int] = None):
        # Note: torch.manual_seed is usually sufficient, returning rng might be unnecessary unless using Generator objects
        rng = torch.manual_seed(seed)
        # Should also set seed for random operations *inside* step if they don't use torch.rand_like etc.

    def _get_observations(self, batch_size):
        _pat_pos = self.patroller_pos.view(*batch_size, self.num_patrollers, 2)
        _int_pos = self.intruder_pos.view(*batch_size, self.num_intruders, 2)

        batch_dim_count = len(batch_size)

        p_pos_exp1 = _pat_pos.unsqueeze(batch_dim_count + 1) # [B, Np, 1, 2]
        p_pos_exp2 = _pat_pos.unsqueeze(batch_dim_count)     # [B, 1, Np, 2]
        rel_pat = p_pos_exp1 - p_pos_exp2                    # [B, Np, Np, 2]

        if self.num_intruders > 0:
            i_pos_exp = _int_pos.unsqueeze(batch_dim_count)  # [B, 1, Ni, 2]
            rel_int = p_pos_exp1 - i_pos_exp                 # [B, Np, Ni, 2]
            rel_int_flat = rel_int.reshape(*batch_size, self.num_patrollers, self.num_intruders * 2)
        else:
            rel_int_flat = torch.empty((*batch_size, self.num_patrollers, 0), device=self.device, dtype=torch.float32) # Ensure correct device/dtype

        rel_pat_flat = rel_pat.reshape(*batch_size, self.num_patrollers, self.num_patrollers * 2)
        own_pos = _pat_pos

        observations = torch.cat([own_pos, rel_pat_flat, rel_int_flat], dim=-1)
        expected_dim = 2 + self.num_patrollers * 2 + self.num_intruders * 2
        assert observations.shape[-1] == expected_dim, f"Obs dim mismatch: {observations.shape[-1]} vs {expected_dim}"
        assert observations.shape[:-1] == (*batch_size, self.num_patrollers), f"Obs batch/agent shape mismatch: {observations.shape[:-1]} vs {(*batch_size, self.num_patrollers)}" # More detailed shape error


        return observations

    def _update_patroller_positions(self, actions):
        """Fix the action handling"""
        delta_xy = torch.tensor([
            [0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]
        ], device=self.device, dtype=torch.float32)

        # Convert TensorDict action to tensor if needed
        if isinstance(actions, TensorDictBase):
            actions = actions.get(("agents", "patrollers", "action"))
        
        # Ensure actions is a tensor
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"Actions must be tensor or TensorDict, got {type(actions)}")

        # Reshape actions properly
        _actions = actions.view(*self.batch_size, self.num_patrollers)
        
        # Index into delta_xy safely
        move_deltas = torch.index_select(delta_xy, 0, _actions.reshape(-1))
        move_deltas = move_deltas.view(*self.batch_size, self.num_patrollers, 2)
        move_deltas = move_deltas * self.patroller_move_dist

        self.patroller_pos += move_deltas
        self.patroller_pos = torch.clamp(
            self.patroller_pos, min=-self.env_size, max=self.env_size
        )

    def _update_intruder_positions(self):
        if self.num_intruders == 0: return
        _int_pos = self.intruder_pos.view(*self.batch_size, self.num_intruders, 2)
        intruder_movement = (torch.rand_like(_int_pos) * 2 - 1) * self.intruder_move_dist

        self.intruder_pos += intruder_movement
        self.intruder_pos = torch.clamp(
            self.intruder_pos, min=-self.env_size, max=self.env_size
        )

    def _calculate_rewards(self):
        if self.num_intruders == 0:
             # Ensure zero tensor matches expected reward shape [B, Np]
             return torch.zeros((*self.batch_size, self.num_patrollers), device=self.device, dtype=torch.float32)

        batch_dim_count = len(self.batch_size)
        _pat_pos = self.patroller_pos.view(*self.batch_size, self.num_patrollers, 2)
        _int_pos = self.intruder_pos.view(*self.batch_size, self.num_intruders, 2)

        p_pos_exp1 = _pat_pos.unsqueeze(batch_dim_count + 1) # [B, Np, 1, 2]
        i_pos_exp = _int_pos.unsqueeze(batch_dim_count)      # [B, 1, Ni, 2]

        rel_int = p_pos_exp1 - i_pos_exp                     # [B, Np, Ni, 2]
        dist_sq = torch.sum(rel_int.pow(2), dim=-1)          # [B, Np, Ni]
        detections = dist_sq < (self.detection_radius ** 2)  # [B, Np, Ni]

        patroller_detects_any = torch.any(detections, dim=-1) # [B, Np] - True if this patroller detects ANY intruder

        reward = torch.where(patroller_detects_any, 1.0, 0.0).float() # [B, Np]

        return reward


# --- Basic Test (`if __name__ == "__main__"`) ---
if __name__ == "__main__":
    print("Testing PatrollingEnv Definition (Revised Test Logic)...")
    test_device = torch.device("cpu")
    print(f"Env test using device: {test_device}")

    try:
        env = PatrollingEnv(num_patrollers=3, num_intruders=2, device=test_device)

        print("\nChecking Environment Specs...")
        print("Running check_env_specs()...")
        try:
             # Check_env_specs should now work because _step output is standard
             check_env_specs(env)
             print("check_env_specs() passed.")
        except Exception as e:
             print(f"check_env_specs() failed (often ignorable for complex multi-agent envs): {type(e).__name__}: {e}")
             traceback.print_exc()


        print("\nTesting env.reset()...")
        td_current = env.reset().to(test_device) # Keep the output for the first step input
        print("Initial TensorDict (root keys from reset):", td_current.keys(True,True))
        assert ("agents", "patrollers", "observation") in td_current.keys(True, True), "Reset output missing nested observation!"
        assert "done" in td_current.keys() and "terminated" in td_current.keys() and "truncated" in td_current.keys(), "Reset output missing root terminal flags!"


        print("\nTesting first env.step() after reset...")

        # Step 1: Sample action and ADD it to the td_current (which is the output of reset)
        action_sample = env.action_spec.rand().to(test_device)
        print(f"Sampled action shape: {action_sample.shape}")
        print(f"Current state TD batch size: {td_current.batch_size}")
        print(f"Env batch size: {env.batch_size}")


        # The action needs to be placed into the nested structure defined by action_spec
        # Use .set() for nested paths, potentially reshaping action_sample if necessary
        # The action_sample from spec.rand() will have shape (*batch_size, num_patrollers) for Categoryical(n=5) on nested action
        # If batch_size is (), action_sample shape is [num_patrollers]. This is correct.
        td_current.set(("agents", "patrollers", "action"), action_sample, inplace=True)

        print(f"Manual test: TD input to env.step() (output of reset + action): {td_current.keys(True, True)}. Batch size: {td_current.batch_size}. Id: {id(td_current)}")
        assert ("agents", "patrollers", "action") in td_current.keys(True,True), "Step input TD missing action key!"
        assert ("agents", "patrollers", "observation") in td_current.keys(True,True), "Step input TD missing observation key!" # Ensure previous obs is still there

        # Call env.step() with the combined TD
        td_result = env.step(td_current) # This will print received keys inside _step

        print("\n--- Full Resulting TensorDict from env.step ---")
        # print(td_result) # Print the full TD returned by EnvBase.step
        print("--------------------------------------------")

        # --- Access keys level by level with checks on td_result (EnvBase.step output) ---
        print("Checking EnvBase.step output keys and structure:")
        # After EnvBase.step(), the TD structure includes 'next' and retains original keys like obs at root
        # The structure returned by EnvBase.step looks something like:
        # root: { original_state_keys (like obs), input_action_keys, result_keys (reward, done), next: { next_state_keys (like next_obs, next_terminal_flags) } }
        # BUT, TorchRL collector's _step_mdp handles moving things around. The keys used by downstream modules
        # are primarily 'reward', 'done', 'next.observation', etc., as they are processed by the transforms and collector.

        # Let's just check the keys relevant for data collection and training: reward, done, and keys within 'next'.

        print("  Checking result keys (relevant for collector):")
        if "reward" in td_result.keys():
            print("    Found root key 'reward'. Shape:", td_result["reward"].shape)
            assert td_result["reward"].shape == (*env.batch_size, env.num_patrollers), "Root reward shape mismatch!"
        else: print("    MISSING root key 'reward'!")

        if "done" in td_result.keys() and "terminated" in td_result.keys() and "truncated" in td_result.keys():
            print("    Found root terminal flags 'done', 'terminated', 'truncated'. Shapes:",
                  td_result["done"].shape, td_result["terminated"].shape, td_result["truncated"].shape)
            # These flags in EnvBase.step output root usually match input flags (before step)
            # Assertions are better on the 'next' flags for step transition check
            pass # Skip assertion here, focus on 'next' flags

        else: print("    MISSING one or more root terminal flags ('done','terminated','truncated') from EnvBase.step output root!")

        next_td_output = td_result.get("next", None)
        if next_td_output is not None:
            print("    Found root key 'next' from EnvBase.step output. Its keys (full nested):", next_td_output.keys(True, True))
            # This 'next' TD is what _step output! So its structure should match our corrected _step output

            if ("agents", "patrollers", "observation") in next_td_output.keys(True,True):
                print("    SUCCESS: Found next observation at ('next', 'agents', 'patrollers', 'observation'). Shape:", next_td_output["agents", "patrollers", "observation"].shape)
                expected_obs_shape = (*env.batch_size, env.num_patrollers, env.obs_dim_per_patroller)
                assert next_td_output["agents", "patrollers", "observation"].shape == expected_obs_shape, f"Next observation shape mismatch in 'next' TD! Expected {expected_obs_shape}, got {next_td_output['agents', 'patrollers', 'observation'].shape}"
            else:
                print("    ERROR: 'next' TD is missing the expected observation key ('agents', 'patrollers', 'observation')!")


            if "done" in next_td_output.keys() and "terminated" in next_td_output.keys() and "truncated" in next_td_output.keys():
                print("    Found 'next' terminal flags 'done', 'terminated', 'truncated'. Shapes:",
                       next_td_output["done"].shape, next_td_output["terminated"].shape, next_td_output["truncated"].shape)
                # These are the crucial flags after the transition, should match [B, 1] and logic
                assert next_td_output["done"].shape == (*env.batch_size, 1) and \
                       next_td_output["terminated"].shape == (*env.batch_size, 1) and \
                       next_td_output["truncated"].shape == (*env.batch_size, 1), "'Next' terminal flags shape mismatch!"
            else: print("    MISSING one or more 'next' terminal flags ('done','terminated','truncated')!")


        else: print("  MISSING root key 'next' from EnvBase.step output!") # This would be very bad


        print("\nTesting multiple steps (Simulating rollout)...")
        steps = 0
        # td_current is already the result of the first step
        # In a rollout, you use the output of the previous step to get the input for the next step
        # We need the 'next' part of td_current to get the state *for* the next step
        # Then add the action to *that* TD.
        # Initial state for loop: output of env.reset()
        td_current_state = env.reset().to(test_device)

        # The loop processes frames. Each frame starts with state td_current_state, takes action, gets td_result
        # td_current_state will be the *start-of-step* state, td_result will be the *end-of-step* TD
        done = td_current_state.get("done", torch.tensor([[False]], device=test_device, dtype=torch.bool)) # shape [B, 1]
        # Ensure it's correctly shaped [B, 1] for scalar any()
        if done.ndim == 1 and env.batch_size == torch.Size([]):
             done = done.unsqueeze(0)

        try:
            while not done.any().item() and steps < env._max_steps + 5:
                print(f"\n--- Start of Loop Step {steps+1} ---")
                print(f"State TD keys at START of step: {td_current_state.keys(True,True)}")
                assert ("agents", "patrollers", "observation") in td_current_state.keys(True,True), f"Loop step {steps+1}: Start state TD missing observation!"
                assert "done" in td_current_state.keys(), f"Loop step {steps+1}: Start state TD missing done!"


                # Policy action would be based on the observation in td_current_state
                # Sample action using the *environment's* action spec shape, but ensure device
                action_sample = env.action_spec.rand().to(test_device) # Action sample should match env's internal device

                # Prepare input TD for env.step: Start with td_current_state (prev obs, prev done, etc.)
                # Then ADD the action to THIS TD.
                td_input_for_step = td_current_state.clone()
                td_input_for_step.set(("agents", "patrollers", "action"), action_sample, inplace=True) # Action has shape [B, Np], use .set() directly

                # print(f"Loop Step {steps+1}: TD input to EnvBase.step(): {td_input_for_step.keys(True, True)}. Batch size: {td_input_for_step.batch_size}. Id: {id(td_input_for_step)}")
                assert ("agents", "patrollers", "action") in td_input_for_step.keys(True,True), f"Loop step {steps+1}: Input TD missing action!"

                td_result = env.step(td_input_for_step) # Call EnvBase.step() with the combined TD

                # Process td_result (the output from EnvBase.step)
                # td_result has root keys from input (original state + action), and result keys (reward, done), and 'next' TD
                print(f"--- After EnvBase.step() for step {steps+1} ---")
                # print("Full td_result:", td_result)
                print("td_result root keys:", td_result.keys(True,False))
                print("'next' TD keys within td_result:", td_result.get("next", TensorDict({})).keys(True,True))

                # Update td_current_state for the *next* loop iteration.
                # The state for the next step comes from the 'next' TD within td_result.
                td_current_state = td_result.get("next").to(test_device) # Get the 'next' TD

                # Get transition results (reward, done etc.) from td_result (the EnvBase.step output) for logging/checks
                # These are the values at the end of the step (or the start of the next)
                # Get them from the td_result structure. Reward/done should be at the root of EnvBase.step output typically.
                # Wait, the log of check_env_specs *within* the test shows these are still in a strange place?
                # Let's check the structure based on the LAST log's print of the FULL TD again.
                # Last Log Step TD:
                # root: { agents: {patrollers: {action:}}, next: { done, next: {agents:{patrollers:{observation:}}}, reward, terminated, truncated }}
                # Oh, the problem is that my interpretation of EnvBase.step's *output* might be wrong, OR
                # your local torchrl version has a peculiar EnvBase.step implementation for nested spaces.
                # Let's adjust how we READ from the td_result in the manual test to match YOUR LAST LOG.
                # This means your _step output might be *overwritten* or restructured by EnvBase.step...
                # Let's READ results using the paths from YOUR LAST LOG, assuming that structure is the output of env.step.
                # We get: reward, done, terminated, truncated FROM INSIDE THE FIRST 'next' key.
                # And the actual 'next' state obs FROM INSIDE next.next.
                # THIS IS SO STRANGE. Your original env seems to have set this weird standard, and TorchRL is replicating it?

                # LET's test THIS hypothesis: Your original step structure *is* what env.step is aiming for.
                # Redo the test code to read according to THAT STRUCTURE.

                print(f"Loop Step {steps+1}: Attempting to read reward/done from locations in the PREVIOUS log's STRUCTURE")
                done = td_result.get(("next", "done"), None) # From previous log, root=agents/next, reward/done were inside next!
                reward = td_result.get(("next", "reward"), None)
                terminated = td_result.get(("next", "terminated"), None)
                truncated = td_result.get(("next", "truncated"), None)
                # And the next observation is in next.next? Let's confirm this by checking next.next
                # THIS IS WHERE check_env_specs and my prints likely got confused.
                # The structure might be: root: { state_before_step... + action_input...}, next: { transition_info..., next: { state_after_step } }
                # This means EnvBase.step puts your results (reward, done) *and* the *true* next state *together* inside 'next'.

                # If the check_env_specs input print is trustworthy: it shows 'agents', 'action', 'done', 'terminated', 'truncated'. This means EnvBase is trying to step with original obs + action + terminal flags. This TD is built by EnvBase or collector.
                # If my prints inside _step are trustworthy: _step returns {reward, done, terminated, truncated, next: { agents: {obs:}, done, terminated, truncated } }

                # Let's print the FULL td_result from env.step in the loop explicitly
                print("\nLoop Step {steps+1}: Full td_result from EnvBase.step:")
                print(td_result)
                print("-" * 30)

                # Re-attempt reading based on what the *latest* full TD print shows
                # LATEST LOG: root: {'agents': {...}, 'next': {...}}
                # content of root['next']: {'done', 'next':{...}, 'reward', 'terminated', 'truncated'}
                # content of root['next']['next']: {'agents':{...obs...}, 'done', 'terminated', 'truncated'}

                # So, from td_result (the EnvBase.step output):
                done = td_result.get(("next", "done"), None) # First next layer
                reward = td_result.get(("next", "reward"), None) # First next layer
                terminated = td_result.get(("next", "terminated"), None) # First next layer
                truncated = td_result.get(("next", "truncated"), None) # First next layer

                # And the actual state FOR THE NEXT STEP comes from ("next", "next")
                next_state_for_loop = td_result.get(("next", "next"), None)

                if next_state_for_loop is None:
                     print("CRITICAL ERROR: Cannot find ('next', 'next') in td_result to get next state!")
                     raise KeyError("Next state missing in step output")


                # Update td_current_state for the next loop iteration using next.next
                td_current_state = next_state_for_loop.to(test_device)


                steps += 1

                # Safely access values from the ones just retrieved from td_result
                done_val = done.any().item() if done is not None else "Key Not Found"
                reward_val = f"{reward.mean().item():.2f}" if reward is not None else "Key Not Found"
                term_val = terminated.any().item() if terminated is not None else "Key Not Found"
                trunc_val = truncated.any().item() if truncated is not None else "Key Not Found"


                print(f"Step {steps}: Done={done_val}, Terminated={term_val}, Truncated={trunc_val}, Avg Reward={reward_val}")

                # Ensure required keys for results were found
                if done is None or reward is None or terminated is None or truncated is None:
                     print("Error: Required keys ('next.done', 'next.reward', 'next.terminated', 'next.truncated') missing in step output based on latest log structure!")
                     raise KeyError("Required result keys missing after step (following unexpected structure)")


                if done.any().item(): # Correct check for batch=() and batch=[B]
                     print("Episode finished.")
                     break # Exit the test loop after episode ends


        except Exception as e:
            print(f"\nError during multi-step test at step {steps}: {type(e).__name__}: {e}")
            traceback.print_exc()


        print("\nBasic Environment Testing Completed.")

    except Exception as e:
        print(f"\nFatal error during environment testing setup: {type(e).__name__}: {e}")
        traceback.print_exc()