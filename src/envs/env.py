import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    UnboundedContinuousTensorSpec as Unbounded,
    CompositeSpec as Composite,
    DiscreteTensorSpec as Categorical,
    BinaryDiscreteTensorSpec as Binary,
)
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs, set_exploration_type

from typing import Optional

DEFAULT_DEVICE = "cpu"

class PatrollingEnv(EnvBase):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    device = DEFAULT_DEVICE

    # Inside the PatrollingEnv class:

    def __init__(
        self,
        num_patrollers: int = 3,
        num_intruders: int = 2,
        env_size: float = 10.0,
        max_steps: int = 100,
        patroller_move_dist: float = 0.5, # Added
        intruder_move_dist: float = 0.2,  # Added
        detection_radius: float = 1.0,    # Added
        device: Optional[torch.device] = None,
        batch_size: Optional[torch.Size] = None,
        **kwargs,
    ):
        if batch_size is None: batch_size = torch.Size([])
        if device is None: device = torch.device(self.device)

        # --- Core Parameters ---
        self.num_patrollers = num_patrollers
        self.num_intruders = num_intruders
        self.env_size = env_size
        self._max_steps = max_steps
        self.patroller_move_dist = patroller_move_dist # Store parameter
        self.intruder_move_dist = intruder_move_dist   # Store parameter
        self.detection_radius = detection_radius     # Store parameter
        self.step_count = None

        # --- Dynamic State ---
        self.patroller_pos = None
        self.intruder_pos = None

        # --- Specs & Superclass Init ---
        # Store obs_dim for use in _make_specs
        # Observation: own_pos (2) + rel_pat ((Np-1)*2) + rel_int (Ni*2) - This was the original MPE style
        # Let's use: own_pos (2) + rel_ALL_pat (Np*2) + rel_ALL_int (Ni*2)
        # Because calculating ALL relative positions is easier with broadcasting.
        # Note: rel_pat includes self-relative position (which is zero).
        self.obs_dim_per_patroller = 2 + self.num_patrollers * 2 + self.num_intruders * 2

        self._make_specs()
        super().__init__(device=device, batch_size=batch_size)

        print(f"Initialized PatrollingEnv on device {self.device} with batch_size {self.batch_size}")
        print(f"Num Patrollers: {self.num_patrollers}, Num Intruders: {self.num_intruders}")
        print(f"Move Dist (P/I): {self.patroller_move_dist}/{self.intruder_move_dist}, Detect Radius: {self.detection_radius}")

    def _make_specs(self) -> None:
        # Use the obs_dim calculated in __init__
        obs_dim = self.obs_dim_per_patroller

        # Observation Spec: Use key "observation"
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

        # Action Spec: Use key "action"
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

        # Reward Spec: Keep at root for simplicity
        self.reward_spec = Unbounded(
            shape=(*self.batch_size, self.num_patrollers),
            dtype=torch.float32, device=self.device,
        )

        # Done Spec: Stays the same
        self.done_spec = Composite({
            "done": Binary(n=1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
            "terminated": Binary(n=1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
            "truncated": Binary(n=1, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device),
        }, shape=self.batch_size)

    def _reset(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        if tensordict is None or tensordict.is_empty(): _reset_batch_size = self.batch_size
        else: _reset_batch_size = tensordict.batch_size

        self.step_count = torch.zeros(_reset_batch_size, device=self.device, dtype=torch.int64)
        # Ensure positions have the batch dimension, even if empty
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
                    # Shape: [B, Np, obs_dim]
                }, batch_size=(*_reset_batch_size, self.num_patrollers), device=self.device)
            }, batch_size=_reset_batch_size, device=self.device),
            "done": done, # Shape: [B, 1]
            "terminated": terminated, # Shape: [B, 1]
            "truncated": truncated, # Shape: [B, 1]
        }, batch_size=_reset_batch_size, device=self.device)
        return out_td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        actions = tensordict["agents", "patrollers", "action"].to(self.device)  

        self._update_patroller_positions(actions)
        self._update_intruder_positions()
        self.step_count += 1

        # Calculate reward based on the new state
        rewards = self._calculate_rewards() # Shape: [B, Np]

        # Calculate termination based on the new state
        if self.num_intruders > 0:
            batch_dim_count = len(self.batch_size)
            p_pos_exp1 = self.patroller_pos.unsqueeze(batch_dim_count + 1)
            i_pos_exp = self.intruder_pos.unsqueeze(batch_dim_count)
            rel_int = p_pos_exp1 - i_pos_exp
            dist_sq = torch.sum(rel_int.pow(2), dim=-1)
            detections = dist_sq < (self.detection_radius ** 2)
            terminated = torch.any(detections.view(*self.batch_size, -1), dim=-1) # Shape: [B]
        else:
            terminated = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device) # Shape: [B]

        truncated = (self.step_count >= self._max_steps) # Shape: [B]
        done = terminated | truncated # Shape: [B]

        # Calculate next observations based on the new state
        next_observations = self._get_observations(self.batch_size) # Shape: [B, Np, obs_dim]

        # Construct the output TensorDict following standard convention
        out_td = TensorDict({
                 "reward": rewards, # Shape: [B, Np]
                 "done": done.view(self.batch_size + (1,)), # Shape: [B, 1]
                 "terminated": terminated.view(self.batch_size + (1,)), # Shape: [B, 1]
                 "truncated": truncated.view(self.batch_size + (1,)), # Shape: [B, 1]
                 "next": TensorDict({ # "next" contains the state after the step
                         "agents": TensorDict({
                             "patrollers": TensorDict({
                                 "observation": next_observations # Shape: [B, Np, obs_dim]
                                 }, batch_size=(*self.batch_size, self.num_patrollers), device=self.device)
                             }, batch_size=self.batch_size, device=self.device),
                         # Done flags are also often included in "next" for convenience
                         "done": done.view(self.batch_size + (1,)),
                         "terminated": terminated.view(self.batch_size + (1,)),
                         "truncated": truncated.view(self.batch_size + (1,)),
                    }, batch_size=self.batch_size, device=self.device
                 ),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return out_td

    def _set_seed(self, seed: Optional[int] = None):
        rng = torch.manual_seed(seed)

    def _get_observations(self, batch_size):
        """
        Calculates observations for all patrollers based on current state.
        Observation: own_pos (2) + rel_ALL_pat (Np*2) + rel_ALL_int (Ni*2)
        """
        # Ensure positions have batch dim, even if B=[] -> shape [Np, 2] or [Ni, 2]
        # This was likely the issue: _reset created tensors with batch dim,
        # but subsequent accesses might not have had it if B=[]
        _pat_pos = self.patroller_pos.view(*batch_size, self.num_patrollers, 2)
        _int_pos = self.intruder_pos.view(*batch_size, self.num_intruders, 2)

        batch_dim_count = len(batch_size) # Use function arg batch_size

        p_pos_exp1 = _pat_pos.unsqueeze(batch_dim_count + 1) # [B, Np, 1, 2]
        p_pos_exp2 = _pat_pos.unsqueeze(batch_dim_count)     # [B, 1, Np, 2]
        rel_pat = p_pos_exp1 - p_pos_exp2                    # [B, Np, Np, 2]

        if self.num_intruders > 0:
            i_pos_exp = _int_pos.unsqueeze(batch_dim_count)  # [B, 1, Ni, 2]
            rel_int = p_pos_exp1 - i_pos_exp                 # [B, Np, Ni, 2]
            rel_int_flat = rel_int.reshape(*batch_size, self.num_patrollers, self.num_intruders * 2)
        else:
            rel_int_flat = torch.empty((*batch_size, self.num_patrollers, 0), device=self.device, dtype=torch.float32)

        rel_pat_flat = rel_pat.reshape(*batch_size, self.num_patrollers, self.num_patrollers * 2)
        own_pos = _pat_pos # Use view with batch dim

        observations = torch.cat([own_pos, rel_pat_flat, rel_int_flat], dim=-1)
        # Expected final shape: [B, Np, 2 + Np*2 + Ni*2]
        # Check against self.obs_dim_per_patroller
        expected_dim = 2 + self.num_patrollers * 2 + self.num_intruders * 2
        assert observations.shape[-1] == expected_dim, f"Obs dim mismatch: {observations.shape[-1]} vs {expected_dim}"
        assert observations.shape[:-1] == (*batch_size, self.num_patrollers), f"Obs batch/agent shape mismatch: {observations.shape[:-1]}"

        return observations

    def _update_patroller_positions(self, actions):
        """Updates patroller positions based on discrete actions."""
        # actions shape: [B, Np]
        delta_xy = torch.tensor([
            [0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]
        ], device=self.device, dtype=torch.float32)

        # Ensure actions have batch dim if B=[] -> shape [Np]
        _actions = actions.view(*self.batch_size, self.num_patrollers)
        move_deltas = delta_xy[_actions] # Shape: [B, Np, 2]

        self.patroller_pos += move_deltas * self.patroller_move_dist
        self.patroller_pos = torch.clamp(
            self.patroller_pos, min=-self.env_size, max=self.env_size
        )

    def _update_intruder_positions(self):
        """Updates intruder positions (simple random walk)."""
        if self.num_intruders == 0: return
        # Ensure intruder_pos has batch dim for rand_like
        _int_pos = self.intruder_pos.view(*self.batch_size, self.num_intruders, 2)
        intruder_deltas = (torch.rand_like(_int_pos) * 2 - 1) # Shape: [B, Ni, 2]

        self.intruder_pos += intruder_deltas * self.intruder_move_dist
        self.intruder_pos = torch.clamp(
            self.intruder_pos, min=-self.env_size, max=self.env_size
        )

    def _calculate_rewards(self):
        """Calculates rewards for patrollers based on detection."""
        if self.num_intruders == 0:
             return torch.zeros((*self.batch_size, self.num_patrollers), device=self.device, dtype=torch.float32)

        batch_dim_count = len(self.batch_size)
        # Ensure positions have batch dim
        _pat_pos = self.patroller_pos.view(*self.batch_size, self.num_patrollers, 2)
        _int_pos = self.intruder_pos.view(*self.batch_size, self.num_intruders, 2)

        p_pos_exp1 = _pat_pos.unsqueeze(batch_dim_count + 1) # [B, Np, 1, 2]
        i_pos_exp = _int_pos.unsqueeze(batch_dim_count)      # [B, 1, Ni, 2]

        rel_int = p_pos_exp1 - i_pos_exp                     # [B, Np, Ni, 2]
        dist_sq = torch.sum(rel_int.pow(2), dim=-1)          # [B, Np, Ni]
        detections = dist_sq < (self.detection_radius ** 2)  # [B, Np, Ni]

        patroller_detects_any = torch.any(detections, dim=-1) # [B, Np]
        reward = torch.where(patroller_detects_any, 1.0, 0.0).float() # [B, Np]

        return reward


# --- Basic Test (`if __name__ == "__main__"`) ---
# (Keep the __main__ block exactly as it was in the last successful run)
if __name__ == "__main__":
    print("Testing PatrollingEnv Definition...")
    env = PatrollingEnv(num_patrollers=3, num_intruders=2, device=DEFAULT_DEVICE)

    print("\nChecking Environment Specs...")
    print("Skipping check_env_specs() due to internal errors. Proceeding with manual tests.")

    print("\nTesting env.reset()...")
    initial_td = env.reset()
    print("Initial TensorDict (root keys):", initial_td.keys())
    print("Initial Patroller Obs Shape:", initial_td["agents", "patrollers", "observation"].shape)

    print("\nTesting env.step()...")
    action_sample = env.action_spec.rand()
    print(f"Type of action_sample from spec.rand(): {type(action_sample)}")
    if isinstance(action_sample, torch.Tensor):
        print("Action sample is a Tensor, wrapping it in TensorDict...")
        action_td = TensorDict({
            "agents": TensorDict({ "patrollers": TensorDict({ "action": action_sample },
                           batch_size=(*env.batch_size, env.num_patrollers), device=env.device)
            }, batch_size=env.batch_size, device=env.device)
        }, batch_size=env.batch_size, device=env.device)
    elif isinstance(action_sample, TensorDictBase):
         print("Action sample is already a TensorDict.")
         action_td = action_sample
    else: raise TypeError(f"Unexpected type from action_spec.rand(): {type(action_sample)}")

    step_td_result = env.step(action_td)

    print("\n--- Full Resulting TensorDict from env.step ---")
    print(step_td_result)
    print("--------------------------------------------")

    print("Resulting TensorDict from env.step (root keys):", step_td_result.keys()) # Should be ['agents', 'next']

    # --- Access using the CORRECT double-nested paths ---
    # Access the NEXT observation: result -> next -> next -> agents -> patrollers -> observation
    print("Next Patroller Obs Shape:", step_td_result["next", "next", "agents", "patrollers", "observation"].shape)
    # Access the reward: result -> next -> reward
    print("Patroller Reward Shape:", step_td_result["next", "reward"].shape)
    # Access done flag: result -> next -> done (or result -> next -> next -> done)
    print("Done Flag (root via next):", step_td_result["next", "done"])
    print("Done Flag (nested via next->next):", step_td_result["next", "next", "done"])


    # Test step multiple times - get done from the correct nested key
    print("\nTesting multiple steps...")
    steps = 0
    td_current = env.reset()
    # Reset doesn't have 'next', done is at root
    done = td_current.get("done", torch.tensor([[False]], device=env.device))
    while not done.any() and steps < 10:
        action_sample = env.action_spec.rand()
        # Wrap action
        if isinstance(action_sample, torch.Tensor):
            action_td_loop = TensorDict({
                "agents": TensorDict({"patrollers": TensorDict({ "action": action_sample },
                           batch_size=(*env.batch_size, env.num_patrollers), device=env.device)
                }, batch_size=env.batch_size, device=env.device)
            }, batch_size=env.batch_size, device=env.device)
        else: action_td_loop = action_sample

        td_result = env.step(action_td_loop)

        # Get done from the DOUBLE nested 'next' key
        done = td_result.get(("next", "next", "done"))
        # Get reward from the SINGLE nested 'next' key
        reward = td_result.get(("next", "reward"))

        steps += 1
        # Check if keys were found before trying .item()/.mean()
        done_val = done.item() if done is not None else "Key Not Found"
        reward_val = f"{reward.mean().item():.2f}" if reward is not None else "Key Not Found"
        print(f"Step {steps}: Done={done_val}, Reward={reward_val}")

        if done is None or reward is None:
             print("Error: Required keys ('next','next','done' or 'next','reward') not found!")
             break
        if done.any():
             print("Episode finished.")


    print("\nBasic Environment Testing Completed.")