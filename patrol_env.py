import numpy as np
from gymnasium.utils import EzPickle
from gymnasium.spaces import Box, Discrete

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.utils.conversions import parallel_wrapper_fn

from baseEnv.baseEnv import BaseEnv, make_env
from baseEnv.patrolWorld import PatrolWorld

import logging

logging.basicConfig(
    filename="patrol_env.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w",
)
logger = logging.getLogger(__name__)


class PatrolEnv(BaseEnv, EzPickle):
    """
    Custom multi-agent environment for patrollers and intruders using PettingZoo and Gymnasium.
    """

    def __init__(
        self,
        num_intruders=1,
        num_patrollers=3,
        num_obstacles=5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        mode="train",
    ):
        EzPickle.__init__(
            self,
            num_intruders=num_intruders,
            num_patrollers=num_patrollers,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            mode=mode,
        )

        scenario = Scenario()
        world = scenario.make_world(num_intruders, num_patrollers, num_obstacles, mode)

        BaseEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )

        self.metadata["name"] = "patrolling_v1"


env = make_env(PatrolEnv)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    """
    Scenario:
    - Goal at center
    - Obstacles randomized every reset
    - Energy stations fixed at corners
    - Step4: energy is part of observation (+1 dim)
    - Step5: add simple "station awareness" to observation (distance + dir to closest station) (+3 dim)
    """

    def __init__(self):
        self.intruder_won = False
        self.intruder_caught = False
        self.mode = "train"
        self.sensor_noise_std = 0.05

    # ---------- WORLD CREATION ----------

    def make_world(self, num_intruders=1, num_patrollers=3, num_obstacles=5, mode="train", max_obstacles=12):
        world = PatrolWorld()
        world.dim_c = 0
        self.mode = mode

        world.max_obstacles = int(max_obstacles)
        world.num_obstacles_active = int(num_obstacles)

        num_agents = num_intruders + num_patrollers
        world.agents = [Agent() for _ in range(num_agents)]

        self._configure_patrollers(world, num_patrollers)
        self._configure_intruders(world, num_intruders, num_patrollers)

        # ALWAYS build a fixed-size landmark set
        world.landmarks = self._create_landmarks_with_energy(max_obstacles=world.max_obstacles)

        obs_dim = self._calculate_observation_space(world, num_agents)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        action_space = Discrete(5)
        for a in world.agents:
            a.action_space = action_space

        return world

    def _create_landmarks_with_energy(self, max_obstacles: int):
        landmarks = []

        # Fixed maximum obstacle slots
        for i in range(max_obstacles):
            lm = Landmark()
            lm.name = f"landmark_{i}"
            lm.collide = True
            lm.movable = False
            lm.size = 0.1
            lm.boundary = False
            lm.color = np.array([0.65, 0.16, 0.16])
            lm.state.p_pos = np.zeros(2, dtype=np.float32)
            landmarks.append(lm)

        # Goal
        goal = Landmark()
        goal.name = "landmark_goal"
        goal.collide = True
        goal.movable = False
        goal.size = 0.1
        goal.boundary = False
        goal.color = np.array([0.25, 0.98, 0.25])
        goal.state.p_pos = np.array([0.0, 0.0], dtype=np.float32)
        landmarks.append(goal)

        # Energy stations
        energy_station_positions = [
            np.array([-1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([1.0, -1.0], dtype=np.float32),
        ]
        for i, pos in enumerate(energy_station_positions):
            st = Landmark()
            st.name = f"energy_station_{i}"
            st.collide = False
            st.movable = False
            st.size = 0.15
            st.boundary = False
            st.color = np.array([0.68, 0.85, 0.9])
            st.state.p_pos = pos
            landmarks.append(st)

        return landmarks

    # ---------- RESET / PLACEMENT ----------

    def _sample_arena_bounds(self, np_random):
        arena_half_extent = float(np_random.uniform(0.6, 1.0))
        return -arena_half_extent, arena_half_extent, arena_half_extent

    def _sample_non_overlapping_pos(
        self, np_random, world, occupied, candidate_size, low=-0.9, high=0.9, max_tries=5000
    ):
        for _ in range(max_tries):
            p = np_random.uniform(low, high, world.dim_p)
            ok = True
            for ent in occupied:
                q = getattr(ent.state, "p_pos", None)
                if q is None:
                    continue
                min_sep = float((getattr(ent, "size", 0.0) or 0.0) + candidate_size + 1e-3)
                if np.linalg.norm(p - q) < min_sep:
                    ok = False
                    break
            if ok:
                return p

        p = np_random.uniform(low, high, world.dim_p)
        return p + np_random.uniform(-1e-3, 1e-3, size=world.dim_p)

    def _too_close(self, p, q, min_dist) -> bool:
        return np.linalg.norm(p - q) < float(min_dist)

    def reset_world(self, world, np_random, env_map=None):
        """
        Place goal center, keep energy stations fixed,
        randomize obstacle & agent placement,
        and RESET ALL EPISODE STATE.
        """
        occupied = []

        # --- Arena bounds ---
        low, high, boundary_limit = self._sample_arena_bounds(np_random)
        world.boundary_limit = boundary_limit

        # --- Goal fixed at center ---
        goal = self.get_goal_landmark(world)
        goal_pos = np.array([0.0, 0.0], dtype=np.float32)
        if goal is not None:
            goal.state.p_pos = goal_pos
            occupied.append(goal)

        # --- Keep energy stations fixed ---
        for lm in world.landmarks:
            if "energy_station" in lm.name:
                occupied.append(lm)

        goal_clear_radius = 0.35

        # --- Place obstacles ---
        for lm in world.landmarks:
            if "goal" in lm.name or "energy_station" in lm.name:
                continue

            placed = False
            for _ in range(5000):
                p = self._sample_non_overlapping_pos(
                    np_random,
                    world,
                    occupied,
                    candidate_size=lm.size,
                    low=low,
                    high=high,
                )
                if not self._too_close(p, goal_pos, goal_clear_radius):
                    lm.state.p_pos = p
                    placed = True
                    break

            if not placed:
                lm.state.p_pos = p

            occupied.append(lm)

        # --- Place agents ---
        for agent in world.agents:
            # Reset motion
            agent.state.p_pos = self._sample_non_overlapping_pos(
                np_random,
                world,
                occupied,
                candidate_size=agent.size,
                low=low,
                high=high,
            )
            occupied.append(agent)

            agent.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)
            agent.state.c = np.zeros(world.dim_c, dtype=np.float32)

            # --- RESET EPISODE STATE ---
            if agent.patroller:
                agent.max_speed = 1.5
                agent.energy = agent.max_energy  # ✅ CRITICAL
                agent.recharging = False
                agent.last_distance_to_intruder = None
                agent.last_position = agent.state.p_pos.copy()
            else:
                agent.max_speed = 1.0
                agent.last_distance_to_center = None
                agent.previous_position = None

    # ---------- REWARDS ----------

    def reward(self, agent, world):
        return self.patroller_reward(agent, world) if agent.patroller else self.intruder_reward(agent, world)

    def intruder_reward(self, agent, world):
        reward = 0.0
        goal = self.get_goal_landmark(world)
        goal_pos = goal.state.p_pos if goal is not None else np.array([0.0, 0.0], dtype=np.float32)

        d_goal = np.linalg.norm(agent.state.p_pos - goal_pos)

        # Proximity to goal
        reward += max(0.0, (1.0 - d_goal)) * 10.0

        # Caught
        for p in world.agents:
            if p is not agent and p.patroller and self.is_collision(agent, p):
                reward -= 10.0
                self.intruder_caught = True
                if self.mode == "train":
                    logger.info(f"Intruder {agent.name} loses by being caught by {p.name}.")
                return reward

        if agent.last_distance_to_center is not None:
            delta = agent.last_distance_to_center - d_goal
            reward += 10.0 * max(0.0, delta)
            reward -= 5.0 * max(0.0, -delta)
        agent.last_distance_to_center = d_goal

        # Collisions with landmarks
        for lm in world.landmarks:
            if self.is_collision(agent, lm):
                if "goal" in lm.name:
                    reward += 100.0
                    self.intruder_won = True
                    if self.mode == "train":
                        logger.info(f"Intruder {agent.name} wins by reaching the goal.")
                    return reward
                elif "energy_station" in lm.name:
                    pass  # intruders ignore stations
                else:
                    reward -= 10.0
                break

        boundary_limit = getattr(world, "boundary_limit", 1.0)
        if np.any(agent.state.p_pos <= -boundary_limit) or np.any(agent.state.p_pos >= boundary_limit):
            reward -= 50.0

        if np.linalg.norm(agent.state.p_vel) < 0.05:
            reward -= 2.0

        reward -= 0.1
        return reward

    def patroller_reward(self, agent, world):
        reward = 0.0
        vision_range = agent.patrol_radius * 1.5

        d_intr, intr = self._find_closest_intruder(agent, world, vision_range)
        if intr is not None:
            reward += max(0.0, (vision_range - d_intr) / vision_range) * 20.0

            direction = (intr.state.p_pos - agent.state.p_pos) / (d_intr + 1e-6)
            vel_align = float(np.dot(agent.state.p_vel, direction))
            reward += vel_align * 5.0 if vel_align > 0 else -2.0

            patrollers = [a for a in world.agents if a.patroller and a is not agent]
            if patrollers:
                close = sum(1 for p in patrollers if np.linalg.norm(p.state.p_pos - intr.state.p_pos) < vision_range)
                reward += close * 2.0

                avg_d = self._calculate_average_distance(agent, patrollers)
                reward += (avg_d - 0.5) * 2.0

            catch = 0.0
            for other in world.agents:
                if not other.patroller and other is not agent and self.is_collision(agent, other):
                    catch += 50.0
            reward += catch
        else:
            v = np.linalg.norm(agent.state.p_vel)
            reward += v * 1.0
            if v < 0.1:
                reward -= 1.0

            patrollers = [a for a in world.agents if a.patroller and a is not agent]
            if patrollers:
                avg_d = self._calculate_average_distance(agent, patrollers)
                reward += (avg_d - 0.5) * 2.0

        return reward

    # ---------- OBSERVATION  ----------

    def observation(self, agent, world):
        agent_pos = agent.state.p_pos
        agent_vel = agent.state.p_vel

        landmarks_info = self._get_landmarks_info(agent, world)
        other_agents_info = self._get_other_agents_info(agent, world)
        goal_info = self._get_goal_info(agent, world)

        # Step4: Energy feature (normalized)
        if getattr(agent, "patroller", False) and hasattr(agent, "energy") and hasattr(agent, "max_energy"):
            energy_norm = float(agent.energy) / float(agent.max_energy + 1e-6)
        else:
            energy_norm = 0.0

        # Step5: closest energy station (distance + direction) for patrollers only
        if getattr(agent, "patroller", False):
            st_dist, st_dir = self._closest_energy_station(agent, world)
            station_feat = np.array([st_dist, st_dir[0], st_dir[1]], dtype=np.float32)
        else:
            station_feat = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        obs = np.concatenate([
            agent_pos.astype(np.float32),
            agent_vel.astype(np.float32),
            landmarks_info,
            goal_info,
            other_agents_info,
            np.array([energy_norm], dtype=np.float32),  # +1
            station_feat,                               # +3
        ])
        return obs

    def _closest_energy_station(self, agent, world):
        stations = [lm for lm in world.landmarks if "energy_station" in lm.name]
        if not stations:
            return 0.0, np.zeros(2, dtype=np.float32)

        best_d = float("inf")
        best_dir = np.zeros(2, dtype=np.float32)
        for st in stations:
            d = np.linalg.norm(st.state.p_pos - agent.state.p_pos)
            if d < best_d:
                best_d = d
                best_dir = (st.state.p_pos - agent.state.p_pos) / (d + 1e-6)
        return float(best_d), best_dir.astype(np.float32)

    # ---------- HELPERS ----------

    def get_goal_landmark(self, world):
        for lm in world.landmarks:
            if "goal" in lm.name:
                return lm
        return None

    def is_collision(self, e1, e2):
        delta = e1.state.p_pos - e2.state.p_pos
        dist = np.linalg.norm(delta)
        return dist < (e1.size + e2.size)

    def _configure_patrollers(self, world, num_patrollers):
        for i in range(num_patrollers):
            a = world.agents[i]
            a.name = f"patroller_{i}"
            a.collide = True
            a.silent = True
            a.size = 0.05
            a.accel = 2.0
            a.max_speed = 1.5
            a.patroller = True
            a.alive = True
            a.patrol_radius = 0.25
            a.sensor_range = a.patrol_radius * 1.3
            a.color = np.array([0.25, 0.25, 0.98])

            # Energy init (BaseEnv consumes these)
            a.max_energy = 100.0
            a.energy = a.max_energy
            a.recharging = False

    def _configure_intruders(self, world, num_intruders, num_patrollers):
        for i in range(num_intruders):
            a = world.agents[num_patrollers + i]
            a.name = f"intruder_{i}"
            a.collide = True
            a.silent = True
            a.size = 0.05
            a.accel = 2.0
            a.max_speed = 1.0
            a.patroller = False
            a.alive = True
            a.patrol_radius = 0.25
            a.last_distance_to_center = None
            a.previous_position = None
            a.color = np.array([0.98, 0.25, 0.25])

    def _calculate_observation_space(self, world, num_agents):
        # base
        obs_dim = 2  # pos
        obs_dim += 2  # vel
        obs_dim += 3 * len(world.landmarks)
        obs_dim += 3  # goal info
        obs_dim += 3 * (num_agents - 1)

        obs_dim += 1  # energy_norm


        obs_dim += 3  # closest station dist + dir2

        return obs_dim

    def _find_closest_intruder(self, agent, world, vision_range):
        best_d = float("inf")
        best = None
        for other in world.agents:
            if other is agent or other.patroller:
                continue
            d = np.linalg.norm(agent.state.p_pos - other.state.p_pos)
            if d <= agent.sensor_range and d < best_d:
                best_d = d
                best = other
            if best is not None and self.is_collision(agent, best):
                return d, best
        return best_d, best

    def _calculate_average_distance(self, agent, patrollers):
        ds = [np.linalg.norm(agent.state.p_pos - p.state.p_pos) for p in patrollers]
        return float(np.mean(ds)) if ds else 0.0

    def _sense_radius(self, agent):
        """
        Unified sensing radius for range-limited perception.
        """
        return agent.patrol_radius * 1.3

    def _blocked_by_obstacle(self, p0, p1, world):
        """
        Returns True if the line segment p0 -> p1 intersects any obstacle.
        Obstacles are landmarks with collide=True and not goal or energy stations.
        """
        for lm in world.landmarks:
            if not lm.collide:
                continue
            if "goal" in lm.name or "energy_station" in lm.name:
                continue

            c = lm.state.p_pos
            r = lm.size

            # Line segment p0->p1 vs circle (c, r)
            d = p1 - p0
            f = p0 - c

            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c_val = np.dot(f, f) - r * r

            discriminant = b * b - 4 * a * c_val
            if discriminant < 0:
                continue

            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2 * a)
            t2 = (-b + discriminant) / (2 * a)

            if (0.0 < t1 < 1.0) or (0.0 < t2 < 1.0):
                return True

        return False

    def _apply_noise(self, x: np.ndarray):
        """
        Apply Gaussian noise to sensor readings.
        """
        if self.sensor_noise_std <= 0.0:
            return x

        noise = np.random.normal(
            loc=0.0,
            scale=self.sensor_noise_std,
            size=x.shape
        )
        return x + noise

    def _get_landmarks_info(self, agent, world):
        out = []
        sensing_range = agent.patrol_radius * 1.3

        for lm in world.landmarks:
            d = np.linalg.norm(agent.state.p_pos - lm.state.p_pos)

            if d <= sensing_range and "goal" not in lm.name:
                direction = (lm.state.p_pos - agent.state.p_pos) / (d + 1e-6)

                vec = np.array([d, direction[0], direction[1]], dtype=np.float32)
                vec = self._apply_noise(vec)
                out.extend(vec.tolist())
            else:
                out.extend([0.0, 0.0, 0.0])

        return np.array(out, dtype=np.float32)

    def _get_other_agents_info(self, agent, world):
        out = []
        sensing_range = agent.patrol_radius * 1.3

        for other in world.agents:
            if other is agent:
                continue

            d = np.linalg.norm(agent.state.p_pos - other.state.p_pos)

            if d <= sensing_range:
                direction = (other.state.p_pos - agent.state.p_pos) / (d + 1e-6)
                vec = np.array([d, direction[0], direction[1]], dtype=np.float32)
                vec = self._apply_noise(vec)
                out.extend(vec.tolist())
            else:
                out.extend([0.0, 0.0, 0.0])

        return np.array(out, dtype=np.float32)


    def _get_goal_info(self, agent, world):
        goal = self.get_goal_landmark(world)
        if goal is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        d = np.linalg.norm(goal.state.p_pos - agent.state.p_pos)
        direction = (goal.state.p_pos - agent.state.p_pos) / (d + 1e-6)

        in_range = d <= agent.patrol_radius * 1.3
        blocked = self._blocked_by_obstacle(agent.state.p_pos, goal.state.p_pos, world)

        if in_range and not blocked:
            return np.concatenate([
                np.array([d], dtype=np.float32),
                direction.astype(np.float32)
            ])
        else:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
