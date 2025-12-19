import numpy as np
from gymnasium.utils import EzPickle
from gymnasium.spaces import Box, Discrete

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from baseEnv.baseEnv import BaseEnv, make_env
from baseEnv.patrolWorld import PatrolWorld
from pettingzoo.utils.conversions import parallel_wrapper_fn
import logging

logging.basicConfig(
    filename="patrol_env.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w"
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
        mode="train"
    ):
        """
        Initialize the Patrol Environment.

        Args:
            num_intruders (int): Number of intruder agents.
            num_patrollers (int): Number of patroller agents.
            num_obstacles (int): Number of static obstacles (landmarks).
            max_cycles (int): Maximum number of simulation cycles per episode.
            continuous_actions (bool): Whether agents have continuous action spaces.
            render_mode (str): Rendering mode for visualization.
        """
        EzPickle.__init__(
            self,
            num_intruders=num_intruders,
            num_patrollers=num_patrollers,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )

        scenario = Scenario()
        world = scenario.make_world(num_intruders, num_patrollers, num_obstacles, mode, continuous_actions=continuous_actions,)

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
    Defines the scenario for the Patrol Environment, including agent and landmark setup,
    reward functions, and observations.
    """

    def __init__(self):
        """
        Initialize the Scenario with flags for intruder outcomes.
        """
        self.intruder_won = False  # Flag when an intruder reaches the goal
        self.intruder_caught = False  # Flag when an intruder is caught by a patroller

    def make_world(self, num_intruders=1, num_patrollers=3, num_obstacles=5, mode="train", continuous_actions=False):
        """
        Create the world with agents and landmarks.

        Args:
            num_intruders (int): Number of intruder agents.
            num_patrollers (int): Number of patroller agents.
            num_obstacles (int): Number of static obstacles (landmarks).

        Returns:
            PatrolWorld: Configured world instance.
        """
        world = PatrolWorld()
        world.dim_c = 0

        self.mode = mode
        self.scale_factor = 100.0
        self.continuous_actions = continuous_actions

        num_agents = num_intruders + num_patrollers

        world.agents = [Agent() for _ in range(num_agents)]

        self._configure_patrollers(world, num_patrollers)

        self._configure_intruders(world, num_intruders, num_patrollers)

        world.landmarks = self.create_landmarks(num_obstacles=num_obstacles)

        obs_dim = self._calculate_observation_space(world, num_agents)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        return world

    def _min_sep(self, a, b, extra=1e-3) -> float:
        sa = float(getattr(a, "size", 0.0) or 0.0)
        sb = float(getattr(b, "size", 0.0) or 0.0)
        return sa + sb + extra

    def _sample_non_overlapping_pos(self, np_random, world, occupied, candidate_size, low=-0.9, high=0.9, max_tries=5000):
        """
        Samples a position that is at least x away from every entity.
        """
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

    def _sample_arena_bounds(self, np_random):
        """
        Randomize arena per episode.
        """
        arena_half_extent = float(np_random.uniform(0.6, 1.0))
        return -arena_half_extent, arena_half_extent, arena_half_extent

    def reset_world(self, world, np_random, env_map=None):
        """
        Reset the world's agents and landmarks to initial states.

        Args:
            world (PatrolWorld): The world instance to reset.
            np_random (numpy.random.RandomState): Random number generator.
            env_map (_generate_map): Generated obstacle map.
        """
        occupied = []

        low, high, boundary_limit = self._sample_arena_bounds(np_random)
        world.boundary_limit = boundary_limit

        goal = self.get_goal_landmark(world)
        goal_pos = np.array([0.0, 0.0], dtype=np.float32)
        if goal is not None:
            goal.state.p_pos = goal_pos
            occupied.append(goal)

        goal_clear_radius = 0.35

        for lm in world.landmarks:
            if "goal" in lm.name:
                continue

            placed = False
            for _ in range(5000):
                p = self._sample_non_overlapping_pos(
                    np_random, world, occupied,
                    candidate_size=lm.size,
                    low=low, high=high
                )
                if not self._too_close(p, goal_pos, goal_clear_radius):
                    lm.state.p_pos = p
                    placed = True
                    break

            if not placed:
                lm.state.p_pos = p

            occupied.append(lm)

        for agent in world.agents:
            agent.max_speed = 1.5 if agent.patroller else 1.0

            agent.state.p_pos = self._sample_non_overlapping_pos(
                np_random, world, occupied, candidate_size=agent.size, low=low, high=high
            )
            occupied.append(agent)

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

            if agent.patroller:
                agent.recharging = False
                agent.last_distance_to_intruder = None
                agent.last_position = agent.state.p_pos.copy()
            else:
                agent.last_distance_to_center = None
                agent.previous_position = None

    def reward(self, agent, world):
        """
        Assign rewards to agents based on their type.

        Args:
            agent (Agent): The agent to assign reward.
            world (PatrolWorld): The world instance.

        Returns:
            float: Calculated reward for the agent.
        """

        if agent.patroller:
            return self.patroller_reward(agent, world)
        else:
            return self.intruder_reward(agent, world)

    def intruder_reward(self, agent, world):
        """
        Calculate the reward for an intruder agent.

        Args:
            agent (Agent): The intruder agent.
            world (PatrolWorld): The world instance.

        Returns:
            float: Calculated reward.
        """
        reward = 0.0
        goal_position = self.get_goal_landmark(world).state.p_pos if self.get_goal_landmark(world) is not None else np.array([0.0, 0.0], dtype=np.float32)

        current_distance_to_goal = np.linalg.norm(agent.state.p_pos - goal_position)

        # Linear reward for proximity to the goal
        proximity_reward = max(0.0, (1.0 - current_distance_to_goal)) * 10.0
        reward += proximity_reward

        # Check for collisions with patrollers
        for other_agent in world.agents:
            if other_agent != agent and other_agent.patroller:
                if self.is_collision(agent, other_agent):
                    reward -= 10.0  # Penalty for being caught
                    self.intruder_caught = True

                    if self.mode == "train":
                        logger.info(f"Intruder {agent.name} loses by getting caught by {other_agent.name}!")
                    return reward

        if agent.last_distance_to_center is not None:
            distance_change = (
                agent.last_distance_to_center - current_distance_to_goal
            )
            reward += 10.0 * max(0.0, distance_change)
            reward -= 5.0 * max(0.0, -distance_change)

        agent.last_distance_to_center = current_distance_to_goal

        for landmark in world.landmarks:
            if self.is_collision(agent, landmark):
                if "goal" in landmark.name:
                    reward += 100.0  # Large bonus for reaching the goal
                    self.intruder_won = True
                    if self.mode == "train":
                        logger.info(f"Intruder {agent.name} wins by reaching the center!")
                    return reward
                else:
                    reward -= 10.0  # Penalty for hitting obstacles
                break  

        # Penalize collisions with boundaries
        boundary_limit = getattr(world, "boundary_limit", 1.0)
        if (
            np.any(agent.state.p_pos <= -boundary_limit)
            or np.any(agent.state.p_pos >= boundary_limit)
        ):
            reward -= 50.0 

        # Penalize minimal movement to discourage hovering
        movement_threshold = 0.05
        velocity = np.linalg.norm(agent.state.p_vel)
        if velocity < movement_threshold:
            reward -= 2.0 

        reward -= 0.1

        return reward

    def patroller_reward(self, agent, world):
        """
        Calculate the reward for a patroller agent.

        Args:
            agent (Agent): The patroller agent.
            world (PatrolWorld): The world instance.

        Returns:
            float: Calculated reward.
        """
        reward = 0.0
        vision_range = agent.patrol_radius * 1.5

        # Find the closest intruder
        min_distance_to_intruder, closest_intruder = self._find_closest_intruder(agent, world, vision_range)

        if closest_intruder:
            # Reward for proximity to the closest intruder
            proximity_reward = max(0.0, (vision_range - min_distance_to_intruder) / vision_range) * 20.0
            reward += proximity_reward 

            # Encourage movement towards the intruder
            direction_to_intruder = (
                closest_intruder.state.p_pos - agent.state.p_pos
            ) / (min_distance_to_intruder + 1e-6)
            velocity_aligned = np.dot(agent.state.p_vel, direction_to_intruder)
            if velocity_aligned > 0:
                reward += velocity_aligned * 5.0 
            else:
                reward -= 2.0 

            # Encourage coordinated pursuit
            patrollers = [a for a in world.agents if a.patroller and a != agent]
            if patrollers:
                close_patrollers = sum([
                    1 for p in patrollers
                    if np.linalg.norm(p.state.p_pos - closest_intruder.state.p_pos) < vision_range
                ])
                coordination_reward = close_patrollers * 2.0
                reward += coordination_reward

                # Encourage patrollers to spread out
                avg_distance = self._calculate_average_distance(agent, patrollers)
                optimal_distance = 0.5 
                distance_diff = avg_distance - optimal_distance
                reward += distance_diff * 2.0 

            # Reward for catching an intruder
            catch_reward = 0.0
            for other_agent in world.agents:
                if other_agent != agent and not other_agent.patroller:
                    if self.is_collision(agent, other_agent):
                        catch_reward += 50.0 
            reward += catch_reward

        else:
            # No intruders detected, encourage patrollers to explore
            velocity = np.linalg.norm(agent.state.p_vel)
            reward += velocity * 1.0 

            # Penalize for staying still
            if velocity < 0.1:
                reward -= 1.0

            patrollers = [a for a in world.agents if a.patroller and a != agent]
            if patrollers:
                avg_distance = self._calculate_average_distance(agent, patrollers)
                optimal_distance = 0.5 
                distance_diff = avg_distance - optimal_distance
                reward += distance_diff * 2.0  

        return reward


    def is_collision(self, entity1, entity2):
        """
        Determine if two entities have collided based on their positions and sizes.

        Args:
            entity1 (Agent or Landmark): First entity.
            entity2 (Agent or Landmark): Second entity.

        Returns:
            bool: True if collision occurs, False otherwise.
        """
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = entity1.size + entity2.size
        return dist < dist_min

    def observation(self, agent, world):
        """
        Generate the observation for an agent, including positions, velocities,
        landmarks, goal information, and other agents within visibility.

        Args:
            agent (Agent): The agent for which to generate the observation.
            world (PatrolWorld): The world instance.

        Returns:
            np.ndarray: The concatenated observation vector.
        """
        # Agent's own position and velocity
        agent_pos = agent.state.p_pos  # Shape: (2,)
        agent_vel = agent.state.p_vel  # Shape: (2,)

        # Landmarks information
        landmarks_info = self._get_landmarks_info(agent, world)

        # Other agents' information
        other_agents_info = self._get_other_agents_info(agent, world)

        # Goal information
        goal_info = self._get_goal_info(agent, world)

        """if agent.patroller:
            energy_info = np.array([agent.energy / 100.0], dtype=np.float32)  # Normalize to [0, 1]

            closest_station, distance_to_station = self._find_closest_energy_station(agent, world)
            if closest_station:
                direction_to_station = (
                    closest_station.state.p_pos - agent.state.p_pos
                ) / (distance_to_station + 1e-6)  # Avoid division by zero
                energy_station_info = np.concatenate([
                    np.array([distance_to_station], dtype=np.float32),  # Distance
                    direction_to_station.astype(np.float32)             # Direction (2 elements)
                ])
            else:
                energy_station_info = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        else:
            energy_info = np.array([0.0], dtype=np.float32)
            energy_station_info = np.array([0.0, 0.0, 0.0], dtype=np.float32)"""

        observation = np.concatenate([
            agent_pos.astype(np.float32),          # 2 elements
            agent_vel.astype(np.float32),          # 2 elements
            landmarks_info,                        # 3 * num_landmarks elements
            goal_info,                             # 3 elements
            other_agents_info                     # 3 * (num_agents - 1) elements
            #energy_info,                           # 1 element
            #energy_station_info                    # 3 elements
        ])

        return observation

    def get_goal_landmark(self, world):
        """
        Retrieve the goal landmark from the world's landmarks.

        Args:
            world (PatrolWorld): The world instance.

        Returns:
            Landmark or None: The goal landmark if exists, else None.
        """
        for landmark in world.landmarks:
            if "goal" in landmark.name:
                return landmark
        return None

    def create_static_landmarks(self):
        """
        Create static landmarks including obstacles and the goal.

        Returns:
            list[Landmark]: List of configured landmark instances.
        """
        landmark_positions = [
            # Top Line
            np.array([-0.7, 0.5]),
            np.array([-0.2, 0.5]),
            np.array([0.3, 0.5]),
            # Bottom Line
            np.array([-0.7, -0.5]),
            np.array([-0.1, -0.5]),
            np.array([0.5, -0.5]),
        ]
        energy_station_positions = [
            np.array([-1.0, 1.0]),  # Top-left
            np.array([1.0, 1.0]),   # Top-right
            np.array([-1.0, -1.0]), # Bottom-left
            np.array([1.0, -1.0])   # Bottom-right
        ]

        landmarks = [Landmark() for _ in range(len(landmark_positions) + len(energy_station_positions) + 1)]  # +1 for the goal + 4

        for i, landmark in enumerate(landmarks):
            if i < len(landmark_positions):
                landmark.name = f"landmark_{i}"
                landmark.collide = True
                landmark.movable = False
                landmark.size = 0.1
                landmark.boundary = False
                landmark.state.p_pos = landmark_positions[i]
                landmark.color = np.array([0.65, 0.16, 0.16])
            elif i == len(landmark_positions):
                # Configure the goal landmark
                landmark.name = "landmark_goal"
                landmark.collide = True
                landmark.movable = False
                landmark.size = 0.1
                landmark.boundary = False
                landmark.state.p_pos = np.array([0.0, 0.0])
                landmark.color = np.array([0.25, 0.98, 0.25])
            else:
                # Configure energy stations
                energy_index = i - len(landmark_positions) - 1
                landmark.name = f"energy_station_{energy_index}"
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.15  # Larger size for visibility
                landmark.boundary = False
                landmark.state.p_pos = energy_station_positions[energy_index]
                landmark.color = np.array([0.68, 0.85, 0.9])

        return landmarks

    def create_landmarks(self, num_obstacles: int):
        landmarks = []

        for i in range(num_obstacles):
            lm = Landmark()
            lm.name = f"landmark_{i}"
            lm.collide = True
            lm.movable = False
            lm.size = 0.1
            lm.boundary = False
            lm.color = np.array([0.65, 0.16, 0.16])
            lm.state.p_pos = np.zeros(2, dtype=np.float32)
            landmarks.append(lm)

        goal = Landmark()
        goal.name = "landmark_goal"
        goal.collide = True
        goal.movable = False
        goal.size = 0.1
        goal.boundary = False
        goal.color = np.array([0.25, 0.98, 0.25])
        goal.state.p_pos = np.zeros(2, dtype=np.float32)
        landmarks.append(goal)

        energy_station_positions = [
            np.array([-1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([1.0, -1.0], dtype=np.float32),
        ]

        for idx, pos in enumerate(energy_station_positions):
            st = Landmark()
            st.name = f"energy_station_{idx}"
            st.collide = False
            st.movable = False
            st.size = 0.15
            st.boundary = False
            st.color = np.array([0.68, 0.85, 0.9])
            st.state.p_pos = pos
            landmarks.append(st)

        return landmarks

    def _configure_patrollers(self, world, num_patrollers):
        """
        Configure patroller agents within the world.

        Args:
            world (PatrolWorld): The world instance.
            num_patrollers (int): Number of patroller agents to configure.
        """
        for i in range(num_patrollers):
            agent = world.agents[i]
            agent.name = f"patroller_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 2.0
            agent.max_speed = 1.5 
            agent.patroller = True
            agent.alive = True
            agent.patrol_radius = 0.25
            agent.color = np.array([0.25, 0.25, 0.98])

            agent.max_energy = 100.0
            agent.energy = agent.max_energy
            agent.recharging = False

    """def reduce_energy(self, agent):
        if agent.recharging:
            return  # Skip energy reduction if the agent is recharging

        velocity = np.linalg.norm(agent.state.p_vel)
        energy_cost = velocity * 0.3  # Reduce energy based on speed
        agent.energy = max(0.0, agent.energy - energy_cost)  # Ensure energy doesn't go below 0

    def restrict_movement_if_no_energy(self, agent):
        if agent.energy <= 0:
            agent.state.p_vel = np.zeros_like(agent.state.p_vel)  # Stop movement
            agent.max_speed = 0.0  # Temporarily disable movement
        else:
            agent.max_speed = 1.5  # Restore movement if energy is available

    def recharge_energy(self, agent, world):
        for landmark in world.landmarks:
            if "energy_station" in landmark.name:
                distance_to_station = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                if distance_to_station < landmark.size + agent.size:  # Inside station
                    agent.recharging = True
                    agent.energy = min(100.0, agent.energy + 2.0)  # Recharge rate
                    return

        agent.recharging = False  # Not inside any energy station"""



    def _configure_intruders(self, world, num_intruders, num_patrollers):
        """
        Configure intruder agents within the world.

        Args:
            world (PatrolWorld): The world instance.
            num_intruders (int): Number of intruder agents to configure.
            num_patrollers (int): Number of patroller agents to offset intruder indexing.
        """
        for i in range(num_intruders):
            agent = world.agents[num_patrollers + i]
            agent.name = f"intruder_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 2.0
            agent.max_speed = 1.0
            agent.patroller = False
            agent.alive = True
            agent.patrol_radius = 0.25
            agent.last_distance_to_center = None
            agent.previous_position = None
            agent.color = np.array([0.98, 0.25, 0.25])  # Red color

    def _calculate_observation_space(self, world, num_agents):
        """
        Calculate the observation space dimensions based on the world configuration.

        Args:
            world (PatrolWorld): The world instance.
            num_agents (int): Total number of agents in the world.

        Returns:
            int: Total dimension of the observation space.
        """
        obs_dim = 2  # Agent's own position
        obs_dim += 2  # Agent's own velocity
        obs_dim += 3 * len(world.landmarks)  # Landmarks info (distance + direction_x + direction_y)
        obs_dim += 3  # Goal info (distance + direction_x + direction_y)
        obs_dim += 3 * (num_agents - 1)  # Other agents' info
        #obs_dim += 1   # energy
        #obs_dim += 3  # Energy station info (distance + direction_x + direction_y)
        return obs_dim

    def _find_closest_intruder(self, agent, world, vision_range):
        """
        Identify the closest intruder within the patroller's vision range.

        Args:
            agent (Agent): The patroller agent.
            world (PatrolWorld): The world instance.
            vision_range (float): The vision range of the patroller.

        Returns:
            tuple: (min_distance_to_intruder, closest_intruder_agent)
        """
        min_distance = float('inf')
        closest_intruder = None

        for other_agent in world.agents:
            if other_agent != agent and not other_agent.patroller:
                distance = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
                if distance <= vision_range:
                    if distance < min_distance:
                        min_distance = distance
                        closest_intruder = other_agent

                    if self.is_collision(agent, other_agent):
                        return distance, other_agent

        return min_distance, closest_intruder

    def _calculate_average_distance(self, agent, patrollers):
        """
        Calculate the average distance between the patroller and other patrollers.

        Args:
            agent (Agent): The patroller agent.
            patrollers (list[Agent]): List of other patroller agents.

        Returns:
            float: Average distance to other patrollers.
        """
        distances = [np.linalg.norm(agent.state.p_pos - p.state.p_pos) for p in patrollers]
        return np.mean(distances) if distances else 0.0

    def _get_landmarks_info(self, agent, world):
        """
        Retrieve landmarks information relative to the agent.

        Args:
            agent (Agent): The observing agent.
            world (PatrolWorld): The world instance.

        Returns:
            np.ndarray: Flattened list of landmarks' distance and direction.
        """
        landmarks_info = []
        for landmark in world.landmarks:
            distance = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            direction = (landmark.state.p_pos - agent.state.p_pos) / (distance + 1e-6) 

            # Include information only if within vision range and not the goal
            if agent.patrol_radius * 1.3 > distance and "goal" not in landmark.name:
                landmarks_info.append(distance)                   # 1 element
                landmarks_info.extend(direction.tolist())          # 2 elements
            else:
                landmarks_info.extend([0.0, 0.0, 0.0])            # 3 elements

        return np.array(landmarks_info, dtype=np.float32)

    def _get_other_agents_info(self, agent, world):
        """
        Retrieve information about other agents relative to the observing agent.

        Args:
            agent (Agent): The observing agent.
            world (PatrolWorld): The world instance.

        Returns:
            np.ndarray: Flattened list of other agents' distance and direction.
        """
        other_agents_info = []
        for other_agent in world.agents:
            if other_agent is agent:
                continue  # Skip self

            distance = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
            direction = (other_agent.state.p_pos - agent.state.p_pos) / (distance + 1e-6)  # Avoid division by zero

            # Include information only if within vision range
            if agent.patrol_radius * 1.3 > distance:
                other_agents_info.append(distance)               # 1 element
                other_agents_info.extend(direction.tolist())      # 2 elements
            else:
                other_agents_info.extend([0.0, 0.0, 0.0])        # 3 elements

        return np.array(other_agents_info, dtype=np.float32)

    def _get_goal_info(self, agent, world):
        """
        Retrieve goal information relative to the agent.

        Args:
            agent (Agent): The agent observing the goal.
            world (PatrolWorld): The world instance.

        Returns:
            np.ndarray: Distance and direction to the goal.
        """
        goal_landmark = self.get_goal_landmark(world)
        if goal_landmark:
            goal_position = goal_landmark.state.p_pos
            distance_to_goal = np.linalg.norm(goal_position - agent.state.p_pos)
            direction_to_goal = (goal_position - agent.state.p_pos) / (distance_to_goal + 1e-6)
            goal_info = np.concatenate([
                np.array([distance_to_goal], dtype=np.float32), 
                direction_to_goal.astype(np.float32)         
            ])
        else:
            # If no goal landmark is defined, default to zero
            goal_info = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return goal_info

    """def _find_closest_energy_station(self, agent, world):
        min_distance = float('inf')
        closest_station = None

        for landmark in world.landmarks:
            if "energy_station" in landmark.name:
                distance = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_station = landmark

        return closest_station, min_distance"""
