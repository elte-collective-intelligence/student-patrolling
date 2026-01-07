import os

import gymnasium
import numpy as np
import pygame
import pygame.freetype
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.mpe._mpe_utils.core import Agent
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class BaseEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 40,
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
        continuous_actions=False,
        local_ratio=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.viewer = None
        self.width = 1000
        self.height = 1000
        self.max_size = 1

        self.window = None
        self.screen = None
        self.clock = None
        self.game_font = None

        # Set up the drawing window
        self.renderOn = False
        self._seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                if agent.movable:
                    self.action_spaces[agent.name] = spaces.Box(
                        low=-1.0, high=1.0, shape=(self.world.dim_p,), dtype=np.float32
                    )
                else:
                    self.action_spaces[agent.name] = spaces.Box(
                        low=-1.0, high=1.0, shape=(0,), dtype=np.float32
                    )
            else:
                if agent.movable:
                    self.action_spaces[agent.name] = spaces.Discrete(5)
                else:
                    self.action_spaces[agent.name] = spaces.Discrete(1)
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0
        self.current_actions = [None] * self.num_agents

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        ).astype(np.float32)

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def _generate_map(self):
        n_obs = 8
        obs = []
        for _ in range(n_obs):
            x = self.np_random.uniform(-0.9, 0.9)
            y = self.np_random.uniform(-0.9, 0.9)
            r = self.np_random.uniform(0.05, 0.12)
            obs.append((x, y, r))
        return obs


    #  ENERGY + RECHARGE


    def _handle_recharge(self):
        """
        Recharge patrollers when inside an energy station landmark.
        Stations are landmarks whose name contains 'energy_station'.
        """
        for agent in self.world.agents:
            if not getattr(agent, "patroller", False):
                continue

            # Safety
            if not hasattr(agent, "energy") or not hasattr(agent, "max_energy"):
                continue

            agent.recharging = False

            # If no landmarks exist, do nothing
            if not hasattr(self.world, "landmarks"):
                continue

            for lm in self.world.landmarks:
                if "energy_station" not in getattr(lm, "name", ""):
                    continue

                dist = np.linalg.norm(agent.state.p_pos - lm.state.p_pos)
                if dist <= (agent.size + lm.size):
                    agent.recharging = True
                    agent.energy = min(agent.max_energy, agent.energy + 2.0)  # recharge rate
                    break

    def _update_energy(self):
        """
        Drain energy for patrollers based on velocity.
        Intruders ignore energy completely.
        """
        for agent in self.world.agents:
            if not getattr(agent, "patroller", False):
                continue  # intruders ignore energy

            if getattr(agent, "recharging", False):
                continue

            # Safety: if energy attrs are missing, skip instead of crashing
            if not hasattr(agent, "energy") or not hasattr(agent, "max_energy"):
                continue

            velocity = np.linalg.norm(agent.state.p_vel)
            energy_cost = 0.3 * velocity
            agent.energy = max(0.0, agent.energy - energy_cost)

            # If out of energy, stop movement
            if agent.energy <= 0.0:
                agent.state.p_vel[:] = 0.0
                agent.max_speed = 0.0
            else:
                # Restore normal speed if energy is available
                agent.max_speed = 1.5

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.map_obs = self._generate_map()
        self.scenario.reset_world(self.world, self.np_random, env_map=self.map_obs)


        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0
        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                if self.continuous_actions:
                    scenario_action.append(np.asarray(action, dtype=np.float32))
                else:
                    scenario_action.append(int(action))

            if (not agent.silent) and (self.world.dim_c > 0):
                raise NotImplementedError(
                    "Communication actions are enabled but not implementd."
                )

            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        # Recharge first, then drain
        self._handle_recharge()
        self._update_energy()

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p, dtype=np.float32)
        agent.action.c = np.zeros(self.world.dim_c, dtype=np.float32)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p, dtype=np.float32)

            if self.continuous_actions:
                u = np.clip(np.asarray(action[0], dtype=np.float32), -1.0, 1.0)
                u = 4.0 * u
                agent.action.u[:] = u
            else:
                # process discrete action
                if int(action[0]) == 0:
                    pass
                elif int(action[0]) == 1:
                    agent.action.u[0] = -1.0
                elif int(action[0]) == 2:
                    agent.action.u[0] = +1.0
                elif int(action[0]) == 3:
                    agent.action.u[1] = -1.0
                elif int(action[0]) == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = np.asarray(action[0], dtype=np.float32)
            else:
                agent.action.c = np.zeros(self.world.dim_c, dtype=np.float32)
                agent.action.c[int(action[0])] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            # All agents have acted, now we advance the world
            self._execute_world_step()
            self.steps += 1
            self.scenario.episode_steps += 1

            # Check termination conditions from your scenario
            if self.scenario.intruder_won:
                # End episode: intruder reached the goal
                for a in self.agents:
                    self.terminations[a] = True
            elif self.scenario.intruder_caught:
                # End episode: intruder was caught
                for a in self.agents:
                    self.terminations[a] = True

            if self.steps >= self.max_cycles:
                # End due to max cycles
                for a in self.agents:
                    self.truncations[a] = True

            # If the episode ended this step, populate info["episode"]
            if all(self.terminations[a] or self.truncations[a] for a in self.agents):
                metrics = self.scenario.episode_metrics()
                for a in self.agents:
                    self.infos[a].update(metrics)

        else:
            self._clear_rewards()

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        if self.renderOn:
            return

        pygame.init()

        if mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.screen = self.window

        elif mode == "rgb_array":
            self.screen = pygame.Surface((self.width, self.height))

        self.clock = pygame.time.Clock()
        self.renderOn = True

        if self.game_font is None:
            self.game_font = pygame.freetype.Font(
                os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
            )

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        if self.render_mode == "human":
            pygame.event.pump()

        self.draw()

        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))

        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def draw(self):
        # Clear screen
        self.screen.fill((255, 255, 255))

        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))
        cam_range = 1.0

        agents = [entity for entity in self.world.entities if isinstance(entity, Agent)]
        other_entities = [
            entity for entity in self.world.entities if not isinstance(entity, Agent)
        ]

        def calculate_screen_coords(x, y):
            y *= -1
            x = (x / cam_range) * self.width // 2 * 0.9
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            return int(x), int(y)

        for entity in other_entities:
            x, y = calculate_screen_coords(*entity.state.p_pos)
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.size * 350
            )
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )

        for agent in agents:
            x, y = calculate_screen_coords(*agent.state.p_pos)
            pygame.draw.circle(
                self.screen, agent.color * 200, (x, y), agent.size * 350
            )
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), agent.size * 350, 1
            )

            if isinstance(agent, Agent):
                pygame.draw.circle(
                    self.screen, (0, 191, 255), (x, y), agent.patrol_radius * 350, 2
                )

        text_line = 0
        for agent in agents:
            if not agent.silent:
                if np.all(agent.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in agent.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(agent.state.c)]

                message = agent.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def close(self):
        try:
            if self.window is not None:
                pygame.display.quit()
                self.window = None
            pygame.quit()
        except Exception:
            pass
        finally:
            self.screen = None
            self.clock = None
            self.renderOn = False
