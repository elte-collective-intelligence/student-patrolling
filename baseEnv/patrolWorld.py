from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
import numpy as np

class PatrolWorld(World):
    def __init__(self):
        super().__init__()


    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        p_force = self.apply_action_force(p_force)
        p_force = self.apply_environment_force(p_force)

        # integrate physical state
        self.integrate_state(p_force)

        # Clip agent positions to stay within bounds [-1, 1]
        for agent in self.agents:
            agent.state.p_pos = np.clip(agent.state.p_pos, -1.0, 1.0)

        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
