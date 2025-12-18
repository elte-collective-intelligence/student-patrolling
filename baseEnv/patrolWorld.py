from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
import numpy as np

class PatrolWorld(World):
    def __init__(self):
        super().__init__()

    def _min_sep(self, a, b, extra=1e-6) -> float:
        sa = float(getattr(a, "size", 0.0) or 0.0)
        sb = float(getattr(b, "size", 0.0) or 0.0)
        return sa + sb + extra

    def resolve_degenerate_overlaps(self, extra=1e-6, max_passes=2):
        """
        Prevent zero-distance overlaps
        """
        entities = list(getattr(self, "entities", []))
        if not entities:
            return

        for _ in range(max_passes):
            moved_any = False

            for i in range(len(entities)):
                ei = entities[i]
                pi = getattr(ei.state, "p_pos", None)
                if pi is None:
                    continue

                for j in range(i + 1, len(entities)):
                    ej = entities[j]
                    pj = getattr(ej.state, "p_pos", None)
                    if pj is None:
                        continue

                    d = pj - pi
                    dist = float(np.linalg.norm(d))
                    min_sep = self._min_sep(ei, ej, extra=extra)

                    if dist < min_sep:
                        if dist < 1e-12:
                            direction = np.random.uniform(-1.0, 1.0, size=pi.shape)
                            direction /= (np.linalg.norm(direction) + 1e-12)
                        else:
                            direction = d / (dist + 1e-12)

                        push = 0.5 * (min_sep - dist) * direction
                        ei.state.p_pos = pi - push
                        ej.state.p_pos = pj + push
                        moved_any = True

            if not moved_any:
                break

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

        self.resolve_degenerate_overlaps(extra=1e-6, max_passes=3)

        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
