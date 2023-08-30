import numpy as np
from lib.discrete_agent import DiscreteAgent
from grid_traversal_env import GridTraversalDiscreteEnvironment

class DijkstraDiscreteAgent(DiscreteAgent):
    def __init__(self, env: GridTraversalDiscreteEnvironment):
        self.env = env
        self.operations = self.compute_operations_dj(self.env.environment, self.env.agent_pos, self.env.target_pos)
        super().__init__(env)
        # self.env.print()
        # print(f"{[self.env.action_to_direction(x) for x in self.operations]=}")

    def get_action(self, observation: tuple) -> int:
        if len(self.operations) > 0:
            return self.operations.pop(0)
        raise Exception("Am done")

    def print(self) -> None:
        print("BFS Agent")

    def compute_operations_dj(self, environment, agent_pos, target_pos):
        if np.all(agent_pos == target_pos):
            return []
        visit_weight = np.full_like(environment, 999999999, dtype=int)
        will_visit = np.zeros_like(environment, dtype=bool)
        queue = [(agent_pos, [])]
        visit_weight[tuple(agent_pos)] = 0
        while len(queue) > 0:
            cagent_pos, path = queue.pop(0)
            if np.all(cagent_pos == target_pos):
                return path
            if visit_weight[tuple(cagent_pos)] < len(path):
                continue
            visit_weight[tuple(cagent_pos)] = len(path)
            for action in range(len(self.env.actions)):
                direction = self.env.action_to_direction(action)
                new_pos = cagent_pos + direction
                if self.env.is_valid_pos(new_pos) and not will_visit[tuple(new_pos)] and visit_weight[tuple(cagent_pos)] + 1 < visit_weight[tuple(new_pos)]:
                    will_visit[tuple(new_pos)] = True
                    queue.append((new_pos, [*path, action]))
        return None
