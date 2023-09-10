import numpy as np
from lib.discrete_agent import DiscreteAgent
from grid_traversal_env import GridTraversalDiscreteEnvironment
from restriction import DummyRestriction


class LDFSDiscreteAgent(DiscreteAgent):
    def __init__(self, env: GridTraversalDiscreteEnvironment, coefficient: float = 1.0):
        self.env = env
        self.explore_count = 0
        self.max_depth = int((env.agent_pos[0] + env.agent_pos[1] + env.target_pos[0] + env.target_pos[
            1]) * coefficient)  # Manhatan distance * coefficient
        self.operations = self.compute_operations_stack(self.env.environment, self.env.agent_pos, self.env.target_pos)
        super().__init__(env)
        env.add_restriction(DummyRestriction({'explored': self.explore_count}))

    def get_action(self, observation: tuple) -> int:
        if len(self.operations) > 0:
            return self.operations.pop(0)
        return 0

    def print(self) -> None:
        print("DFS Agent")

    def compute_operations_stack(self, environment, agent_pos, target_pos):
        if np.all(agent_pos == target_pos):
            return []
        visited = np.zeros_like(environment, dtype=bool)
        stack = [(agent_pos, [])]
        while len(stack) > 0:
            cagent_pos, path = stack.pop()
            self.explore_count += 1
            if visited[tuple(cagent_pos)]:
                continue
            if np.all(cagent_pos == target_pos):
                return path
            visited[tuple(cagent_pos)] = True
            for action in range(1, len(self.env.actions)):
                direction = self.env.action_to_direction(action)
                new_pos = cagent_pos + direction
                if self.env.is_valid_pos(new_pos) and not visited[tuple(new_pos)] and len(path) + 1 < self.max_depth:
                    stack.append((new_pos, [*path, action]))
        return []
