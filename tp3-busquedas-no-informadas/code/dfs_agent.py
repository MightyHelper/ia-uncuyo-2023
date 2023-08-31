import numpy as np
from lib.discrete_agent import DiscreteAgent
from grid_traversal_env import GridTraversalDiscreteEnvironment

class DFSDiscreteAgent(DiscreteAgent):
    def __init__(self, env: GridTraversalDiscreteEnvironment):
        self.env = env
        self.operations = self.compute_operations_stack(self.env.environment, self.env.agent_pos, self.env.target_pos)
        super().__init__(env)
        # self.env.print()
        # print(f"{[self.env.action_to_direction(x) for x in self.operations]=}")

    def get_action(self, observation: tuple) -> int:
        if len(self.operations) > 0:
            return self.operations.pop(0)
        return 0

    def print(self) -> None:
        print("DFS Agent")

    def compute_operations(self, environment, agent_pos, target_pos, path=None, visited=None):
        if visited is None:
            visited = np.zeros_like(environment, dtype=bool)
        if path is None:
            path = []
        if visited[tuple(agent_pos)]:
            return
        if np.all(agent_pos == target_pos):
            return path
        visited[tuple(agent_pos)] = True
        for action in range(len(self.env.actions)):
            direction = self.env.action_to_direction(action)
            new_pos = agent_pos + direction
            if self.env.is_valid_pos(new_pos) and not visited[tuple(new_pos)]:
                path.append(action)
                result = self.compute_operations(environment, new_pos, target_pos, path, visited)
                if result is not None:
                    return result
                path.pop()
        return []

    def compute_operations_stack(self, environment, agent_pos, target_pos):
        if np.all(agent_pos == target_pos):
            return []
        visited = np.zeros_like(environment, dtype=bool)
        stack = [(agent_pos, [])]
        while len(stack) > 0:
            cagent_pos, path = stack.pop()
            # print("CP", cagent_pos, path, cagent_pos == target_pos, np.all(cagent_pos == target_pos))
            if visited[tuple(cagent_pos)]:
                continue
            if np.all(cagent_pos == target_pos):
                # print("Solved")
                return path
            visited[tuple(cagent_pos)] = True
            for action in range(1, len(self.env.actions)):
                direction = self.env.action_to_direction(action)
                # print("Testing dir", direction)
                new_pos = cagent_pos + direction
                if self.env.is_valid_pos(new_pos) and not visited[tuple(new_pos)]:
                    stack.append((new_pos, [*path, action]))
                    # print("ST", stack)
                else:
                    pass
                    # print("Invalid", new_pos)
            # print("ST", stack)
        return []
