from dataclasses import dataclass, field
from typing import Any

import numpy as np
from queue import PriorityQueue
from lib.discrete_agent import DiscreteAgent
from grid_traversal_env import GridTraversalDiscreteEnvironment

## https://docs.python.org/3/library/queue.html
@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

class AStarDiscreteAgent(DiscreteAgent):
    def __init__(self, env: GridTraversalDiscreteEnvironment):
        self.env = env
        self.operations = self.compute_operations_queue(self.env.environment, self.env.agent_pos, self.env.target_pos)
        super().__init__(env)

    def get_action(self, observation: tuple) -> int:
        if len(self.operations) > 0:
            return self.operations.pop(0)
        return 0

    def print(self) -> None:
        print("AStar Agent")

    def heuristic(self, pos, target_pos):
        norm = np.linalg.norm(pos - target_pos)
        return norm

    def compute_operations_queue(self, environment, agent_pos, target_pos):
        if self.is_same(agent_pos, target_pos):
            return []
        visited = np.zeros_like(environment, dtype=bool)
        queue = PriorityQueue()
        queue.put(PrioritizedItem(0, (agent_pos, [])))
        while not queue.empty():
            cagent_pos, path = queue.get().item
            if visited[tuple(cagent_pos)]:
                continue
            if self.is_same(cagent_pos, target_pos):
                return path
            visited[tuple(cagent_pos)] = True
            for action in range(1, len(self.env.actions)):
                direction = self.env.action_to_direction(action)
                new_pos = cagent_pos + direction
                if self.env.is_valid_pos(new_pos) and not visited[tuple(new_pos)]:
                    item = PrioritizedItem(self.heuristic(new_pos, target_pos), (new_pos, [*path, action]))
                    queue.put(item)
        return []

    def is_same(self, agent_pos, target_pos):
        return np.all(agent_pos == target_pos)
