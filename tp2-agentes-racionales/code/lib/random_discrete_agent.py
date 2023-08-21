from .discrete_agent import DiscreteAgent
from numpy.random import randint as rand
class RandomAgent(DiscreteAgent):
    def get_action(self, observation: list) -> int:
        return rand(0, len(self.actions))

    def print(self) -> None:
        print("Random agent")
