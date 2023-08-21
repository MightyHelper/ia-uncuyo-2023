from abc import ABC, abstractmethod
from .discrete_env import DiscreteEnvironment

class DiscreteAgent(ABC):
    actions: list[str] = []

    def __init__(self, environment: DiscreteEnvironment):
        self.actions = environment.list_actions()

    @abstractmethod
    def get_action(self, observation: list) -> int:
        pass

    @abstractmethod
    def print(self) -> None:
        pass
