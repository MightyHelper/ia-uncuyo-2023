from abc import ABC, abstractmethod


class DiscreteEnvironment(ABC):
    actions: list[str] = []
    def __init__(self, actions: list[str]):
        self.actions = actions

    def list_actions(self)-> list:
        """Return a list of actions that the agent can perform"""
        return self.actions

    @abstractmethod
    def accept_action(self, action) -> tuple:
        pass

    @abstractmethod
    def initial_state(self) -> tuple:
        pass

    @abstractmethod
    def get_performance(self) -> float:
        pass

    @abstractmethod
    def print(self) -> None:
        pass
