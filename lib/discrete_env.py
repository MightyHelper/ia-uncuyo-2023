from abc import ABC, abstractmethod
from typing import Any

from lib.restriction import Restriction


class DiscreteEnvironment(ABC):
    actions: list[str] = []
    restrictions: list[Restriction] = []

    def __init__(self, actions: list[str]):
        self.actions = actions

    def list_actions(self) -> list:
        """Return a list of actions that the agent can perform"""
        return self.actions

    def add_restriction(self, restriction: Restriction):
        self.restrictions.append(restriction)

    def restrictions_ok(self) -> bool:
        for restriction in self.restrictions:
            if not restriction.is_ok(self.get_state()):
                return False
        return True

    def process_action(self, action: int) -> tuple:
        for restriction in self.restrictions:
            restriction.process_action(action)
        self._accept_action(action)
        return self.get_state()

    def is_done(self):
        return not self.restrictions_ok() or self.objective_reached()

    def find_failing_restriction(self):
        for restriction in self.restrictions:
            if not restriction.is_ok(self.get_state()):
                return restriction
        return None

    @abstractmethod
    def initial_state(self) -> tuple:
        """Return the initial state of the environment"""
        pass

    @abstractmethod
    def get_performance(self) -> float:
        """Return the performance of the agent in the environment"""
        pass

    def print(self) -> None:
        print(self.get_state())
        for restriction in self.restrictions:
            restriction.print()

    @abstractmethod
    def get_state(self) -> Any:
        """Return the current state of the environment"""
        pass

    @abstractmethod
    def _accept_action(self, action) -> None:
        """Return the new state of the environment after accepting the action"""
        pass

    @abstractmethod
    def objective_reached(self) -> bool:
        """Return true if the objective of the environment was reached"""
        pass

    def get_restriction(self, r_type: type[Restriction]) -> Restriction | None:
        for restriction in self.restrictions:
            if isinstance(restriction, r_type):
                return restriction
        return None
