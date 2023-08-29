from abc import ABC, abstractmethod
from typing import Any


class Restriction(ABC):
    @abstractmethod
    def is_ok(self, state: tuple) -> bool:
        pass

    @abstractmethod
    def process_action(self, action: int):
        pass

    @abstractmethod
    def print(self):
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        pass


class TimeRestriction(Restriction):

    def __init__(self, max_time: int):
        self.max_time = max_time
        self.remaining_time = max_time

    def is_ok(self, state: tuple) -> bool:
        return self.remaining_time < self.max_time

    def process_action(self, action: int):
        self.remaining_time -= 1

    def print(self):
        print("Remaining time: ", self.remaining_time)

    def get_used_time(self):
        return self.max_time - self.remaining_time

    def get_stats(self) -> dict[str, Any]:
        return {'used_time': self.get_used_time()}