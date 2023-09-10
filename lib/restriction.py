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


class DummyRestriction(Restriction):
    def __init__(self, output):
        self.output = output

    def is_ok(self, state: tuple) -> bool:
        return True

    def process_action(self, action: int):
        pass

    def print(self):
        print(f"DummyRestriction")

    def get_stats(self) -> dict[str, Any]:
        return self.output


class TimeRestriction(Restriction):

    def __init__(self, max_time: int):
        self.max_time = max_time
        self.remaining_time = max_time

    def is_ok(self, state: tuple) -> bool:
        return self.remaining_time > 0

    def process_action(self, action: int):
        self.remaining_time -= 1

    def print(self):
        print("Remaining time: ", self.remaining_time)

    def get_used_time(self):
        return self.max_time - self.remaining_time

    def get_stats(self) -> dict[str, Any]:
        return {'used_time': self.get_used_time()}

    def __str__(self):
        return f"TimeRestriction({self.remaining_time}/{self.max_time})"
