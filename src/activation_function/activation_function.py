from typing import Callable, NamedTuple, TypeVar

T = TypeVar("T")


class ActivationFunction(NamedTuple):
    activation_function: Callable[[T], T]
    derivative: Callable  # TODO: stricter typing
