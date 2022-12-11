from typing import TypeVar, Generic, List, Callable, Optional
from functools import partial

from ..numdicts import NumDict
from ..base.constructs import Process
from ..base.symbols import Symbol

import pyClarion as cl


T = TypeVar("T", bound=Symbol)


class Constants(Process):
    data: NumDict

    def __init__(self, path: str = "") -> None:
        super().__init__(path)
        self.data = NumDict()

    def initial(self) -> NumDict:
        return self.data

    call = initial


class Relay(Process, Generic[T]):
    """Copies signal from a single source."""

    def initial(self) -> NumDict[T]:
        return NumDict()

    def call(self, d: NumDict[T]) -> NumDict[T]:
        return d


class Shift(Process, Generic[T]):
    """Shifts symbols by one time step."""
    lag: Callable
    lead: bool
    max_lag: int
    min_lag: int

    def __init__(
        self, 
        path: str = "", 
        inputs: Optional[List[str]] = None, 
        lead: bool = False, 
        max_lag: int = 1, 
        min_lag: int = 0
    ) -> None:
        """
        Initialize a new `Lag` propagator.

        :param lead: Whether to lead (or lag) features.
        :param max_lag: Drops features with lags greater than this value.
        :param min_lag: Drops features with lags less than this value.
        """
        super().__init__(path, inputs)
        self.lag = partial(Symbol.lag, val=1 if not lead else -1)
        self.min_lag = min_lag
        self.max_lag = max_lag

    def initial(self) -> NumDict[T]:
        return NumDict()

    def call(self, d: NumDict[T]) -> NumDict[T]:
        return d.transform_keys(kf=self.lag).keep(sf=self._filter)

    def _filter(self, s: T) -> bool:
        return self.min_lag <= s.l <= self.max_lag
