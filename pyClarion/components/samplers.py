from typing import ClassVar, List, Tuple, TypeVar, Optional

from ..numdicts import NumDict
from ..base.constructs import Process
from ..base.symbols import F, V, D
from .. import sym


class ActionSampler(Process):
    """
    Selects actions and relays action paramaters.

    Expects to be linked to an external 'cmds' fspace.

    Actions are selected for each command dimension according to a Boltzmann 
    distribution.
    """
    min_temp: float
    _prms: NumDict[F]
    _d_temp: ClassVar[str] = "temp"

    def __init__(
        self, 
        path: str = "", 
        inputs: Optional[List[str]] = None, 
        min_temp: float = 1e-8
    ) -> None:
        super().__init__(path, inputs)
        self.min_temp = min_temp
        self.__validate()
        self._prms = NumDict({f: 1.0 for f in self._init_prms()})

    def __validate(self) -> None:
        if self.min_temp <= 0:
            raise ValueError("Arg 'min_temp' must be strictly greater "
                "than zero.")

    def initial(self) -> Tuple[NumDict[F], NumDict[F], NumDict[F]]:
        return NumDict(), NumDict(), self._prms

    def call(
        self, p: NumDict[F], d: NumDict[V], a: NumDict[V]
    ) -> Tuple[NumDict[V], NumDict[V], NumDict[F]]:
        """
        Select actions for each client command dimension.
        
        :param p: Selection parameters (temperature). Temperature may be set to 
            a minimum value of self._epsilon
        :param d: Action feature strengths.
        :param a: Action feature indicators.

        :returns: tuple (actions, distributions)
            where
            actions sends selected actions to 1 and everything else to 0, and
            distributions contains the sampling probabilities for each action
        """
        dims = sym.group_by_dims(a)
        temp = p.isolate(key=F(self._d_temp, p=self.path)).max(self.min_temp)
        _dists, _actions = [], [] 
        for fs in dims.values():
            dist = d.with_keys(ks=fs).boltzmann(temp)
            _dists.append(dist)
            _actions.append(dist.sample().squeeze())
        dists = NumDict().merge(*_dists) 
        actions = NumDict().merge(*_actions)
        return actions, dists, self._prms

    def _init_prms(self) -> Tuple[D, ...]:
        return (F(self._d_temp, p=self.path),)


T = TypeVar("T")
class BoltzmannSampler(Process):
    """Samples a node according to a Boltzmann distribution."""
    min_temp: float
    _prms: NumDict[F]
    _d_temp: ClassVar[str] = "temp"

    def __init__(
        self, 
        path: str = "", 
        inputs: Optional[List[str]] = None, 
        min_temp: float = 1e-8
    ) -> None:
        super().__init__(path, inputs)
        self.min_temp = min_temp
        self.__validate()
        self._prms = NumDict({f: 1.0 for f in self._init_prms()})

    def __validate(self) -> None:
        if self.min_temp <= 0:
            raise ValueError("Arg 'min_temp' must be strictly greater "
                "than zero.")        

    def initial(self) -> Tuple[NumDict[T], NumDict[T], NumDict[F]]:
        return NumDict(), NumDict(), self._prms

    def call(
        self, p: NumDict[F], d: NumDict[T]
    ) -> Tuple[NumDict[T], NumDict[T], NumDict[F]]:
        """
        Select a node through an activation-based competition. 
        
        Selection probabilities vary with node strengths according to a 
        Boltzmann distribution.

        :param p: Incoming parameters (temperature & threshold). 
        :param d: Incoming feature activations.
        """
        if len(d):
            dist = d.boltzmann(p.isolate(key=F(self._d_temp, p=self.path)))
            return dist.sample().squeeze(), dist, self._prms
        else:
            return self.initial()

    def _init_prms(self) -> Tuple[F, ...]:
        return (F(self._d_temp, p=self.path),)
