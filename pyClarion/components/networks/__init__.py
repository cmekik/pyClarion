from typing import Type, Sequence, Self, Any, Callable
from datetime import timedelta

from .base import Layer, Optimizer, ErrorSignal, Activation, Train
from .activations import Tanh
from .errors import TDError, Supervised, LeastSquares, Cost
from .optimizers import SGD

from ..base import D, V, DV
from ...system import Process, Event, Site, Priority
from ...knowledge import Family, Atoms, Atom
from ...numdicts import NumDict


__all__ = [
    "Train", "Layer", "Optimizer", "Activation", "Cost",
    "LeastSquares",
    "Tanh",
    "Supervised", "TDError",
    "SGD", 
    "MLP", "IDN"
]


class MLP(Process):
    """
    A Multi-Layer Perceptron.
    
    Learns input-output mappings by backpropagating error signals. 
    """

    class Hidden(Atoms):
        pass

    input: Site
    ilayer: Layer
    olayer: Layer
    layers: list[Layer]
    optimizer: Optimizer

    def __init__(self, 
        name: str, 
        p: Family,
        h: Family, 
        s1: V | DV,
        s2: V | DV | None = None,
        layers: Sequence[int] = (),
        optimizer: Type[Optimizer] = SGD, 
        afunc: Activation | None = None,
        l: int = 0,
        **kwargs: Any
    ) -> None:
        s2 = s1 if s2 is None else s2
        super().__init__(name)
        self.system.check_root(h)
        self.optimizer = optimizer(f"{name}.optimizer", p, **kwargs)
        with self.optimizer:
            if not layers:
                self.ilayer = Layer(f"{name}.layer", s1, s2, afunc=afunc, l=l)
                self.olayer = self.ilayer
            else:
                hs = []
                for i, n in enumerate(layers):
                    hs.append(self._mk_hidden_nodes(h, i, n))
                self.ilayer = Layer(f"{name}.ilayer", s1, hs[0], afunc=afunc, l=l)
                hi = hs[0]
                layer = self.ilayer 
                for i, ho in enumerate(hs[1:]):
                    layer = layer >> Layer(f"{name}.l{i}", hi, ho, afunc=afunc, l=l)
                    self.layers.append(layer)
                    hi = ho
                self.olayer = layer >> Layer(f"{name}.olayer", hi, s2, l=l)
        self.input = Site(self.ilayer.input.index, {}, self.ilayer.input.const)

    def _mk_hidden_nodes(self, h: Family, l: int, n: int) -> Hidden:
        hidden = type(self).Hidden()
        h[f"{self.name}.h{l}"] = hidden
        for _ in range(n):
            hidden[f"n{next(hidden._counter_)}"] = Atom()
        return hidden
    
    def __rshift__[T: Process](self: Self, other: T) -> T:
        if isinstance(other, ErrorSignal):
            return self.olayer >> other
        return NotImplemented
    
    def __rrshift__(self: Self, other: Site | Process) -> Self:
        if isinstance(other, Site):
            self.input = other
            return self
        if isinstance(other, Process):
            try:
                self.input = getattr(other, "main")
            except AttributeError:
                raise
            return self
        return NotImplemented

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.update()
        if event.source == self.ilayer.backward:
            self.optimizer.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        self.system.schedule(self.update, 
            self.ilayer.input.update(self.input[0].d),
            dt=dt, priority=priority)


class IDN(MLP):
    """
    An Implicit Decision Network (IDN).
    
    Learns to make action decisions in the bottom level via temporal difference 
    learning.
    """

    error: TDError

    def __init__(self, 
        name: str, 
        p: Family,
        h: Family,
        r: D | DV, 
        s1: V | DV,
        s2: V | DV | None = None,
        layers: Sequence[int] = (),
        optimizer: Type[Optimizer] = SGD,
        afunc: Activation | None = None,
        func: Callable[[TDError], NumDict] = TDError.qmax,
        gamma: float = .3,
        l: int = 1,
        **kwargs: Any
    ) -> None:
        super().__init__(name, p, h, s1, s2, layers, optimizer, afunc, l, **kwargs)
        self.error = self >> TDError(f"{name}.error", 
            p, r, func=func, gamma=gamma, l=l)
        
