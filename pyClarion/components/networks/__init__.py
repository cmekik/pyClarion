from typing import Type, Sequence, Self, Any
from datetime import timedelta

from .base import LayerBase, Optimizer, Activation
from .optimizers import SGD
from .errors import ErrorSignal
from .layers import SingleLayer, InputLayer, OutputLayer, Layer
from ..elementary import ChoiceBL
from ...system import Process, Event, Site, Priority
from ...knowledge import Family, Sort, Atoms, Atom
from ...numdicts import NumDict


class MLP(Process):
    class Hidden(Atoms):
        pass

    input: Site
    ilayer: LayerBase
    olayer: LayerBase
    layers: list[LayerBase]
    optimizer: Optimizer

    def __init__(self, 
        name: str, 
        p: Family,
        h: Family, 
        d1: Family | Sort | Atom, 
        v1: Family | Sort,
        d2: Family | Sort | Atom | None = None,
        v2: Family | Sort | None = None,
        layers: Sequence[int] = (),
        optimizer: Type[Optimizer] = SGD, 
        f: Activation = NumDict.eye,
        lags: int = 0,
        **kwargs: Any
    ) -> None:
        d2 = d1 if d2 is None else d2
        v2 = v1 if v2 is None else v2
        super().__init__(name)
        self.system.check_root(h)
        self.optimizer = optimizer(f"{name}.optimizer", p, **kwargs)
        with self.optimizer:
            if not layers:
                self.ilayer = self.olayer = SingleLayer(
                    f"{name}.layer", d1, v1, d2, v2, f=f, lags=lags)
            else:
                hs = []
                for i, n in enumerate(layers):
                    hs.append(self._mk_hidden_nodes(h, i, n))
                self.ilayer = InputLayer(
                    f"{name}.ilayer", d1, v1, hs[0], f=f, lags=lags)
                h_in = hs[0]
                layer = self.ilayer 
                for i, h_out in enumerate(hs[1:]):
                    layer = layer >> Layer(
                        f"{name}.l{i}", h_in, h_out, f=f, lags=lags)
                    self.layers.append(layer)
                    h_in = h_out
                self.olayer = layer >> OutputLayer(
                    f"{name}.olayer", h_in, d2, v2, lags=lags)
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
            self.ilayer.input = other
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
            self.ilayer.input.update(self.input[0]),
            dt=dt, priority=priority)
