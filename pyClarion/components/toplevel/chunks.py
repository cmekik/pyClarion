__all__ = ["ChunkStore", "BottomUp", "TopDown"]


from typing import cast

from ...knowledge.base import Root
from ...knowledge.explicit import Chunks
from ...events.updates import ForwardUpdate
from ...events.system import Action, Event, Process
from ...events.sites import Site, State
from ...numdicts.keyspaces import KeyForm, keyform, find_root
from ...numdicts.sentinel import Sentinel, UndefinedType
from ...numdicts.numdicts import NumDict


class ChunkStore[C: Chunks](Process[Root]):
    """
    A chunk store. 

    Maintains a collection of chunks and facilitates top-down and bottom-up 
    activation propagation.
    """

    output_bu: Site[float] = Site()
    output_td: Site[float] = Site() 
    input_bu: Site[float] = Site()
    input_td: Site[float] = Site()
    ciw: Site[float] = Site()
    tdw: Site[float] = Site()
    buw: Site[UndefinedType] = Site()
    weights_by_chunk: KeyForm
    weights_by_value: KeyForm
    weights_by_dim: KeyForm
    dims_by_chunk: KeyForm

    @property
    def c(self) -> C:
        return self._c

    def __init__(self, name: str, c: C) -> None:
        super().__init__(name, cast(Root | None, find_root(c)))
        d = c._dyads_
        l = self.system.depth
        b, v = d.units
        self.system.check_root(c, d)
        idx_c = self.system.get_index(keyform(c))
        idx_d = self.system.get_index(keyform(d))
        ciw, tdw = self.initialize_chunk_weights(c)
        self._c = c
        self.weights_by_chunk = keyform(c) * keyform(b).agg * keyform(v).agg
        self.weights_by_value = keyform(c).agg * keyform(d)
        self.weights_by_dim = keyform(c) * keyform(b) * keyform(v, -1)
        self.dims_by_chunk = keyform(c) * keyform(b).agg * keyform(v, -1).agg
        self.output_bu = State(idx_c, {}, 0.0, l)
        self.output_td = State(idx_d, {}, 0.0, l)
        self.input_bu = State(idx_d, {}, 0.0, l)
        self.input_td = State(idx_c, {}, 0.0, l)
        self.ciw = State(idx_c * idx_c, ciw, 0.0, l)
        self.tdw = State(idx_c * idx_d, tdw, 0.0, l)
        buw = self.tdw[0].div(self.norm(self.tdw[0])).d
        self.buw = State(idx_c * idx_d, buw, Sentinel.UNDEFINED, l)

    def initialize_chunk_weights(self, c: Chunks):
        ciw, tdw = {}, {}
        for name in c:
            chunk = c[name]
            _tdw, _ciw = chunk.compile()
            ciw.update(_ciw)
            tdw.update(_tdw)
        return ciw, tdw

    def norm(self, d: NumDict) -> NumDict:
        return (d
            .pow(2)
            .max(by=self.weights_by_dim)
            .sum(by=self.dims_by_chunk)
            .shift(1.0)) 

    def resolve(self, event: Event) -> None:
        match event.action:
            case BottomUp(p) | TopDown(p) if p is self:
                pass
            case _:
                super().resolve(event)


type AnyChunkStore = ChunkStore[Chunks]


class BottomUp[S: AnyChunkStore](Action[S]):
    def __init__(self, source: S) -> None:
        super().__init__(source)
        output = (source.buw[0]
            .mul(source.input_bu[0], by=source.weights_by_value)
            .max(by=source.weights_by_dim)
            .sum(by=source.dims_by_chunk, c=0.0))
        self.updates.append(ForwardUpdate(source.output_bu, output))


class TopDown[S: AnyChunkStore](Action[S]):
    def __init__(self, source: S) -> None:
        super().__init__(source)
        output = (source.tdw[0]
            .mul(source.input_td[0], 
                 by=source.weights_by_chunk)
            .sum(by=source.weights_by_value))
        self.updates.append(ForwardUpdate(source.output_td, output))
