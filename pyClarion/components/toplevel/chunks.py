__all__ = ["ChunkStore", "ChunkStoreParams", "BottomUp", "TopDown"]


import random

from datetime import timedelta
from typing import Final

from ..base.parametric import Parametric
from ...knowledge.terms import Param
from ...knowledge.sorts import Chunks, Params
from ...knowledge.types import DVPairs
from ...events.updates import ForwardUpdate
from ...events.system import Action
from ...events.sites import Site, State
from ...numdicts.keyspaces import KeyForm, keyform
from ...numdicts.sentinel import Sentinel, UndefinedType
from ...numdicts.numdicts import NumDict


class ChunkStoreParams(Params):
    bu_latency: Param
    td_latency: Param
    _init_val_defaults_ = {"bu_latency": .01, "td_latency": .01}


class ChunkStore[C: Chunks, P: ChunkStoreParams](Parametric[P]):
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
    by_chunk: KeyForm
    by_bus_value: KeyForm
    by_dim: KeyForm
    dim_by_chunk: KeyForm

    @property
    def c(self) -> C:
        return self._c

    def __init__(self, name: str, p: P, c: C) -> None:
        super().__init__(name, p)
        d = c._dyads_
        l = self.system.depth
        b, v = d.paths
        self.system.check_root(c, d)
        idx_c = self.system.get_index(keyform(c))
        idx_d = self.system.get_index(keyform(d))
        ciw, tdw = self.initialize_chunk_weights(c)
        self._c = c
        self.by_chunk = keyform(c) * keyform(b).agg * keyform(v).agg
        self.by_bus_value = keyform(c).agg * keyform(d)
        self.by_dim = keyform(c) * keyform(b) * keyform(v, -1)
        self.dim_by_chunk = keyform(c) * keyform(b).agg * keyform(v, -1).agg
        self.output_bu = State(idx_c, {}, 0.0, l)
        self.output_td = State(idx_d, {}, 0.0, l)
        self.input_bu = State(idx_d, {}, 0.0, l)
        self.input_td = State(idx_c, {}, 0.0, l)
        self.ciw = State(idx_c * idx_c, ciw, 0.0, l)
        self.tdw = State(idx_c * idx_d, tdw, 0.0, l)
        buw = self.tdw[0].div(self.norm(self.tdw[0])).d
        self.buw = State(idx_c * idx_d, buw, Sentinel.UNDEFINED, l)

    @staticmethod
    def initialize_chunk_weights(c: Chunks):
        ciw, tdw = {}, {}
        for name in c:
            chunk = c[name]
            data = chunk._compile_()
            ciw.update(data["ciw"])
            tdw.update(data["tdw"])
        return ciw, tdw

    def norm(self, d: NumDict) -> NumDict:
        return (d
            .pow(2)
            .max(by=self.by_dim)
            .sum(by=self.dim_by_chunk)
            .shift(1.0)) 


type AnyChunkStore = ChunkStore[Chunks, ChunkStoreParams]


class BottomUp[S: AnyChunkStore](Action[S]):
    def __init__(self, source: S) -> None:
        super().__init__(source)
        output = (source.buw[0]
            .mul(source.input_bu[0], by=source.by_bus_value)
            .max(by=source.by_dim)
            .sum(by=source.dim_by_chunk, c=0.0))
        latency = source.params[0][~source.p.bu_latency]
        self.updates.append(ForwardUpdate(source.output_bu, output))
        if 0 < latency:
            self.duration = timedelta(seconds=random.expovariate(1 / latency))


class TopDown[S: AnyChunkStore](Action[S]):
    def __init__(self, source: S) -> None:
        super().__init__(source)
        output = (source.tdw[0]
            .mul(source.input_td[0], by=source.by_chunk)
            .sum(by=source.by_bus_value))
        latency = random.expovariate(1/source.params[0][~source.p.td_latency])
        self.updates.append(ForwardUpdate(source.output_td, output))
        if 0 < latency:
            self.duration = timedelta(seconds=random.expovariate(1 / latency))
