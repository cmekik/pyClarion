__all__ = ["ChunkStore", "BottomUp", "TopDown"]


from ...knowledge.sorts import Chunks
from ...knowledge.types import DVPairs
from ...events.updates import ForwardUpdate
from ...events.system import Process, Action
from ...events.sites import Site, State
from ...numdicts.keyspaces import KeyForm, keyform
from ...numdicts.sentinel import Sentinel, UndefinedType
from ...numdicts.numdicts import NumDict


class ChunkStore[C: Chunks, D: DVPairs](Process):
    """
    A chunk store. 

    Maintains a collection of chunks and facilitates top-down and bottom-up 
    activation propagation.
    """

    c: C
    d: D
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

    def __init__(self, name: str, c: C, d: D) -> None:
        super().__init__(name)
        l = self.system.depth
        b, v = d.paths
        self.system.check_root(c, d)
        idx_c = self.system.get_index(keyform(c))
        idx_d = self.system.get_index(keyform(d))
        ciw, tdw = self.initialize_chunk_weights(c)
        self.c = c
        self.d = d
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
    def initialize_chunk_weights(c: C):
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


class BottomUp[S: ChunkStore](Action[S]):
    def __init__(self, source: S) -> None:
        super().__init__(source)
        output = (source.buw[0]
            .mul(source.input_bu[0], by=source.by_bus_value)
            .max(by=source.by_dim)
            .sum(by=source.dim_by_chunk, c=0.0))
        self.updates.append(ForwardUpdate(source.output_bu, output))


class TopDown[S: ChunkStore](Action[S]):
    def __init__(self, source: S) -> None:
        super().__init__(source)
        output = (source.tdw[0]
            .mul(source.input_td[0], by=source.by_chunk)
            .sum(by=source.by_bus_value))
        self.updates.append(ForwardUpdate(source.output_td, output))
