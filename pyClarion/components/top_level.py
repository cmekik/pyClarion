from itertools import count
from datetime import timedelta

from ..numdicts import Key, NumDict, numdict, path
from ..knowledge import Family, Term, Chunks, Rules, Chunk, Rule, Monads, Dyads, instantiations, ByKwds
from ..system import Process, UpdateSite, UpdateSort, Event
from .elementary import TopDown, BottomUp


class ChunkStore(Process):
    chunks: Chunks
    counter: count
    main: NumDict
    ciw: NumDict
    td: TopDown
    bu: BottomUp
    max_by: ByKwds

    def __init__(self, name: str, family: Family, bl: Dyads) -> None:
        super().__init__(name)
        self.chunks = Chunks()
        self.counter = count()
        family[name] = self.chunks
        tl = Monads(self.chunks)
        idx_w = Dyads(self.chunks, self.chunks)
        self.main = numdict(tl, {}, c=0.0)
        self.ciw = numdict(idx_w, {}, c=float("nan"))
        self.max_by = idx_w.aggr(0)
        with self:
            self.td = TopDown(f"{name}_td", tl, bl)
            self.bu = BottomUp(f"{name}_bu", tl, bl)

    def norm(self, d: NumDict) -> NumDict:
        return d.abs().max(by=self.bu.max_by).sum(**self.bu.sum_by).shift(x=1.0)

    def resolve(self, event: Event) -> None:
        if event.source == self.bu.update:
            self.update()
        if event.affects(self.td.weights):
            self.update_buw()

    def update(self, dt: timedelta = timedelta()) -> None:
        result = self.ciw.mul(self.bu.main, bs=(0,)).max(**self.max_by)
        self.system.schedule(
            self.update,
            UpdateSite(self.main, result.d),
            dt=dt)
        
    def compile(self, *chunks: Chunk, dt: timedelta = timedelta()) -> None:
        new_chunks = []; k = path(self.chunks)
        ciw = {}; tdw = {}
        for chunk in chunks:
            chunk._instances_.extend(instantiations(chunk))
            name = chunk._descr_ or f"c{next(self.counter)}"
            new_chunks.append((name, chunk))
            kc = k.link(Key(name), k.size)
            if chunk._instances_:
                for inst in chunk._instances_:
                    name = f"c{next(self.counter)}"
                    new_chunks.append((name, inst))
                    ki = k.link(Key(name), k.size)
                    ciw[kc.link(ki, 0)] = 1.0
                    for (s1, s2), w in inst._dyads_.items():
                        assert isinstance(s1, Term) and isinstance(s2, Term)
                        kw = ki.link(path(s1), 0).link(path(s2), 0)
                        tdw[kw] = w
            else:
                ciw[kc.link(kc, 0)] = 1.0
                for (s1, s2), w in chunk._dyads_.items():
                    assert isinstance(s1, Term) and isinstance(s2, Term)
                    kw = kc.link(path(s1), 0).link(path(s2), 0)
                    tdw[kw] = w
        self.system.schedule(
            self.compile, 
            UpdateSort(self.chunks, add=tuple(new_chunks)),
            UpdateSite(self.ciw, ciw, reset=False), 
            UpdateSite(self.td.weights, tdw, reset=False),
            dt=dt)

    def update_buw(self, dt: timedelta = timedelta()) -> None:
        weights = self.td.weights.div(self.norm(self.td.weights))
        self.system.schedule(
            self.update_buw, 
            UpdateSite(self.bu.weights, weights.d), 
            dt=dt)


# class RuleStore(Process):
#     rules: Rules
#     main: NumDict
#     riw: NumDict
#     lhw: NumDict
#     rhw: NumDict
#     lhs: ChunkStore
#     rhs: ChunkStore

#     def __init__(self, name: str, family: Family, bl: Dyads) -> None:
#         super().__init__(name)
#         self.rules = Rules()
#         family[name] = self.rules
#         with self:
#             self.lhs = ChunkStore(f"{name}_l", family, bl)
#             self.rhs = ChunkStore(f"{name}_r", family, bl)
#         self.main = numdict(Monads(self.rules), {}, c=0.0)
#         self.riw = numdict(Dyads(self.rules, self.rules), {}, c=float("nan"))
#         self.lhw = numdict(Dyads(self.rules, self.lhs.chunks), {}, c=float("nan"))
#         self.rhw = numdict(Dyads(self.rules, self.rhs.chunks), {}, c=float("nan"))

#     def resolve(self, event: Event) -> None:
#         if event.source == self.lhs.bu.update:
#             self.update(event.time)

#     def update(self, dt: timedelta = timedelta()) -> None:
#         result = self.riw.mul(self.lhs.main).max(by=self.main.i.keyform)
#         self.system.schedule(
#             self.update,
#             UpdateSite(self.main, result.d),
#             dt=dt)