from typing import TypedDict, LiteralString
from itertools import count
from datetime import timedelta

from ..numdicts import Key, NumDict, numdict, path
from ..knowledge import Family, Term, Chunks, Rules, Chunk, Rule, Monads, Dyads, instantiations, ByKwds
from ..system import Process, UpdateSite, UpdateSort, Event
from .elementary import TopDown, BottomUp


class ChunkWeights(TypedDict):
    ciw: dict[Key, float]
    tdw: dict[Key, float]


def compile_chunk(chunk: Chunk, k: Key, counter: count) \
    -> tuple[str, ChunkWeights]:
    ciw = {}; tdw = {}
    chunk._instances_.extend(instantiations(chunk))
    name = chunk._descr_ or f"c{next(counter)}"
    if not chunk._instances_:
        kc = k.link(Key(name), k.size)
        ciw[kc.link(kc, 0)] = 1.0
        for (s1, s2), w in chunk._dyads_.items():
            assert isinstance(s1, Term) and isinstance(s2, Term)
            kw = kc.link(path(s1), 0).link(path(s2), 0)
            tdw[kw] = w 
    return name, ChunkWeights(ciw=ciw, tdw=tdw)


def compile_chunks(*chunks: Chunk, sort: Chunks, counter: count) \
    -> tuple[list[tuple[LiteralString, Chunk]], ChunkWeights]:
    new_chunks = []; k = path(sort)
    ciw = {}; tdw = {}
    for chunk in chunks:
        name, d_ws = compile_chunk(chunk, k, counter)
        new_chunks.append((name, chunk))
        ciw.update(d_ws["ciw"]); tdw.update(d_ws["tdw"])
        if chunk._instances_:
            kc = k.link(Key(name), k.size)
            for inst in chunk._instances_:
                name, d_ws = compile_chunk(inst, k, counter)
                new_chunks.append((name, inst))
                ciw.update(d_ws["ciw"]); tdw.update(d_ws["tdw"])
                ki = k.link(Key(name), k.size)
                ciw[kc.link(ki, 0)] = 1.0
    return new_chunks, ChunkWeights(ciw=ciw, tdw=tdw)


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
        new, d_ws = compile_chunks(
            *chunks, 
            sort=self.chunks, 
            counter=self.counter)
        self.system.schedule(
            self.compile, 
            UpdateSort(self.chunks, add=tuple(new)),
            UpdateSite(self.ciw, d_ws["ciw"], reset=False), 
            UpdateSite(self.td.weights, d_ws["tdw"], reset=False),
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
#         self.counter = count()
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

#     def compile(self, *rules: Rule, dt: timedelta = timedelta()) -> None:
#         new_rules = []; k = path(self.rules)
#         conditions = []; actions = []
#         riw = {}
#         for rule in rules:
#             rule._instances_.extend(instantiations(rule))
#             name = rule._descr_ or f"r{next(self.counter)}"
#             new_rules.append((name, rule))
#             kr = k.link(Key(name), k.size)
#             if rule._instances_:
#                 for inst in rule._instances_:
#                     name = f"r{next(self.counter)}"
#                     new_rules.append((name, inst))
#                     ki = k.link(Key(name), k.size)
#                     riw[kr.link(ki, 0)] = 1.0
#                     conditions.extend(inst._chunks_)
#                     actions.append(conditions.pop())
#             else:
#                 conditions.extend(rule._chunks_)
#                 actions.append(conditions.pop())
            