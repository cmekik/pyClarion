from datetime import timedelta

from ..numdicts import Index, NumDict, numdict, path, root
from ..knowledge import (Family, Chunks, Rules, Chunk, Rule, 
    compile_chunks, compile_rules, ByKwds)
from ..system import Process, UpdateSite, UpdateSort, Event, Priority
from .elementary import TopDown, BottomUp


class ChunkStore(Process):
    chunks: Chunks
    main: NumDict
    ciw: NumDict
    td: TopDown
    bu: BottomUp
    max_by: ByKwds

    def __init__(self, name: str, tl: Family, bl1: Family, bl2: Family) -> None:
        super().__init__(name)
        root = self.system.root
        self.chunks = Chunks()
        tl[name] = self.chunks; kt = path(self.chunks)
        idx_m = Index(root, kt, (1,))
        idx_w = Index(root, kt.link(kt, 0), (1, 1))
        self.main = numdict(idx_m, {}, c=0.0)
        self.ciw = numdict(idx_w, {}, c=float("nan"))
        self.max_by = ByKwds(by=self.main.i.keyform, b=0)
        with self:
            self.td = TopDown(f"{name}_td", self.chunks, bl1, bl2)
            self.bu = BottomUp(f"{name}_bu", self.chunks, bl1, bl2)

    def norm(self, d: NumDict) -> NumDict:
        return (d
            .abs()
            .max(by=self.bu.max_by)
            .sum(by=self.bu.main.i.keyform)
            .shift(x=1.0))

    def resolve(self, event: Event) -> None:
        if event.affects(self.bu.main):
            self.update()
        if event.affects(self.td.weights):
            self.update_buw()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        result = self.ciw.mul(self.bu.main, bs=(0,)).max(**self.max_by)
        self.system.schedule(self.update, 
            UpdateSite(self.main, result.d),
            dt=dt, priority=priority)
        
    def compile(self, *chunks: Chunk, 
        dt: timedelta = timedelta(), 
        priority=Priority.LEARNING
    ) -> None:
        new, data = compile_chunks(*chunks, sort=self.chunks)
        self.system.schedule(
            self.compile, 
            UpdateSort(self.chunks, add=tuple((c._name_, c) for c in new)),
            UpdateSite(self.ciw, data["ciw"], reset=False), 
            UpdateSite(self.td.weights, data["tdw"], reset=False),
            dt=dt, priority=priority)

    def update_buw(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.LEARNING
    ) -> None:
        weights = self.td.weights.div(self.norm(self.td.weights))
        self.system.schedule(
            self.update_buw, 
            UpdateSite(self.bu.weights, weights.d), 
            dt=dt, priority=priority)


class RuleStore(Process):
    rules: Rules
    lhs: ChunkStore
    rhs: ChunkStore
    main: NumDict
    riw: NumDict
    lhw: NumDict
    rhw: NumDict

    def __init__(self, name: str, tl: Family, bl1: Family, bl2: Family) -> None:
        super().__init__(name)
        root_ = self.system.root
        if not root_ == root(tl) == root(bl1) == root(bl2):
            raise ValueError("Mismatched keyspace roots.")
        self.rules = Rules()
        tl[name] = self.rules
        with self:
            self.lhs = ChunkStore(f"{name}_l", tl, bl1, bl2)
            self.rhs = ChunkStore(f"{name}_r", tl, bl1, bl2)
        kr = path(self.rules)
        k_lhs = path(self.lhs.chunks)
        k_rhs = path(self.rhs.chunks)
        idx_m = Index(root_, kr, (1,))
        idx_i = Index(root_, kr.link(kr, 0), (1, 1))
        idx_lhs = Index(root_, kr.link(k_lhs, 0), (1, 1))
        idx_rhs = Index(root_, kr.link(k_rhs, 0), (1, 1))
        self.main = numdict(idx_m, {}, c=0.0)
        self.riw = numdict(idx_i, {}, c=float("nan"))
        self.lhw = numdict(idx_lhs, {}, c=float("nan"))
        self.rhw = numdict(idx_rhs, {}, c=float("nan"))

    def resolve(self, event: Event) -> None:
        if event.affects(self.lhs.bu.main):
            self.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        main = self.lhw.mul(self.lhs.bu.main).max(by=self.main.i.keyform, b=0)
        self.system.schedule(self.update, UpdateSite(self.main, main.d), 
            dt=dt, priority=priority)

    def compile(self, *rules: Rule, 
        dt: timedelta = timedelta(),
        priority: int = Priority.LEARNING) -> None:
        new, data = compile_rules(*rules, 
            sort=self.rules, lhs=self.lhs.chunks, rhs=self.rhs.chunks)
        lhs = []; rhs = []
        for rule in new:
            chunks = list(rule._chunks_)
            lhs.extend(chunks[:-1]); rhs.append(chunks[-1])
        self.system.schedule(
            self.compile, 
            UpdateSort(self.lhs.chunks, add=tuple((c._name_, c) for c in lhs)),
            UpdateSort(self.rhs.chunks, add=tuple((c._name_, c) for c in rhs)),
            UpdateSort(self.rules, add=tuple((r._name_, r) for r in new)),
            UpdateSite(self.lhs.ciw, data["lhs"]["ciw"], reset=False),
            UpdateSite(self.rhs.ciw, data["rhs"]["ciw"], reset=False), 
            UpdateSite(self.lhs.td.weights, data["lhs"]["tdw"], reset=False), 
            UpdateSite(self.rhs.td.weights, data["rhs"]["tdw"], reset=False),
            UpdateSite(self.riw, data["riw"], reset=False),
            UpdateSite(self.lhw, data["lhw"], reset=False),
            UpdateSite(self.rhw, data["rhw"], reset=False),
            dt=dt, priority=priority)
