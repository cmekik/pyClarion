from datetime import timedelta

from ..numdicts import Index, NumDict, numdict, path
from ..numdicts import root as get_root
from ..knowledge import (Family, Chunks, Rules, Chunk, Rule, 
    compile_chunks, compile_rules, ByKwds)
from ..system import Process, UpdateSite, UpdateSort, Event, Priority
from .elementary import TopDown, BottomUp


class Associations(Process):
    main: NumDict
    input: NumDict
    weights: NumDict
    sum_by: ByKwds

    def __init__(self, name: str, input: Chunks, output: Chunks) -> None:
        super().__init__(name)
        root = self.system.root
        if not root == get_root(input) == get_root(output):
            raise ValueError("Mismatched root keyspaces")
        k0 = path(input); k1 = path(output)
        idx_m = Index(root, k0, (1,))
        idx_i = Index(root, k1, (1,))
        idx_w = Index(root, k1.link(k0, 0), (1, 1))
        self.main = numdict(idx_m, {}, 0.0)
        self.input = numdict(idx_i, {}, 0.0)
        self.weights = numdict(idx_w, {}, 0.0)
        self.by = ByKwds(by=idx_m.keyform, b=1)

    def resolve(self, event: Event) -> None:
        if event.affects(self.weights):
            self.update(priority=Priority.LEARNING)
        elif event.affects(self.input):
            self.update(priority=Priority.PROPAGATION)

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        output = self.weights.mul(self.input).sum(**self.sum_by)
        self.system.schedule(self.update, 
            UpdateSite(self.main, output.d), 
            dt=dt, priority=priority)


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
        if not root == get_root(tl) == get_root(bl1) == get_root(bl2):
            raise ValueError("Mismatched root keyspaces")
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
        if event.source == self.bu.update:
            self.update()
        if event.source == self.compile:
            self.update_buw()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        result = self.ciw.mul(self.bu.main, bs=(1,)).max(**self.max_by)
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

    def __init__(self, 
        name: str, 
        tl: Family, 
        bli0: Family, 
        bli1: Family, 
        blo0: Family, 
        blo1: Family
    ) -> None:
        super().__init__(name)
        root = self.system.root
        if not root == get_root(tl) == get_root(bli0) == get_root(bli1) \
            == get_root(blo0) == get_root(blo1):
            raise ValueError("Mismatched root keyspaces")
        self.rules = Rules()
        tl[name] = self.rules
        with self:
            self.lhs = ChunkStore(f"{name}_l", tl, bli0, bli1)
            self.rhs = ChunkStore(f"{name}_r", tl, blo0, blo1)
        kr = path(self.rules)
        k_lhs = path(self.lhs.chunks)
        k_rhs = path(self.rhs.chunks)
        idx_m = Index(root, kr, (1,))
        idx_i = Index(root, kr.link(kr, 0), (1, 1))
        idx_lhs = Index(root, kr.link(k_lhs, 0), (1, 1))
        idx_rhs = Index(root, kr.link(k_rhs, 0), (1, 1))
        self.main = numdict(idx_m, {}, c=0.0)
        self.riw = numdict(idx_i, {}, c=float("nan"))
        self.lhw = numdict(idx_lhs, {}, c=float("nan"))
        self.rhw = numdict(idx_rhs, {}, c=float("nan"))

    def resolve(self, event: Event) -> None:
        if event.source == self.lhs.bu.update:
            self.update()
        if event.source == self.compile:
            self.lhs.update_buw()
            self.rhs.update_buw()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        main = self.lhw.mul(self.lhs.bu.main).max(by=self.main.i.keyform)
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
