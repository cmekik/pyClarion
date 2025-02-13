from datetime import timedelta
import logging

from ..numdicts import NumDict, numdict, KeyForm
from ..knowledge import (Family, Chunks, Rules, Chunk, Rule, 
    compile_chunks, compile_rules, ByKwds, keyform, Sort, Atom, describe)
from ..system import Process, UpdateSite, UpdateSort, Event, Priority
from .elementary import TopDown, BottomUp


class ChunkStore(Process):
    chunks: Chunks
    main: NumDict
    ciw: NumDict
    td: TopDown
    bu: BottomUp
    max_by: KeyForm

    def __init__(self, 
        name: str, 
        c: Family, 
        d: Family | Sort | Atom, 
        v: Family | Sort
    ) -> None:
        super().__init__(name)
        self.system.check_root(c, d, v)
        self.chunks = Chunks(); c[name] = self.chunks
        index = self.system.get_index(keyform(self.chunks))
        self.main = numdict(index, {}, c=0.0)
        self.ciw = numdict(index * index, {}, c=float("nan"))
        self.max_by = keyform(self.chunks) * keyform(self.chunks).agg 
        with self:
            self.td = TopDown(f"{name}.td", self.chunks,  d, v)
            self.bu = BottomUp(f"{name}.bu", self.chunks, d, v)

    def norm(self, d: NumDict) -> NumDict:
        return (d
            .abs()
            .max(by=self.bu.max_by)
            .sum(by=self.bu.sum_by)
            .shift(x=1.0))

    def resolve(self, event: Event) -> None:
        if event.source == self.bu.update:
            self.update()
        if event.source == self.compile:
            # This next check is probably not idiomatic, is there a way to 
            # avoid needlessly computing log data that is idiomatic?
            if self.system.logger.level <= logging.DEBUG:
                self.log_compilation(event)
            self.update_buw()

    def log_compilation(self, event: Event) -> None:
        if event.source != self.compile:
            raise ValueError()
        assert isinstance(event.updates[0], UpdateSort)
        assert event.updates[0].sort is self.chunks
        data = [f"    Added the following new chunk(s)"]
        for _, c in event.updates[0].add:
            data.append(describe(c).replace("\n", "\n    "))
        self.system.logger.debug("\n    ".join(data))

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        result = self.ciw.mul(self.bu.main, bs=(1,)).max(by=self.max_by)
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
        r: Family,
        c: Family, 
        d: Family | Sort | Atom, 
        v: Family | Sort, 
    ) -> None:
        super().__init__(name)
        self.system.check_root(r, d, v)
        self.rules = Rules(); r[name] = self.rules
        with self:
            self.lhs = ChunkStore(f"{name}.lhs", c, d, v)
            self.rhs = ChunkStore(f"{name}.rhs", c, d, v)
        idx_r = self.system.get_index(keyform(self.rules))
        idx_lhs = self.system.get_index(keyform(self.lhs.chunks))
        idx_rhs = self.system.get_index(keyform(self.rhs.chunks))
        self.main = numdict(idx_r, {}, c=0.0)
        self.riw = numdict(idx_r * idx_r, {}, c=float("nan"))
        self.lhw = numdict(idx_r * idx_lhs, {}, c=float("nan"))
        self.rhw = numdict(idx_r * idx_rhs, {}, c=float("nan"))

    def resolve(self, event: Event) -> None:
        if event.source == self.lhs.bu.update:
            self.update()
        if event.source == self.compile:
            # This next check is probably not idiomatic, is there a way to 
            # avoid needlessly computing log data that is idiomatic?
            if self.system.logger.level <= logging.DEBUG:
                self.log_compilation(event)
            self.lhs.update_buw()
            self.rhs.update_buw()

    def log_compilation(self, event: Event) -> None:
        if event.source != self.compile:
            raise ValueError()
        assert isinstance(event.updates[2], UpdateSort)
        assert event.updates[2].sort is self.rules
        data = [f"    Added the following new rule(s)"]
        for _, c in event.updates[2].add:
            data.append(describe(c).replace("\n", "\n    "))
        self.system.logger.debug("\n    ".join(data))

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
        seen_lhs, add_lhs = set(), []
        for c in lhs:
            if c not in seen_lhs:
                add_lhs.append((c._name_, c))
                seen_lhs.add(c)
        seen_rhs, add_rhs = set(), []
        for c in rhs:
            if c not in seen_rhs:
                add_rhs.append((c._name_, c))
                seen_rhs.add(c)
        self.system.schedule(
            self.compile, 
            UpdateSort(self.lhs.chunks, add=tuple(add_lhs)),
            UpdateSort(self.rhs.chunks, add=tuple(add_rhs)),
            UpdateSort(self.rules, add=tuple((r._name_, r) for r in new)),
            UpdateSite(self.lhs.ciw, data["lhs"]["ciw"], reset=False),
            UpdateSite(self.rhs.ciw, data["rhs"]["ciw"], reset=False), 
            UpdateSite(self.lhs.td.weights, data["lhs"]["tdw"], reset=False), 
            UpdateSite(self.rhs.td.weights, data["rhs"]["tdw"], reset=False),
            UpdateSite(self.riw, data["riw"], reset=False),
            UpdateSite(self.lhw, data["lhw"], reset=False),
            UpdateSite(self.rhw, data["rhw"], reset=False),
            dt=dt, priority=priority)
