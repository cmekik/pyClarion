from datetime import timedelta
import logging

from ..numdicts import NumDict, KeyForm
from ..knowledge import (Family, Chunks, Rules, Chunk, Rule, 
    compile_chunks, compile_rules, keyform, Sort, Atom, describe)
from ..system import Process, UpdateSort, Event, Priority, Site
from .elementary import TopDown, BottomUp


class ChunkStore(Process):
    """
    A chunk store. 

    Maintains a collection of chunks and facilitates top-down and bottom-up 
    activation propagation.
    """
    
    chunks: Chunks
    main: Site
    ciw: Site
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
        self.main = Site(index, {}, c=0.0)
        self.ciw = Site(index * index, {}, c=float("nan"))
        self.mul_by = keyform(self.chunks).agg * keyform(self.chunks)
        self.max_by = keyform(self.chunks) * keyform(self.chunks).agg 
        with self:
            self.td = TopDown(f"{name}.td", self.chunks,  d, v)
            self.bu = BottomUp(f"{name}.bu", self.chunks, d, v)

    def norm(self, d: NumDict, max_by: KeyForm, sum_by: KeyForm) -> NumDict:
        return (d
            .abs()
            .max(by=max_by)
            .sum(by=sum_by)
            .shift(x=1.0))

    def resolve(self, event: Event) -> None:
        if event.source == self.bu.update:
            self.update()
        if event.source == self.compile:
            if self.system.logger.isEnabledFor(logging.DEBUG):
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
        """Compute and update abstract chunk activations."""
        main = (self.ciw[0]
            .mul(self.bu.main[0], by=self.mul_by)
            .max(by=self.max_by)
            .with_default(c=self.main.const))
        self.system.schedule(self.update, 
            self.main.update(main),
            dt=dt, priority=priority)
        
    def compile(self, *chunks: Chunk, 
        dt: timedelta = timedelta(), 
        priority=Priority.LEARNING
    ) -> None:
        """Encode a collection of new chunks."""
        new, data = compile_chunks(*chunks, sort=self.chunks)
        self.system.schedule(
            self.compile, 
            UpdateSort(self.chunks, add=tuple((c._name_, c) for c in new)),
            self.ciw.update(data["ciw"], Site.write_inplace),
            self.td.weights.update(data["tdw"], Site.write_inplace),
            dt=dt, priority=priority)

    def update_buw(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.LEARNING
    ) -> None:
        """Update bottom-up weights to be consistent with top-down weights."""
        weights = (self.td.weights[0]
            .div(self.norm(self.td.weights[0], self.bu.max_by, self.bu.sum_by)))
        self.system.schedule(
            self.update_buw, 
            self.bu.weights.update(weights),
            dt=dt, priority=priority)


class RuleStore(Process):
    """
    A rule store. 

    Maintains a collection of rules.
    """

    rules: Rules
    lhs: ChunkStore
    rhs: ChunkStore
    main: Site
    riw: Site
    lhw: Site
    rhw: Site

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
        self.main = Site(idx_r, {}, c=0.0)
        self.riw = Site(idx_r * idx_r, {}, c=float("nan"))
        self.lhw = Site(idx_r * idx_lhs, {}, c=float("nan"))
        self.rhw = Site(idx_r * idx_rhs, {}, c=float("nan"))

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
        main = (self.lhw[0]
            .mul(self.lhs.bu.main[0])
            .max(by=self.main.index.kf)
            .with_default(c=self.main.const))
        self.system.schedule(self.update, 
            self.main.update(main), 
            dt=dt, priority=priority)

    def compile(self, *rules: Rule, 
        dt: timedelta = timedelta(),
        priority: int = Priority.LEARNING
    ) -> None:
        """Encode a collection of new rules."""
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
            self.lhs.ciw.update(data["lhs"]["ciw"], Site.write_inplace),
            self.rhs.ciw.update(data["rhs"]["ciw"], Site.write_inplace), 
            self.lhs.td.weights.update(data["lhs"]["tdw"], Site.write_inplace), 
            self.rhs.td.weights.update(data["rhs"]["tdw"], Site.write_inplace),
            self.riw.update(data["riw"], Site.write_inplace),
            self.lhw.update(data["lhw"], Site.write_inplace),
            self.rhw.update(data["rhw"], Site.write_inplace),
            dt=dt, priority=priority)
