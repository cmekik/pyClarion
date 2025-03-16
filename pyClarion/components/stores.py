from datetime import timedelta
import logging

from ..numdicts import NumDict, KeyForm, keyform
from ..knowledge import (Family, Chunks, Rules, Chunk, Rule, Sort, Atom)
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
            ud, = event.updates
            assert isinstance(ud, UpdateSort)
            self.compile_weights(*ud.add)

    def log_compilation(self, event: Event) -> None:
        if event.source != self.compile:
            raise ValueError()
        assert isinstance(event.updates[0], UpdateSort)
        assert event.updates[0].sort is self.chunks
        data = [f"    Added the following new chunk(s)"]
        for c in event.updates[0].add:
            data.append(str(c).replace("\n", "\n    "))
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
        
    def compile(self, 
        *chunks: Chunk, 
        dt: timedelta = timedelta(), 
        priority=Priority.LEARNING
    ) -> None:
        """Encode a collection of new chunks."""
        new = [*chunks]
        for chunk in chunks:
            instances = list(chunk._instantiations_())
            chunk._instances_.update(instances)
            new.extend(instances)
        self.system.schedule(
            self.compile, 
            UpdateSort(self.chunks, add=tuple(new)),
            dt=dt, priority=priority)
        
    def compile_weights(self, *chunks: Chunk, 
        dt: timedelta = timedelta(), 
        priority=Priority.LEARNING
    ) -> None:
        ciw, tdw = {}, {}
        for chunk in chunks:
            data = chunk._compile_()
            ciw.update(data["ciw"])
            tdw.update(data["tdw"])
        buw = self.td.weights.new(tdw)
        buw = buw.div(self.norm(buw, self.bu.max_by, self.bu.sum_by))
        self.system.schedule(
            self.compile_weights,
            self.ciw.update(ciw, Site.write_inplace),
            self.td.weights.update(tdw, Site.write_inplace),
            self.bu.weights.update(buw, Site.write_inplace),
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
            if self.system.logger.isEnabledFor(logging.DEBUG):
                self.log_compilation(event)
            ud_lhs, ud_rhs, ud_rules = event.updates
            assert isinstance(ud_lhs, UpdateSort)
            assert isinstance(ud_rhs, UpdateSort)
            assert isinstance(ud_rules, UpdateSort)
            self.lhs.compile_weights(*ud_lhs.add)
            self.rhs.compile_weights(*ud_rhs.add)
            self.compile_weights(*ud_rules.add)

    def log_compilation(self, event: Event) -> None:
        if event.source != self.compile:
            raise ValueError()
        assert isinstance(event.updates[2], UpdateSort)
        assert event.updates[2].sort is self.rules
        data = [f"    Added the following new rule(s)"]
        for r in event.updates[2].add:
            data.append(str(r).replace("\n", "\n    "))
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
        new_rules = []
        new_lhs_chunks = []
        new_rhs_chunks = []
        for rule in rules:
            for i, chunk in enumerate(rule._chunks_):
                chunk_instances = list(chunk._instantiations_())
                chunk._instances_.update(chunk_instances)
                if i < len(rule._chunks_) - 1:
                    new_lhs_chunks.append(chunk)
                    new_lhs_chunks.extend(chunk_instances)
                else:
                    new_rhs_chunks.append(chunk)
                    new_rhs_chunks.extend(chunk_instances)
            rule_instances = list(rule._instantiations_())
            rule._instances_.update(rule_instances)
            new_rules.append(rule)
            new_rules.extend(rule_instances)
        self.system.schedule(
            self.compile, 
            UpdateSort(self.lhs.chunks, add=tuple(new_lhs_chunks)),
            UpdateSort(self.rhs.chunks, add=tuple(new_rhs_chunks)),
            UpdateSort(self.rules, add=tuple(new_rules)),
            dt=dt, priority=priority)

    def compile_weights(self, 
        *rules: Rule, 
        dt: timedelta = timedelta(), 
        priority=Priority.LEARNING
    ) -> None:
        riw, lhw, rhw = {}, {}, {}
        for rule in rules:
            data = rule._compile_()
            riw.update(data["riw"])
            lhw.update(data["lhw"])
            rhw.update(data["rhw"])
        self.system.schedule(
            self.compile_weights,
            self.riw.update(riw, Site.write_inplace),
            self.lhw.update(lhw, Site.write_inplace),
            self.rhw.update(rhw, Site.write_inplace),
            dt=dt, priority=priority)
