from datetime import timedelta
import logging

from .base import Component
from .ops import cam
from ..numdicts import NumDict, KeyForm, keyform, ks_crawl
from ..knowledge import (Family, Chunks, Chunk, Sort, Atom, Term)
from ..system import UpdateSort, Event, Priority, Site
from ..numdicts.ops.base import Unary, Aggregator


class BottomUp(Component):
    """
    A bottom-up activation process.

    Propagates activations from the bottom level to the top level.
    """

    main: Site
    input: Site
    weights: Site
    mul_by: KeyForm
    sum_by: KeyForm
    max_by: KeyForm
    pre: Unary[NumDict] | None
    post: Unary[NumDict] | None

    def __init__(self, 
        name: str, 
        c: Chunks, 
        d: Family | Sort | Term, 
        v: Family | Sort,
        *,
        pre: Unary | None = None,
        post: Unary | None = None
    ) -> None:
        super().__init__(name)
        self.system.check_root(c, d, v)
        idx_c = self.system.get_index(keyform(c))
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        self.main = Site(idx_c, {}, c=0.0)
        self.input = Site(idx_d * idx_v, {}, c=0.0)
        self.weights = Site(idx_c * idx_d * idx_v, {}, c=0.0)
        self.mul_by = keyform(c).agg * keyform(d) * keyform(v)
        self.sum_by = keyform(c) * keyform(d).agg * keyform(v, -1).agg
        self.max_by = keyform(c) * keyform(d) * keyform(v, -1)
        self.pre = pre
        self.post = post

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.system.schedule(self.forward())

    def forward(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> Event:
        input = self.input[0]
        if self.pre is not None:
            input = self.pre(input)
        main = (self.weights[0]
            .mul(input, by=self.mul_by)
            .max(by=self.max_by)
            .sum(by=self.sum_by))
        if self.post is not None:
            main = self.post(main)
        return Event(self.forward, [self.main.update(main)], dt, priority)
    

class TopDown(Component):    
    """
    A top-down activation process.

    Propagates activations from the top level to the bottom level.
    """

    main: Site
    input: Site
    weights: Site
    mul_by: KeyForm
    agg_by: KeyForm
    pre: Unary[NumDict] | None
    post: Unary[NumDict] | None
    agg: Aggregator[NumDict]

    def __init__(self, 
        name: str, 
        c: Chunks, 
        d: Family | Sort | Term, 
        v: Family | Sort,
        *,
        pre: Unary[NumDict] | None = None,
        post: Unary[NumDict] | None = None,
        agg: Aggregator[NumDict] = cam
    ) -> None:
        super().__init__(name)
        self.system.check_root(c, d, v)
        idx_c = self.system.get_index(keyform(c))
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        self.main = Site(idx_d * idx_v, {}, c=0.0)
        self.input = Site(idx_c, {}, c=0.0)
        self.weights = Site(idx_c * idx_d * idx_v, {}, c=0.0) 
        self.mul_by = keyform(c) * keyform(d).agg * keyform(v).agg
        self.agg_by = keyform(c).agg * keyform(d) * keyform(v)
        self.pre = pre
        self.post = post
        self.agg = agg         

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.system.schedule(self.forward())

    def forward(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> Event:
        input = self.input[0]
        if self.pre is not None:
            input = self.pre(input)
        cf = self.weights[0].mul(input, by=self.mul_by)
        if self.post is not None:
            cf = self.post(cf)
        main = self.agg(cf, by=self.agg_by)
        return Event(self.forward, [self.main.update(main)], dt, priority)


class ChunkStore(Component):
    """
    A chunk store. 

    Maintains a collection of chunks and facilitates top-down and bottom-up 
    activation propagation.
    """
    
    chunks: Chunks
    ciw: Site
    tdw: Site
    buw: Site
    sum_by: KeyForm
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
        idx_c = self.system.get_index(keyform(self.chunks))
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        self.ciw = Site(idx_c * idx_c, {}, c=0.0)
        self.tdw = Site(idx_c * idx_d * idx_v, {}, c=0.0)
        self.buw = Site(idx_c * idx_d * idx_v, {}, c=0.0)
        self.sum_by = keyform(self.chunks)
        self.max_by = keyform(self.chunks) * keyform(d) * keyform(v, -1)

    def norm(self, d: NumDict) -> NumDict:
        return (d
            .abs()
            .max(by=self.max_by, c=0.0)
            .sum(by=self.sum_by)
            .shift(1.0))

    def resolve(self, event: Event) -> None:
        new_chunks = [chunk 
            for ud in event.updates 
                if isinstance(ud, UpdateSort) and ud.sort is self.chunks 
                    for chunk in ud.add]
        if new_chunks:
            if event.source == self.encode \
                and self.system.logger.isEnabledFor(logging.DEBUG):
                self.log_encoding(new_chunks)
            self.system.schedule(self.encode_weights(*new_chunks))

    def log_encoding(self, chunks: list[Chunks]) -> None:
        data = [f"    Added the following new chunk(s)"]
        for c in chunks:
            data.append(str(c).replace("\n", "\n    "))
        self.system.logger.debug("\n    ".join(data))
        
    def encode(self, 
        *chunks: Chunk, 
        dt: timedelta = timedelta(), 
        priority=Priority.LEARNING
    ) -> Event:
        """Encode a collection of new chunks."""
        new = [*chunks]
        for chunk in chunks:
            instances = list(chunk._instantiations_())
            chunk._instances_.update(instances)
            new.extend(instances)
        return Event(self.encode, 
            [UpdateSort(self.chunks, add=tuple(new))],
            dt, priority)
        
    def encode_weights(self, *chunks: Chunk, 
        dt: timedelta = timedelta(), 
        priority=Priority.LEARNING
    ) -> Event:
        ciw, tdw = {}, {}
        for chunk in chunks:
            data = chunk._compile_()
            ciw.update(data["ciw"])
            tdw.update(data["tdw"])
        buw = self.buw.new(tdw)
        buw = buw.div(self.norm(buw))
        return Event(self.encode_weights,
            [self.ciw.update(ciw, Site.write_inplace),
             self.tdw.update(tdw, Site.write_inplace),
             self.buw.update(buw, Site.write_inplace)],
            dt, priority)
    
    def bottom_up(self, 
        name: str, 
        *, 
        pre: Unary | None = None, 
        post: Unary | None = None
    ) -> BottomUp:
        _, k_d, k_v = self.buw.index.kf.k.split()
        c = self.chunks
        d = ks_crawl(self.system.root, k_d)
        v = ks_crawl(self.system.root, k_v)
        assert isinstance(d, (Family, Sort, Term))
        assert isinstance(v, (Family, Sort))
        with self:
            obj = BottomUp(name, c, d, v, pre=pre, post=post)
        obj.weights = self.buw
        return obj


    def top_down(self, 
        name: str, 
        *, 
        pre: Unary | None = None, 
        post: Unary | None = None
    ) -> TopDown:
        _, k_d, k_v = self.tdw.index.kf.k.split()
        c = self.chunks
        d = ks_crawl(self.system.root, k_d)
        v = ks_crawl(self.system.root, k_v)
        assert isinstance(d, (Family, Sort, Term))
        assert isinstance(v, (Family, Sort))
        with self:
            obj = TopDown(name, c, d, v, pre=pre, post=post)
        obj.weights = self.tdw
        return obj
