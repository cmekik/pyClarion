from typing import Callable
from datetime import timedelta

from ..components.base import Stateful, Priority, ChunkUpdate
from ..components.chunks import ChunkStore
from ..components.io import Discriminal, Choice, Controller
from ..components.layers import Accumulator, Router
from ..components.stats import BaseLevel
from ..events import Event, Site, State, ForwardUpdate
from ..knowledge import (Family, DataFamily, ChunkFamily, Chunks, 
    Chunk, Atoms, Atom, Bus)
from ..numdicts import ks_crawl


class BufferOps(Atoms):
    nil: Atom
    clear: Atom
    add: Atom
    store: Atom
    flip: Atom
    fetch: Atom


class Buffer(Stateful):
    c: Chunks
    s: BufferOps
    main: Site
    status: Site
    reader: Discriminal
    requester: Accumulator
    retriever: Choice

    def __init__(self, 
        name: str,         
        p: Family,
        m: Bus,
        b: Bus,
        v: DataFamily,
        c: Chunks,
        s: BufferOps
    ) -> None:
        triggers = (self.start_clear, self.start_add, self.start_store, 
            self.start_flip, self.start_fetch)
        super().__init__(name, *triggers)
        self.system.check_root(p, m, b, v, c, s)
        idx_c, idx_s = self._init_indexes(c, s)
        self.c = c
        self.s = s
        self.main = State(idx_c, {}, 0.0)
        self.status = State(idx_s, {~s.nil: 1.0}, 0.0)
        with self:
            self.reader = Discriminal(f"{name}.reader", p, (m, v))
            self.router = Router(f"{name}.router", m, b, v)
            self.requester = Accumulator(f"{name}.requester", (b, v))
            self.retriever = Choice(f"{name}.retriever", p, v, c)
        self.requester = self.reader >> self.router >> self.requester

    def resolve(self, event: Event) -> None:

        status = self.current_status
        source = event.source
        schedule = self.system.schedule

        if status == ~self.s.nil:
            pass

        elif status == ~self.s.clear:
            if source == self.start_clear:
                schedule(self.requester.clear())
            elif source == self.requester.clear:
                schedule(self.finish_clear())

        elif status == ~self.s.store:
            if source == self.start_store:
                schedule(self._store())
            elif source == self._store:
                schedule(self.requester.clear())
            elif source == self.requester.clear:
                schedule(self.finish_store())

        elif status == ~self.s.add:
            if source == self.start_add:
                schedule(self.reader.select())
            elif source == self.requester.forward:
                schedule(self.finish_add())

        elif status == ~self.s.flip:
            if source == self.start_flip:
                schedule(self._flip())
            elif source == self._flip:
                schedule(self.requester.clear())
            elif source == self.requester.clear:
                schedule(self.finish_flip())

        elif status == ~self.s.fetch:        
            if source == self.start_fetch:
                schedule(self.retriever.trigger())
            elif source == self.retriever.select:
                schedule(self._fetch())
            elif source == self._fetch:
                schedule(self.requester.clear())
            elif source == self.requester.clear:
                schedule(self.finish_fetch())

    def start_clear(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.DEFERRED
    ) -> Event:
        return self._trigger(self.start_clear, self.s.clear, dt, priority)
    
    def finish_clear(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        return self._terminate(self.finish_clear, dt, priority)

    def start_add(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.DEFERRED
    ) -> Event:
        return self._trigger(self.start_add, self.s.add, dt, priority)
    
    def finish_add(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        return self._terminate(self.finish_add, dt, priority)
    
    def start_store(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.DEFERRED
    ) -> Event:
        return self._trigger(self.start_store, self.s.store, dt, priority)

    def _store(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        chunk = self._extract_chunk()
        ud = ChunkUpdate(self.c, add=(chunk,))
        return Event(self._store, [ud], dt, priority)

    def finish_store(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        return self._terminate(self.finish_store, dt, priority)

    def start_flip(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.DEFERRED
    ) -> Event:
        return self._trigger(self.start_flip, self.s.flip, dt, priority)

    def _flip(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        c = self._extract_chunk()
        main = self.main.new({~c: 1.0})
        uds = [ChunkUpdate(self.c, add=(c,)), ForwardUpdate(self.main, main)]
        return Event(self._flip, uds, dt, priority)

    def finish_flip(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        return self._terminate(self.finish_flip, dt, priority)

    def start_fetch(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.DEFERRED
    ) -> Event:
        return self._trigger(self.start_fetch, self.s.fetch, dt, priority)

    def _fetch(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        c = self.retriever.poll()[~self.c]
        main = self.main.new({c: 1.0})
        ud = ForwardUpdate(self.main, main)
        return Event(self._fetch, [ud], dt, priority)
    
    def finish_fetch(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        return self._terminate(self.finish_fetch, dt, priority)

    def _extract_chunk(self) -> Chunk:
        root = self.system.root
        dyads = {}
        for k, w in self.requester.main[0].d.items():
            # key must be a dvpair.
            k_d, k_v = k.split()
            d, v = ks_crawl(root, k_d), ks_crawl(root, k_v)
            dyads[(d, v)] = w
        return Chunk(dyads)


class Stack(Stateful):
    s: BufferOps
    status: Site = Site()
    chunks: ChunkStore
    bla: BaseLevel
    buffer: Buffer
    controller: Controller
    _triggers: set[Callable[..., Event]]

    def __init__(
        self, 
        name: str,
        p: Family,
        c: ChunkFamily | DataFamily,
        a: Bus,
        m: Bus,
        b: Bus,
        v: DataFamily,
        s: BufferOps
    ) -> None:
        triggers = (self.start_clear, self.start_add, self.start_store, 
            self.start_flip, self.start_fetch)
        super().__init__(name, *triggers)
        self.system.check_root(s)
        idx_s, = self._init_indexes(s)
        self.s = s
        self.status = State(idx_s, {~self.s.nil: 1.0}, 0.0)
        with self:            
            self.chunks = ChunkStore(f"{name}.chunks", c, (b, v))
            self.bla = BaseLevel(f"{name}.bla", p, v, self.chunks.c)
            self.buffer = Buffer(f"{name}.buffer", p, m, b, v, self.chunks.c, s)
            self.controller = Controller(f"{name}.controller", a, s,
                clear=self.start_clear, 
                add=self.start_add, 
                store=self.start_store,
                flip=self.start_flip,
                fetch=self.start_fetch)
        self.buffer.retriever = self.bla >> self.buffer.retriever

        
    def resolve(self, event: Event) -> None:
        status = self.current_status
        source = event.source
        schedule = self.system.schedule

        if status == ~self.s.nil:
            pass

        elif status == ~self.s.clear:
            if source == self.start_clear:
                schedule(self.buffer.start_clear())
            elif source == self.buffer.finish_clear:
                schedule(self.finish_clear())

        elif status == ~self.s.add:
            if source == self.start_add:
                schedule(self.buffer.start_add())
            elif source == self.buffer.finish_add:
                schedule(self.finish_add())

        elif status == ~self.s.store:
            if source == self.start_store:
                schedule(self.buffer.start_add())
            elif source == self.buffer.finish_add:
                schedule(self.buffer.start_store())
            elif source == self.buffer.finish_store:
                schedule(self.finish_store())

        elif status == ~self.s.flip:
            if source == self.start_flip:
                schedule(self.buffer.start_add())
            elif source == self.buffer.finish_add:
                schedule(self.buffer.start_flip())
            elif source == self.buffer.finish_flip:
                schedule(self.finish_flip())

        elif status == ~self.s.fetch:
            if source == self.start_fetch:
                schedule(self.buffer.start_add())
            elif source == self.buffer.finish_add:
                schedule(self.bla.advance())
            elif source == self.bla.advance:
                schedule(self.buffer.start_fetch())
            elif source == self.buffer.finish_fetch:
                schedule(self.finish_fetch())

        else:
            raise RuntimeError(f"{type(self).__name__} object '{self.name}' in "
                "invalid state.")
    
    def start_clear(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.DEFERRED
    ) -> Event:
        return self._trigger(self.start_clear, self.s.clear, dt, priority)
                             
    def finish_clear(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        return self._terminate(self.finish_clear, dt, priority)
                
    def start_add(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.DEFERRED
    ) -> Event:
        return self._trigger(self.start_add, self.s.add, dt, priority)
                             
    def finish_add(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        return self._terminate(self.finish_add, dt, priority)

    def start_store(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.DEFERRED
    ) -> Event:
        return self._trigger(self.start_store, self.s.store, dt, priority)
                             
    def finish_store(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        return self._terminate(self.finish_store, dt, priority)

    def start_flip(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.DEFERRED
    ) -> Event:
        return self._trigger(self.start_flip, self.s.flip, dt, priority)
                             
    def finish_flip(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        return self._terminate(self.finish_flip, dt, priority)

    def start_fetch(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.DEFERRED
    ) -> Event:
        return self._trigger(self.start_fetch, self.s.fetch, dt, priority)

    def finish_fetch(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        return self._terminate(self.finish_fetch, dt, priority)
