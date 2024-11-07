from datetime import timedelta

from ..system import Process, Event, Priority, UpdateSite
from ..knowledge import Family 
from ..numdicts import NumDict, numdict
from .elementary import Choice
from .top_level import RuleStore, ChunkStore


class FixedRules(Process):
    main: NumDict
    store: RuleStore
    choice: Choice
    lhs: ChunkStore
    rhs: ChunkStore

    def __init__(self, 
        name: str, 
        p: Family,
        tl: Family, 
        bli0: Family, 
        bli1: Family, 
        blo0: Family, 
        blo1: Family,
        *,
        sd: float = 1.0,
        lf: float = 0.0
    ) -> None:
        super().__init__(name)
        self.store = RuleStore(f"{name}_st", tl, bli0, bli1, blo0, blo1)
        self.choice = Choice(f"{name}_ch", p, self.store.rules, sd=sd, lf=lf)
        self.lhs = self.store.lhs
        self.rhs = self.store.rhs
        self.main = numdict(self.store.main.i, {}, 0.0)
        self.choice.input = self.store.main

    def resolve(self, event: Event) -> None:
        if event.source == self.trigger:
            self.choice.select()
        if event.source == self.choice.select:
            self.update(dt=self.compute_rt())

    def compute_rt(self) -> timedelta:
        return timedelta(milliseconds=50)

    def trigger(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.DEFERRED
    ) -> None:
        self.system.schedule(self.trigger, dt=dt, priority=priority)

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        choice = (self.store.riw
            .mul(self.choice.main, bs=(1,))
            .sum(by=self.main.i.keyform, b=0))
        input_td = (self.store.rhw
            .mul(self.choice.main)
            .sum(by=self.store.rhs.td.input.i.keyform))
        self.system.schedule(self.update, 
            UpdateSite(self.main, choice.d),
            UpdateSite(self.store.rhs.td.input, input_td.d),
            dt=dt, priority=priority)