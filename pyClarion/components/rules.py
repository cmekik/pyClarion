from datetime import timedelta

from ..system import Process, Event, Priority, UpdateSite
from ..knowledge import Family, Sort, Atom, Rule, describe, keyform
from ..numdicts import NumDict, numdict, crawl, KeyForm
from .elementary import ChoiceTL
from .top_level import RuleStore


class FixedRules(Process):
    main: NumDict
    rules: RuleStore
    choice: ChoiceTL
    by: KeyForm

    def __init__(self, 
        name: str, 
        p: Family,
        r: Family,
        c: Family, 
        d: Family | Sort | Atom, 
        v: Family | Sort,
        *,
        sd: float = 1.0
    ) -> None:
        super().__init__(name)
        with self:
            self.rules = RuleStore(f"{name}.rules", r, c, d, v)
            self.choice = ChoiceTL(f"{name}.choice", p, self.rules.rules, sd=sd)
        self.main = numdict(self.rules.main.i, {}, 0.0)
        self.by = keyform(d) * keyform(v, trunc=1)
        self.choice.input = self.rules.main

    def resolve(self, event: Event) -> None:
        if event.source == self.trigger:
            self.choice.select()
        if event.source == self.choice.select:
            self.update(dt=self.compute_rt())
        if event.source == self.update:
            self.log_update()

    def log_update(self):
        rule = crawl(self.system.root, self.choice.main.argmax())
        assert isinstance(rule, Rule)
        message = "\n    ".join([
            "    Fired the following rule", 
            describe(rule).replace("\n", "\n    ")])
        self.system.logger.info(message)

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
        choice = (self.rules.riw
            .mul(self.choice.main, bs=(1,))
            .sum(by=self.main.i.keyform, b=0))
        input_td = (self.rules.rhw
            .mul(self.choice.main)
            .sum(by=self.rules.rhs.td.input.i.keyform))
        self.system.schedule(self.update, 
            UpdateSite(self.main, choice.d),
            UpdateSite(self.rules.rhs.td.input, input_td.d),
            dt=dt, priority=priority)