from datetime import timedelta

from ..system import Process, Event, Priority, Site
from ..knowledge import Family, Sort, Atom, Rule, describe, keyform
from ..numdicts import crawl, KeyForm
from .elementary import Choice
from .top_level import RuleStore


class ActionRules(Process):
    """
    An action rule store.
    
    Maintains a collection of action rules and facilitates explicit action  
    selection.
    """

    main: Site
    rules: RuleStore
    choice: Choice
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
            self.choice = Choice(f"{name}.choice", p, self.rules.rules, sd=sd)
        self.main = Site(self.rules.main.index, {}, 0.0)
        self.mul_by = keyform(self.rules.rules).agg * keyform(self.rules.rules)
        self.sum_by = keyform(self.rules.rules) * keyform(self.rules.rules).agg
        self.choice.input = self.rules.main

    def resolve(self, event: Event) -> None:
        if event.source == self.trigger:
            self.choice.select()
        if event.source == self.choice.select:
            self.update(dt=self.compute_rt())
        if event.source == self.update:
            self.log_update()

    def log_update(self):
        rule = crawl(self.system.root, self.choice.main[0].argmax())
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
        choice = self.choice.main[0]
        main = (self.rules.riw[0]
            .mul(choice, by=self.mul_by)
            .sum(by=self.sum_by)
            .with_default(c=self.main.const))
        td_input = (self.rules.rhw[0]
            .mul(choice)
            .sum(by=self.rules.rhs.td.input.index.keyform)
            .with_default(c=self.rules.rhs.td.input.const))
        self.system.schedule(self.update, 
            self.main.update(main),
            self.rules.rhs.td.input.update(td_input),
            dt=dt, priority=priority)
        

class FixedRules(ActionRules):
    """
    A fixed rule store.
    
    Maintains a collection of user-defined fixed action rules, and facilitates 
    explicit action selection.
    """
    pass