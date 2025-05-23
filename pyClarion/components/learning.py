from typing import Callable
from datetime import timedelta

from .base import Component, D, V, DV
from .io import Choice
from ..system import Event, State, Site, Priority
from ..knowledge import Family, Atom, Term
from ..numdicts import NumDict, Key


class LearningSignal(Component):
    """
    A neural network error signaling process.
    
    Computes and backpropagates error signals based on neural network outputs.
    """

    cost: Site = Site()
    
    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> Event:
        """Compute and schedule update to error value."""
        raise NotImplementedError()


class TDLearning(LearningSignal, Choice):
    
    func: Callable[["TDLearning"], NumDict]
    actions: Site = Site()
    qvals: Site = Site()
    reward: Site = Site()

    def next_Q(self) -> NumDict:
        qvals = self.qvals[0]
        return self.main.new({}).sum(qvals.mul(self.main[0]).sum())

    def expected_Q(self) -> NumDict:
        sd = self.params[0][~self.p.sd]
        qvals = self.qvals[0]
        pvec = qvals.scale(sd).exp()
        pvec = pvec.div(pvec.sum())
        return pvec.mul(qvals).sum()

    def max_Q(self) -> NumDict:
        qvals = self.qvals[0]
        return self.main.new({}).sum(qvals.max(by=self.by, c=0.0))
    
    def __init__(self, 
        name: str, 
        p: Family,
        s_: Family, 
        s: V | DV, 
        r: DV | D,
        *, 
        func: Callable[["TDLearning"], NumDict] = max_Q,
        sd: float = 1.0,
        f: float = 1.0,
        gamma: float = .7,
        l: int = 2
    ) -> None:
        if l < 2:
            raise ValueError("Arg l must be greater than or equal to 2")
        super().__init__(name, p, s_, s, sd=sd, f=f)
        self.p["gamma"] = Atom()
        with self.params[0].mutable() as d:
            d[~self.p["gamma"]] = gamma
        idx_a = self.main.index
        idx_r, = self._init_indexes(r)
        self.cost = State(idx_a, {}, c=0.0)
        self.qvals = State(idx_a, {}, c=0.0, l=l)
        self.actions = State(idx_a, {}, c=0.0, l=l)
        self.reward = State(idx_r, {}, c=0.0, l=l)
        self.func = func

    def resolve(self, event: Event) -> None:
        super().resolve(event)
        if event.source == self.select:
            self.system.schedule(self.update())

    def select(self, 
        dt: timedelta = timedelta(), 
        priority=Priority.CHOICE
    ) -> Event:
        event = super().select(dt, priority)
        qvals = self.qvals.new({}).sum(self.input[0])
        main, *_ = event.updates
        assert isinstance(main, State.Update)
        event.updates.append(self.qvals.update(qvals)) 
        event.updates.append(self.actions.update(main.data))
        return event

    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> Event:
        gamma = self.params[0][~self.p["gamma"]]
        n = len(self.reward)
        error = (self.func(self)
            .scale(gamma ** (n - 1))
            .sum(*(rwd.sum().scale(gamma ** (n - 2 - t)) 
                for t, rwd in enumerate(self.reward.data) if t <= n - 2))
            .sub(self.qvals[-1])
            .mul(self.actions[-1])
            .neg())
        return Event(self.update,
            [self.cost.update(error.pow(2).scale(.5)),
             self.input.update(error, grad=True),
             self.reward.update({})],
            dt, priority)

    def send(self, d: dict[Term | Key, float], 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> Event:
        data = {}
        for k, v in d.items():
            if isinstance(k, Term):
                self.system.check_root(k)
                k = ~k
            if k not in self.reward.index:
                raise ValueError(f"Unexpected key {k}")
            data[k] = v
        return Event(self.send, 
            [self.reward.update(data, State.add_inplace)], 
            dt, priority)
