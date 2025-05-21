from typing import Callable
from datetime import timedelta

from .base import Component, Parametric, D, V, DV
from .io import Choice
from .ops import least_squares_cost, Cost
from ..system import Site, Priority, Event, PROCESS
from ..knowledge import Family, Atoms, Atom, Term
from ..numdicts import NumDict, Key


class LearningSignal(Component):
    """
    A neural network error signaling process.
    
    Computes and backpropagates error signals based on neural network outputs.
    """

    main: Site

    # TODO: FIX
    # def __rrshift__(self: Self, other: Layer) -> Self:
    #     if isinstance(other, Layer):
    #         self.input = other.main
    #         return self
    #     return NotImplemented
    
    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> Event:
        """Compute and schedule update to error value."""
        raise NotImplementedError()


class SupervisedLearning(LearningSignal):
    """
    An error signaling process for supervised learning.
    
    Computes and backpropagates errors based on a supervised cost function.
    """

    cost: Cost
    input: Site
    target: Site
    mask: Site

    def __init__(self, 
        name: str, 
        s: V | DV, 
        cost: Cost = least_squares_cost
    ) -> None:
        super().__init__(name)
        index, = self._init_indexes(s)
        self.cost = cost
        self.main = Site(index, {}, c=0.0)
        self.input = Site(index, {}, c=0.0)
        self.target = Site(index, {}, c=0.0)
        self.mask = Site(index, {}, c=0.0)
    
    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> Event:
        est = self.input[0]
        tgt = self.target[0]
        mask = self.mask[0]
        cost = self.cost(est, tgt, mask)
        main = self.cost.grad(cost.ones(), cost, est, tgt, mask)[0]
        return Event(self.update, (self.main.update(main),), dt, priority)
    

class TDLearning(Parametric, LearningSignal):
    """
    An error signaling process for temporal difference learning.
    
    Computes and backpropagates errors based on a temporal difference function.
    """

    class Params(Atoms):
        gamma: Atom

    p: Params
    choice: Choice
    func: Callable[["TDLearning"], NumDict]
    main: Site
    input: Site
    reward: Site
    qvals: Site
    action: Site
    params: Site

    def next_Q(self) -> NumDict:
        qvals = self.input[0]
        return self.main.new({}).sum(qvals.mul(self.action[0]).sum())

    def expected_Q(self) -> NumDict:
        sd = self.choice.params[0][~self.choice.p.sd]
        qvals = self.input[0]
        pvec = qvals.scale(sd).exp()
        pvec = pvec.div(pvec.sum())
        return pvec.mul(qvals).sum()

    def max_Q(self) -> NumDict:
        qvals = self.input[0]
        return (self.main.new({})
            .sum(qvals
                .max(by=self.choice.by)))
        
    def __init__(self, 
        name: str, 
        p: Family, 
        r: DV | D, 
        *,
        func: Callable[["TDLearning"], NumDict] = max_Q,
        gamma: float = .5,
        l: int = 1
    ) -> None:
        if l < 1:
            raise ValueError(f"Expected 1 <= l but got l == {l}.")
        super().__init__(name)
        self.func = func
        self._connect_to_choice()
        self.p, self.params = self._init_sort(
            p, type(self).Params, gamma=gamma)
        idx_r, = self._init_indexes(r)
        idx_a = self.choice.main.index
        self.main = Site(idx_a, {}, c=0.0)
        self.input = Site(idx_a, {}, c=0.0, l=l + 1)
        self.qvals = Site(idx_a, {}, c=0.0, l=l)
        self.reward = Site(idx_r, {}, c=0.0, l=l)
        self.action = Site(idx_a, {}, c=0.0, l=l)

    def _connect_to_choice(self):
        try:
            sup = PROCESS.get()
        except LookupError:
            raise RuntimeError("TDError must be initalized in Choice context")
        if not isinstance(sup, Choice):
            raise RuntimeError("TDError must be initalized in Choice context")
        self.choice = sup

    def resolve(self, event: Event) -> None:
        if event.source == self.choice.select:
            self.system.schedule(self.update())

    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        gamma = self.params[0][~self.p.gamma]
        n = len(self.reward)
        main = (self.func(self)
            .scale(gamma ** n)
            .sum(*(rwd.sum().scale(gamma ** (n - 1 - t)) 
                for t, rwd in enumerate(self.reward.data)))
            .sub(self.qvals[-1])
            .mul(self.action[-1])
            .neg())
        return Event(self.update,
            (self.main.update(main.pow(2).scale(.5)),
             self.input.update(main, grad=True),
             self.reward.update({}),
             self.qvals.update(self.input[0]),
             self.action.update(self.choice.main[0])),
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
            (self.reward.update(data, Site.add_inplace),), 
            dt, priority)
