from datetime import timedelta
from typing import Sequence

from .base import Parametric
from ..system import Priority, Site, Event
from ..knowledge import Family, Atoms, Atom


class Optimizer[P: Atoms](Parametric):
    """
    A neural network optimization process. 

    Issues updates to weights and biases of a collection of layers. 
    """

    Params: type[P]
    p: P
    params: Site
    sites: set[Site]

    def __init__(self, name: str, p: Family, **params: float) -> None:
        super().__init__(name)
        self.p, self.params = self._init_sort(
            p, type(self).Params, 0.0, 1, **params)
        self.sites = set()

    def add(self, *sites: Site) -> None:
        """Include sites in future updates."""
        self.sites.update(sites)

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> Event:
        """Compute and schedule parameter updates for all client layers."""
        return Event(self.update, 
            [ud for s in self.sites for ud in self.site_updates(s)], 
            dt, priority)
    
    def site_updates(self, site: Site) -> Sequence[Site.Update]:
        raise NotImplementedError()


class SGD(Optimizer):
    """
    A stochastic gradient descent process.
    
    Issues updates to weights and biases of a collection of layers using 
    stochastic gradient descent.
    """

    class Params(Atoms):
        lr: Atom

    def __init__(self, name: str, p: Family, *, lr: float = 1e-2) -> None:
        super().__init__(name, p, lr=lr)

    def site_updates(self, site: Site) -> tuple[Site.Update, Site.Update]:
        lr = self.params[0][~self.p.lr]
        delta = site.grad[-1].scale(-lr)
        return site.update(delta, Site.add_inplace), site.update({}, grad=True)


class Adam(Optimizer):
    """
    An adaptive moment estimation (Adam) process.
    
    Issues updates to weights and biases of a collection of layers using 
    adaptive moment estimation.
    """

    class Params(Atoms):
        lr: Atom # Learning rate
        b1: Atom # Exponential decay rate for moment 1
        b2: Atom # Exponential decay rate for moment 2
        ep: Atom # Epsilon
        bt1: Atom 
        bt2: Atom
    
    # Maybe these should be weak key dicts?
    m1: dict[Site, Site]
    m2: dict[Site, Site]

    def __init__(self, 
        name: str, 
        p: Family, 
        *, 
        lr: float = 1e-2,
        b1: float = 9e-1,
        b2: float = .999, 
        ep: float = 1e-8
    ) -> None:
        super().__init__(name, p, lr=lr, b1=b1, b2=b2, ep=ep, bt1=b1, bt2=b2)
        self.m1 = {}
        self.m2 = {}

    def add(self, *sites: Site) -> None:
        super().add(*sites)
        for s in sites:
            self.m1[s] = Site(s.index, {}, 0.0)
            self.m2[s] = Site(s.index, {}, 0.0)

    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> Event:
        event = super().update(dt, priority)
        b1 = self.params[0][~self.p.b1]
        b2 = self.params[0][~self.p.b2]
        bt1 = self.params[0][~self.p.bt1]
        bt2 = self.params[0][~self.p.bt2]
        bt1 = bt1 * b1
        bt2 = bt2 * b2
        assert isinstance(event.updates, list)
        event.updates.append(
            self.params.update(
                {~self.p.bt1: bt1, ~self.p.bt2: bt2}, 
                Site.write_inplace))
        return event

    def site_updates(self, site: Site) \
        -> tuple[Site.Update, Site.Update, Site.Update, Site.Update]:
        lr = self.params[0][~self.p.lr]
        b1 = self.params[0][~self.p.b1]
        b2 = self.params[0][~self.p.b2]
        bt1 = self.params[0][~self.p.bt1]
        bt2 = self.params[0][~self.p.bt2]
        ep = self.params[0][~self.p.ep]
        m, v = self.m1[site], self.m2[site]
        m_next = m[0].scale(b1).sum(site.grad[-1].scale(1 - b1))
        v_next = v[0].scale(b2).sum(site.grad[-1].pow(2).scale(1 - b2))
        m_hat = m_next.scale(1/(1 - bt1))
        v_hat = v_next.scale(1/(1 - bt2))
        g_hat = m_hat.div(v_hat.pow(0.5).shift(ep))
        delta = (g_hat
            .scale(-lr))
        return (
            site.update(delta, Site.add_inplace), 
            site.update({}, grad=True), 
            m.update(m_next), 
            v.update(v_next))
