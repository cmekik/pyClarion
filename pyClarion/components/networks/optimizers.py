from datetime import timedelta

from .base import Optimizer, Train, Layer
from ...system import Priority, Site
from ...knowledge import Family, Atoms, Atom


class SGD(Optimizer):
    """
    A stochastic gradient descent process.
    
    Issues updates to weights and biases of a collection of layers using 
    stochastic gradient descent with l2 regularization and gradient noise.
    """

    class Params(Atoms):
        lr: Atom # Learning rate
        sd: Atom # Upper bound on standard deviation of gradient noise
        l2: Atom # L2 regularization parameter
    
    def __init__(self, 
        name: str, 
        p: Family, 
        *, 
        lr: float = 1e-2, 
        sd: float = 1.0,
        l2: float = 1e-3
    ) -> None:
        super().__init__(name, p, lr=lr, sd=sd, l2=l2)

    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> None:
        if not self.layers:
            return
        lr = self.params[0][~self.p.lr]
        sd_ = self.params[0][~self.p.sd]
        l2 = self.params[0][~self.p.l2]
        uds = []
        for layer in self.layers:
            if layer.afunc:
                sd = sd_ * layer.afunc.scale(layer) 
            else: 
                sd = sd_ / len(layer.input[0])
            if Train.BIAS in layer.train:
                uds.extend(self._update(layer.bias, lr, sd, l2))
            if Train.WEIGHTS in layer.train:
                uds.extend(self._update(layer.weights, lr, sd, l2))
        self.system.schedule(self.update, *uds, dt=dt, priority=priority)

    def _update(self, 
        param: Site, 
        lr: float, 
        sd: float, 
        l2: float
    ) -> tuple[Site.Update, Site.Update]:
        delta = (param.grad[-1]
            .normalvariate(param.grad[-1].abs().scale(x=sd), c=param.const)
            .sum(param[-1].scale(x=l2))
            .scale(x=-lr)) # Don't miss negative here!
        return (param.update(delta, Site.add_inplace), 
            param.update({}, grad=True))


class Adam(Optimizer):
    """
    An adaptive moment estimation (Adam) process.
    
    Issues updates to weights and biases of a collection of layers using 
    adaptive moment estimation with l2 regularization and gradient noise.
    """

    class Params(Atoms):
        lr: Atom # Learning rate
        b1: Atom # Exponential decay rate for moment 1
        b2: Atom # Exponential decay rate for moment 2
        sd: Atom # Upper bound on standard deviation of gradient noise
        l2: Atom # L2 regularization parameter
        ep: Atom # Epsilon
        bt1: Atom 
        bt2: Atom
    
    wm1: dict[str, Site]
    wm2: dict[str, Site]
    bm1: dict[str, Site]
    bm2: dict[str, Site]

    def __init__(self, 
        name: str, 
        p: Family, 
        *, 
        lr: float = 1e-2,
        b1: float = 9e-1,
        b2: float = .999, 
        sd: float = 1.0,
        l2: float = 1e-3,
        ep: float = 1e-8
    ) -> None:
        super().__init__(name, p, 
            lr=lr, b1=b1, b2=b2, sd=sd, l2=l2, ep=ep, bt1=b1, bt2=b2)
        self.wm1 = {}
        self.wm2 = {}
        self.bm1 = {}
        self.bm2 = {}

    def add(self, layer: Layer) -> None:
        super().add(layer)
        self.wm1[layer.name] = Site(layer.weights.index, {}, 0.0)
        self.wm2[layer.name] = Site(layer.weights.index, {}, 0.0)
        self.bm1[layer.name] = Site(layer.bias.index, {}, 0.0)
        self.bm2[layer.name] = Site(layer.bias.index, {}, 0.0)

    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> None:
        if not self.layers:
            return
        lr = self.params[0][~self.p.lr]
        sd_ = self.params[0][~self.p.sd]
        l2 = self.params[0][~self.p.l2]
        b1 = self.params[0][~self.p.b1]
        b2 = self.params[0][~self.p.b2]
        bt1 = self.params[0][~self.p.bt1]
        bt2 = self.params[0][~self.p.bt2]
        ep = self.params[0][~self.p.ep]
        uds = []
        for layer in self.layers:
            if layer.afunc:
                sd = sd_ * layer.afunc.scale(layer) 
            else: 
                sd = sd_ / len(layer.input[0])
            if Train.BIAS in layer.train:
                m, v = self.bm1[layer.name], self.bm2[layer.name]
                uds.extend(self._update(
                    layer.bias, m, v, lr, sd, l2, b1, b2, bt1, bt2, ep))
            if Train.WEIGHTS in layer.train:
                m, v = self.wm1[layer.name], self.wm2[layer.name]
                uds.extend(self._update(
                    layer.weights, m, v, lr, sd, l2, b1, b2, bt1, bt2, ep))
        bt1 = bt1 * b1
        bt2 = bt2 * b2
        uds.append(self.params.update({~self.p.bt1: bt1, ~self.p.bt2: bt2},
            Site.write_inplace))
        self.system.schedule(self.update, *uds, dt=dt, priority=priority)

    def _update(self, 
        param: Site, 
        m: Site,
        v: Site, 
        lr: float, 
        sd: float, 
        l2: float,
        b1: float,
        b2: float,
        bt1: float,
        bt2: float,
        ep: float
    ) -> tuple[Site.Update, Site.Update, Site.Update, Site.Update]:
        m_next = m[0].scale(x=b1).sum(param.grad[-1].scale(x=1 - b1))
        v_next = v[0].scale(x=b2).sum(param.grad[-1].pow(x=2).scale(x=1 - b2))
        m_hat = m_next.scale(x=1/(1 - bt1))
        v_hat = v_next.scale(x=1/(1 - bt2))
        g_hat = m_hat.div(v_hat.pow(x=0.5).shift(x=ep))
        delta = (g_hat
            .normalvariate(g_hat.abs().scale(x=sd), c=param.const)
            .sum(param[-1].scale(x=l2))
            .scale(x=-lr)) # Don't miss negative here!
        return (
            param.update(delta, Site.add_inplace), 
            param.update({}, grad=True), 
            m.update(m_next), 
            v.update(v_next))
