from datetime import timedelta

from ...system import Priority, Site
from ...knowledge import Family, Atoms, Atom
from ...numdicts import path
from .base import Optimizer, Backprop


class SGD(Optimizer):
    class Params(Atoms):
        lr: Atom
        sd: Atom
        l2: Atom
    
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
        lr = self.params[0][path(self.p.lr)]
        sd = self.params[0][path(self.p.sd)]
        l2 = self.params[0][path(self.p.l2)]
        uds = []
        for layer in self.layers:
            if Backprop.Train.BIAS in layer.train:
                bias = (layer.grad_bias[0]
                    .sub(layer.bias[-1].scale(x=l2))
                    .normalvariate(layer.grad_bias[0].abs().scale(x=sd))
                    .scale(x=lr)
                    .with_default(c=layer.bias.const))
                uds.append(layer.bias.update(bias, Site.add_inplace))
                uds.append(layer.grad_bias.update({}))
            if Backprop.Train.WEIGHTS in layer.train:
                weights = (layer.grad_weights[0]
                    .sub(layer.weights[-1].scale(x=l2))
                    .normalvariate(layer.grad_weights[0].abs().scale(x=sd))
                    .scale(x=lr)
                    .with_default(c=layer.weights.const))
                uds.append(layer.weights.update(weights, Site.add_inplace))
                uds.append(layer.grad_weights.update({}))
        self.system.schedule(self.update, *uds, dt=dt, priority=priority)
