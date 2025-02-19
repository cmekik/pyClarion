from datetime import timedelta

from .base import Optimizer, Train
from ...system import Priority, Site
from ...knowledge import Family, Atoms, Atom
from ...numdicts import path


class SGD(Optimizer):
    """
    A stochastic gradient descent process.
    
    Issues updates to weights and biases of a collection of layers using 
    stochastic gradient descent with l2 regularization and gradient noise.
    """

    class Params(Atoms):
        lr: Atom
        sd: Atom
        l2: Atom
    
    def __init__(self, 
        name: str, 
        p: Family, 
        *, 
        lr: float = 1e-2, 
        sd: float = 1e-2,
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
            if layer.afunc:
                scale = sd * layer.afunc.scale(layer) 
            else: 
                scale = sd / (1 + len(layer.input))
            if Train.BIAS in layer.train:
                bias_sd = (layer.grad_bias[-1]
                    .pvariance().neg().exp()
                    .scale(x=scale))
                d_bias = (layer.grad_bias[-1]
                    .normalvariate(bias_sd, c=layer.bias.const)
                    .sum(layer.bias[-1].scale(x=l2))
                    .scale(x=-lr)) # don't miss the minus sign here
                uds.append(layer.bias.update(d_bias, Site.add_inplace))
                uds.append(layer.grad_bias.update({}))
            if Train.WEIGHTS in layer.train:
                weights_sd = (layer.grad_weights[-1]
                    .pvariance().neg().exp()
                    .scale(x=scale))
                d_weights = (layer.grad_weights[0]
                    .normalvariate(weights_sd, c=layer.weights.const)
                    .sum(layer.weights[-1].scale(x=l2))
                    .scale(x=-lr))  # don't miss the minus sign here
                uds.append(layer.weights.update(d_weights, Site.add_inplace))
                uds.append(layer.grad_weights.update({}))
        self.system.schedule(self.update, *uds, dt=dt, priority=priority)
