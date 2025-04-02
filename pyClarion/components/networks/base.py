from typing import Self
from datetime import timedelta
from enum import Flag, auto

from ..base import V, DV, DualRepMixin, ParamMixin
from ...system import Process, Event, Priority, PROCESS, Site
from ...knowledge import Family, Atoms
from ...numdicts import NumDict, KeyForm


class Train(Flag):
    """
    A training flag.
    
    Used to selectively control which elements of a neural network or layer are 
    subject to training via backpropagation. 
    """
    NIL = 0
    WEIGHTS = auto()
    BIAS = auto()
    ALL = WEIGHTS | BIAS


class Activation:
    """A differentiable activation function."""

    def __call__(self, d: NumDict) -> NumDict:
        """Compute activations for each input value."""
        raise NotImplementedError()
    
    def grad(self, d: NumDict) -> NumDict:
        """Compute activation derivative with respect to each input value."""
        raise NotImplementedError()
    
    def scale(self, layer: "Layer") -> float:
        """Compute scaling factor for weight initialization."""
        raise NotImplementedError()


class Layer(DualRepMixin, Process):
    """
    A neural network layer.
    
    Implements forward propagation of activation signals and backward 
    propagation of error signals.
    """

    main: Site
    wsum: Site
    input: Site
    weights: Site
    bias: Site
    afunc: Activation | None
    train: Train
    fw_by: KeyForm
    bw_by: KeyForm

    def __init__(self, 
        name: str, 
        s1: V | DV,
        s2: V | DV | None = None,
        *, 
        afunc: Activation | None = None, 
        l: int = 0,
        train: Train = Train.ALL,
        init_sd: float = 1e-2
    ) -> None:
        s2 = s1 if s2 is None else s2
        super().__init__(name)
        idx_in, idx_out = self._init_indexes(s1, s2)
        self.afunc = afunc
        self.train = train
        self.main = Site(idx_out, {}, 0.0, l)
        self.input = Site(idx_in, {}, 0.0, l)
        self.wsum = Site(idx_out, {}, 0.0, l)
        self.bias = Site(idx_out, {}, 0.0, l)
        self.weights = Site(idx_in * idx_out, {}, 0.0, l)
        self.fw_by = idx_in.kf * idx_out.kf.agg
        self.bw_by = idx_in.kf.agg * idx_out.kf
        self.init_weights(init_sd)
        self._connect_to_optimizer()

    def init_weights(self, sd: float = 1e-2) -> None:
        with self.bias[0].mutable():
            self.bias[0].reset()
        with self.weights[0].mutable():
            self.weights[0].reset()
        if self.afunc:
            scale = sd * self.afunc.scale(self) 
        else: 
            scale = sd / (1 + len(self.input))
        if Train.BIAS in self.train:
            bias_sd = self.bias[0].const().scale(x=scale)
            bias_init = (self.bias[0]
                .const(c=0.0)
                .normalvariate(bias_sd, c=self.bias.const))
            with self.bias[0].mutable():
                self.bias[0].update(bias_init.d)
        if Train.WEIGHTS in self.train:
            weight_sd = self.weights[0].const().scale(x=scale)
            weight_init = (self.weights[0]
                .const(c=0.0)
                .normalvariate(weight_sd, c=self.weights.const))
            with self.weights[0].mutable():
                self.weights[0].update(weight_init.d)

    def _connect_to_optimizer(self):
        try:
            sup = PROCESS.get()
        except LookupError:
            return
        if not isinstance(sup, Optimizer):
            return
        sup.add(self)

    def __rshift__[T](self, other: T) -> T:
        if isinstance(other, Layer):
            other.input = self.main
            return other
        return NotImplemented

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.system.schedule(self.forward)
        if self.main.affected_by(*updates, grad=True):
            self.system.schedule(self.backward)

    def forward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        """Compute and propagate forward activations."""
        wsum = (self.weights[0]
            .mul(self.input[0], by=self.fw_by)
            .sum(by=self.bw_by)
            .sum(self.bias[0]))
        main = wsum
        if self.afunc:
            main = self.afunc(wsum)            
        return Event(self.forward, 
            (self.wsum.update(wsum), 
             self.main.update(main), 
             self.weights.update(self.weights[0]),
             self.bias.update(self.bias[0])),
            dt, priority)
        
    def backward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        """
        Compute gradients and backpropagate errors.
        
        Computed gradients from successive calls to this method will accumulate 
        at gradient sites. This allows layers to receive asynchronous error 
        signals. 
        
        Typically, gradient sites will be cleared by an optimizer after it has 
        consumed their data for weight updates. 
        """
        grad_wsum = self.main.grad[0]
        if self.afunc:
            grad_wsum = grad_wsum.mul(self.afunc.grad(self.wsum[-1]))
        grad_bias = grad_wsum
        grad_weights = (self.weights[-1].const()
            .mul(self.input[-1], grad_wsum, by=(self.fw_by, self.bw_by)))
        back = (self.weights[-1]
            .mul(grad_wsum, by=self.bw_by)
            .sum(by=self.fw_by)) 
        return Event(self.backward,
            (self.input.update(back, grad=True),
             self.bias.update(grad_bias, Site.add_inplace, grad=True),
             self.weights.update(grad_weights, Site.add_inplace, grad=True)),
            dt, priority)
        

class Optimizer[P: Atoms](ParamMixin, Process):
    """
    A neural network optimization process. 

    Issues updates to weights and biases of a collection of layers. 
    """

    Params: type[P]
    p: P
    params: Site
    layers: set[Layer]

    def __init__(self, name: str, p: Family, **params: float) -> None:
        super().__init__(name)
        self.p, self.params = self._init_params(p, type(self).Params, **params)
        self.layers = set()

    def add(self, layer: Layer) -> None:
        """Include layer in future updates."""
        self.layers.add(layer)

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        """Compute and schedule parameter updates for all client layers."""
        raise NotImplementedError()
    

class ErrorSignal(Process):
    """
    A neural network error signaling process.
    
    Computes and backpropagates error signals based on neural network outputs.
    """

    main: Site

    def __rrshift__(self: Self, other: Layer) -> Self:
        if isinstance(other, Layer):
            self.input = other.main
            return self
        return NotImplemented
    
    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> Event:
        """Compute and schedule update to error value."""
        raise NotImplementedError()
