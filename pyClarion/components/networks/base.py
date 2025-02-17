from typing import Callable, cast, ClassVar, Type
from datetime import timedelta
from enum import Flag, auto

from ...system import Process, Event, Priority, PROCESS, Site
from ...knowledge import keyform, Family, Atoms
from ...numdicts import NumDict, KeyForm, path, Index


def sq_err(est: NumDict, tgt: NumDict, mask: NumDict) -> NumDict:
    return est.sub(tgt).pow(x=2).mul(mask).scale(x=.5)


def grad_sq_err(est: NumDict, tgt: NumDict, mask: NumDict) -> NumDict:
    return est.sub(tgt).mul(mask)


def sech_sq(d: NumDict) -> NumDict:
    return d.const().sub(d.tanh().pow(x=2.0))


def expit_grad(d: NumDict) -> NumDict:
    expit = d.expit()
    return expit.mul(expit.const().sub(expit))


Activation = Callable[[NumDict], NumDict]
Cost = Callable[[NumDict, NumDict, NumDict], NumDict]


class Backprop(Process):
    class Train(Flag):
        NIL = 0
        WEIGHTS = auto()
        BIAS = auto()
        ALL = WEIGHTS | BIAS

    GRADFUNCS: ClassVar[dict[Activation | Cost, Activation | Cost]] = {
        NumDict.eye: NumDict.const,
        NumDict.exp: NumDict.exp,
        NumDict.log: NumDict.inv,
        NumDict.tanh: sech_sq,
        NumDict.expit: expit_grad,
        sq_err: grad_sq_err
    }

    @classmethod
    def grad[T: Activation | Cost](cls, f: T) -> T:
        return cast(T, cls.GRADFUNCS[f])

    @classmethod
    def register[T: Activation | Cost](cls, f: T, g: T) -> None:
        cls.GRADFUNCS[f] = g    

    @classmethod
    def check_grad(cls, f: Activation | Cost) -> None:
        try:
            cls.GRADFUNCS[f]
        except KeyError as e:
            raise ValueError(f"Activation {f} has no "
                "registered derivative") from e


class LayerBase(Backprop):

    main: Site
    wsum: Site
    input: Site
    error: Site
    back: Site
    weights: Site
    bias: Site
    grad_weights: Site
    grad_bias: Site
    f: Activation
    train: Backprop.Train
    fw_by: KeyForm
    bw_by: KeyForm

    def __init__(self, name: str, f: Activation, train: Backprop.Train) -> None:
        super().__init__(name)
        self._connect_to_learning_rule()
        if f is not None: 
            type(self).check_grad(f)
        self.f = f
        self.train = train

    def _connect_to_learning_rule(self):
        try:
            sup = PROCESS.get()
        except LookupError:
            return
        if not isinstance(sup, Optimizer):
            return
        sup.add(self)

    def _init(self, idx_in: Index, idx_out: Index, lags: int) -> None:
        self.main = Site(idx_out, {}, 0.0, lags)
        self.input = Site(idx_in, {}, 0.0, lags)
        self.wsum = Site(idx_out, {}, 0.0, lags)
        self.bias = Site(idx_out, {}, 0.0, lags)
        self.weights = Site(idx_in * idx_out, {}, 0.0, lags)
        self.error = Site(idx_out, {}, 0.0)
        self.back = Site(idx_in, {}, 0.0)
        self.grad_bias = Site(idx_out, {}, 0.0)
        self.grad_weights = Site(idx_in * idx_out, {}, 0.0)
        self.fw_by = idx_in.keyform * idx_out.keyform.agg
        self.bw_by = idx_in.keyform.agg * idx_out.keyform

    def __rshift__[T](self, other: T) -> T:
        if isinstance(other, LayerBase):
            other.input = self.main
            self.error = other.back
            return other
        return NotImplemented

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.forward()
        if self.error.affected_by(*updates):
            self.backward()

    def forward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        wsum = (self.weights[0]
            .mul(self.input[0], by=self.fw_by)
            .sum(by=self.bw_by)
            .sum(self.bias[0]))
        main = self.f(wsum)            
        self.system.schedule(self.forward, 
            self.wsum.update(wsum), 
            self.main.update(main), 
            self.weights.update(self.weights[0]),
            self.bias.update(self.bias[0]),
            dt=dt, priority=priority)
        
    def backward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        grad_wsum = self.error[0].mul(self.grad(self.f)(self.wsum[-1]))
        grad_bias = grad_wsum
        grad_weights = (self.weights[-1].const()
            .mul(self.input[-1], grad_wsum, by=(self.fw_by, self.bw_by)))
        back = (self.weights[-1]
            .mul(grad_wsum, by=self.bw_by)
            .sum(by=self.fw_by)) 
        self.system.schedule(self.backward,
            self.back.update(back),
            self.grad_bias.update(grad_bias, Site.add_inplace),
            self.grad_weights.update(grad_weights, Site.add_inplace),
            dt=dt, priority=priority)
        

class Optimizer[P: Atoms](Process):
    Params: Type[P]
    layers: set[LayerBase]
    p: P

    def __init__(self, name: str, p: Family, **params: float) -> None:
        super().__init__(name)
        self.system.check_root(p)
        self.p = type(self).Params()
        p[name] = self.p
        self.layers = set()
        self.params = Site(
            i=self.system.get_index(keyform(self.p)), 
            d={path(self.p[k]): v for k, v in params.items()}, 
            c=float("nan"))

    def add(self, layer: LayerBase) -> None:
        self.layers.add(layer)

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        raise NotImplementedError()