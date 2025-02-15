from typing import Callable, Sequence, Self, overload, cast
from datetime import timedelta
from enum import Flag, auto

from ..system import Process, Event, UpdateSite, Priority, PROCESS
from ..knowledge import keyform, Family, Sort, Atoms, Atom
from ..numdicts import NumDict, numdict, KeyForm, path


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
    GRADFUNCS: dict[Activation | Cost, Activation | Cost] = {
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


class Train(Flag):
    NIL = 0
    WEIGHTS = auto()
    BIAS = auto()
    ALL = WEIGHTS | BIAS


class Optimizer(Process):
    
    def add(self, layer: "LayerBase") -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    class Params(Atoms):
        lr: Atom
        sd: Atom

    sites: list["LayerBase"]
    p: Params
    params: NumDict
    
    def __init__(self, 
        name: str, 
        p: Family, 
        *, 
        lr: float = 1e-2, 
        sd: float = 1e-4
    ) -> None:
        super().__init__(name)
        self.system.check_root(p)
        self.p = type(self).Params(); p[name] = self.p
        self.sites = []
        self.params = numdict(
            i=self.system.get_index(keyform(self.p)), 
            d={path(self.p.lr): lr, path(self.p.sd): sd}, 
            c=float("nan"))

    def add(self, layer: "LayerBase") -> None:
        self.sites.append(layer)

    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> None:
        if not self.sites:
            return
        lr = self.params[path(self.p.lr)]
        sd = self.params[path(self.p.sd)]
        uds = []
        for layer in self.sites:
            if Train.BIAS in layer.train:
                new_bias = (layer.bias
                    .sub(layer.grad_bias
                        .normalvariate(layer.grad_wsum.abs().scale(x=sd))
                        .scale(x=lr)))
                uds.append(UpdateSite(layer.bias, new_bias.d))
                uds.append(UpdateSite(layer.grad_bias, {}))
            if Train.WEIGHTS in layer.train:
                new_weights = (layer.weights
                    .sub(layer.grad_weights
                        .normalvariate(layer.grad_wsum.abs().scale(x=sd))
                        .scale(x=lr)))
                uds.append(UpdateSite(layer.weights, new_weights.d))
                uds.append(UpdateSite(layer.grad_weights, {}))
        self.system.schedule(self.update, *uds, dt=dt, priority=priority)


class SupervisionBase(Backprop):
    cost: Cost
    main: NumDict
    error: NumDict
    input: NumDict
    target: NumDict
    mask: NumDict

    def __init__(self, name: str, cost: Cost = sq_err) -> None:
        super().__init__(name)
        type(self).check_grad(cost)
        self.cost = cost

    def __rrshift__(self: Self, other: "LayerBase") -> Self:
        if isinstance(other, LayerBase):
            self.input = other.main
            other.error = self.error
            return self
        return NotImplemented
    
    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> None:
        error = self.grad(self.cost)(self.input, self.target, self.mask.exp())
        main = self.cost(self.input, self.target, self.mask.exp())
        self.system.schedule(self.update, 
            UpdateSite(self.main, main.d), 
            UpdateSite(self.error, error.d), 
            dt=dt, priority=priority)
    

class SupervisionBL(SupervisionBase):
    def __init__(self, 
        name: str, 
        d: Family | Sort | Atom, 
        v: Family | Sort
    ) -> None:
        super().__init__(name)
        self.system.check_root(d, v)
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        self.main = numdict(idx_d * idx_v, {}, c=0.0)
        self.error = numdict(idx_d * idx_v, {}, c=0.0)
        self.input = numdict(idx_d * idx_v, {}, c=0.0)
        self.target = numdict(idx_d * idx_v, {}, c=0.0)
        self.mask = numdict(idx_d * idx_v, {}, c=0.0)


class LayerBase(Backprop):
    main: NumDict
    wsum: NumDict
    input: NumDict
    error: NumDict
    back: NumDict
    weights: NumDict
    bias: NumDict
    grad_weights: NumDict
    grad_bias: NumDict
    grad_wsum: NumDict
    f: Activation
    fw_by: KeyForm
    bw_by: KeyForm

    def __init__(self, name: str, f: Activation, train: Train) -> None:
        super().__init__(name)
        self._connect_to_optimizer()
        if f is not None: 
            type(self).check_grad(f)
        self.f = f
        self.train = train

    def _connect_to_optimizer(self):
        try:
            sup = PROCESS.get()
        except LookupError:
            return
        if not isinstance(sup, Optimizer):
            return
        sup.add(self)

    def __rshift__[T](self, other: T) -> T:
        if isinstance(other, LayerBase):
            other.input = self.main
            self.error = other.back
            return other
        return NotImplemented

    def resolve(self, event: Event) -> None:
        if event.affects(self.input):
            self.forward()
        if event.affects(self.error):
            self.backward()

    def forward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        wsum = (self.weights
            .mul(self.input, by=self.fw_by)
            .sum(by=self.bw_by)
            .sum(self.bias))
        main = self.f(wsum)            
        self.system.schedule(self.forward, 
            UpdateSite(self.wsum, wsum.d), 
            UpdateSite(self.main, main.d), 
            dt=dt, priority=priority)
        
    def backward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        grad_wsum = self.error.mul(self.grad(self.f)(self.wsum))
        grad_bias = self.grad_bias.sum(grad_wsum)
        grad_weights = (self.grad_weights
            .sum(self.weights
                .const()
                .mul(self.input, grad_wsum, by=(self.fw_by, self.bw_by))))
        back = (self.weights
            .mul(grad_bias, by=self.bw_by)
            .sum(by=self.fw_by)) 
        self.system.schedule(self.backward,
            UpdateSite(self.back, back.d),
            UpdateSite(self.grad_weights, grad_weights.d),
            UpdateSite(self.grad_bias, grad_bias.d),
            UpdateSite(self.grad_wsum, grad_wsum.d),
            dt=dt, priority=priority)


class Layer(LayerBase):
    def __init__(self, 
        name: str, 
        h_in: Sort, 
        h_out: Sort, 
        *, 
        f: Activation = NumDict.eye,
        train: Train = Train.ALL
    ) -> None:
        super().__init__(name, f, train)
        self.system.check_root(h_in, h_out)
        idx_in = self.system.get_index(keyform(h_in))
        idx_out = self.system.get_index(keyform(h_out))
        self.main = numdict(idx_out, {}, 0.0)
        self.wsum = numdict(idx_out, {}, 0.0)
        self.error = numdict(idx_out, {}, 0.0)
        self.input = numdict(idx_in, {}, 0.0)
        self.back = numdict(idx_in, {}, 0.0)
        self.weights = numdict(idx_in * idx_out, {}, 0.0)
        self.bias = numdict(idx_out, {}, 0.0)
        self.grad_weights = numdict(idx_in * idx_out, {}, 0.0)
        self.grad_bias = numdict(idx_out, {}, 0.0)
        self.grad_wsum = numdict(idx_out, {}, 0.0)
        self.fw_by = keyform(h_in).agg * keyform(h_out)
        self.bw_by = keyform(h_in) * keyform(h_out).agg


class InputLayerBL(LayerBase):
    def __init__(self, 
        name: str, 
        d: Family | Sort | Atom, 
        v: Family | Sort,
        h: Sort, 
        *, 
        f: Activation = NumDict.eye,
        train: Train = Train.ALL
    ) -> None:
        super().__init__(name, f, train)
        self.system.check_root(d, v, h)
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        idx_h = self.system.get_index(keyform(h))
        self.main = numdict(idx_h, {}, 0.0)
        self.wsum = numdict(idx_h, {}, 0.0)
        self.error = numdict(idx_h, {}, 0.0)
        self.input = numdict(idx_d * idx_v, {}, 0.0)
        self.back = numdict(idx_d * idx_v, {}, 0.0)
        self.weights = numdict(idx_d * idx_v * idx_h, {}, 0.0)
        self.bias = numdict(idx_h, {}, 0.0)
        self.grad_weights = numdict(idx_d * idx_v * idx_h, {}, 0.0)
        self.grad_bias = numdict(idx_h, {}, 0.0)
        self.grad_wsum = numdict(idx_h, {}, 0.0)
        self.fw_by = keyform(d) * keyform(v) * keyform(h).agg
        self.bw_by = keyform(d).agg * keyform(v).agg * keyform(h)


class OutputLayerBL(LayerBase):
    def __init__(self, 
        name: str, 
        h: Sort, 
        d: Family | Sort | Atom, 
        v: Family | Sort,
        *, 
        f: Activation = NumDict.eye,
        train: Train = Train.ALL
    ) -> None:
        super().__init__(name, f, train)
        self.system.check_root(h, d, v)
        idx_h = self.system.get_index(keyform(h))
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        self.main = numdict(idx_d * idx_v, {}, 0.0)
        self.wsum = numdict(idx_d * idx_v, {}, 0.0)
        self.error = numdict(idx_d * idx_v, {}, 0.0)
        self.input = numdict(idx_h, {}, 0.0)
        self.back = numdict(idx_h, {}, 0.0)
        self.weights = numdict(idx_h * idx_d * idx_v, {}, 0.0)
        self.bias = numdict(idx_d * idx_v, {}, 0.0)
        self.grad_weights = numdict(idx_h * idx_d * idx_v, {}, 0.0)
        self.grad_bias = numdict(idx_d * idx_v, {}, 0.0)
        self.grad_wsum = numdict(idx_d * idx_v, {}, 0.0)
        self.fw_by = keyform(h) * keyform(d).agg * keyform(v).agg
        self.bw_by = keyform(h).agg * keyform(d) * keyform(v)
