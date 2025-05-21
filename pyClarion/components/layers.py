from typing import Self, Any, cast
from datetime import timedelta

from .base import Parametric, Component, V, DV
from .funcs import cam
from ..knowledge import Atoms, Family, Atom
from ..system import Site, Priority, Event
from ..numdicts import Key, KeyForm, NumDict
from ..numdicts.ops.base import Unary, Aggregator


class Layer(Component):
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
    func: Unary[NumDict] | None
    fw_by: KeyForm
    bw_by: KeyForm

    def __init__(self, 
        name: str, 
        s1: V | DV,
        s2: V | DV | None = None,
        *, 
        func: Unary[NumDict] | None = None, 
        l: int = 1
    ) -> None:
        s2 = s1 if s2 is None else s2
        super().__init__(name)
        idx_in, idx_out = self._init_indexes(s1, s2)
        self.func = func
        self.main = Site(idx_out, {}, 0.0, l)
        self.input = Site(idx_in, {}, 0.0, l)
        self.wsum = Site(idx_out, {}, 0.0, l)
        self.bias = Site(idx_out, {}, 0.0, l)
        self.weights = Site(idx_in * idx_out, {}, 0.0, l)
        self.fw_by = idx_in.kf * idx_out.kf.agg
        self.bw_by = idx_in.kf.agg * idx_out.kf

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.system.schedule(self.forward())
        # if self.main.affected_by(*updates, grad=True):
        #     self.system.schedule(self.backward())

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
        if self.func:
            main = self.func(wsum)            
        return Event(self.forward, 
            (self.wsum.update(wsum), 
             self.main.update(main), 
             self.weights.update(self.weights[0]),
             self.bias.update(self.bias[0])),
            dt, priority)
        
    # def backward(self, 
    #     dt: timedelta = timedelta(), 
    #     priority: Priority = Priority.PROPAGATION
    # ) -> Event:
    #     """
    #     Compute gradients and backpropagate errors.
        
    #     Computed gradients from successive calls to this method will accumulate 
    #     at gradient sites. This allows layers to receive asynchronous error 
    #     signals. 
        
    #     Typically, gradient sites will be cleared by an optimizer after it has 
    #     consumed their data for weight updates. 
    #     """
    #     grad_wsum = self.main.grad[0]
    #     if self.func:
    #         grad_wsum = grad_wsum.mul(self.func.grad(self.wsum[-1]))
    #     grad_bias = grad_wsum
    #     grad_weights = (self.weights[-1].zeros()
    #         .mul(self.input[-1], grad_wsum, by=(self.fw_by, self.bw_by)))
    #     back = (self.weights[-1]
    #         .mul(grad_wsum, by=self.bw_by)
    #         .sum(by=self.fw_by)) 
    #     return Event(self.backward,
    #         (self.input.update(back, grad=True),
    #          self.bias.update(grad_bias, Site.add_inplace, grad=True),
    #          self.weights.update(grad_weights, Site.add_inplace, grad=True)),
    #         dt, priority)


class Pool(Parametric):
    """
    An activation pooling process.

    Combines activation strengths from multiple sources.
    """

    class Params(Atoms):
        pass

    p: Params
    main: Site
    aggregate: Site
    params: Site
    inputs: dict[Key, Site]
    scaled: dict[Key, Site]
    agg: Aggregator[NumDict]
    post: Unary[NumDict] | None

    def __init__(self, 
        name: str, 
        p: Family, 
        s: V | DV, 
        *, 
        agg: Aggregator[NumDict] = cam, 
        post: Unary[NumDict] | None = None,
        l: int = 1
    ) -> None:
        super().__init__(name)
        index, = self._init_indexes(s)
        psort, psite = self._init_sort(p, type(self).Params, l=l)
        self.p = psort
        self.params = psite
        self.main = Site(index, {}, 0.0, l=l)
        self.aggregate = Site(index, {}, 0.0, l=l)
        self.inputs = {}
        self.scaled = {}
        self.agg = agg
        self.post = post

    def __rrshift__(self: Self, other: Any) -> Self:
        if isinstance(other, tuple):
            if not all(isinstance(elt, Component) for elt in other):
                return NotImplemented
            inputs = cast(tuple[Component, ...], other)
        elif isinstance(other, Component):
            inputs = (other,)
        else:
            return NotImplemented
        for comp in inputs:
            site = getattr(comp, "main", None)
            if not isinstance(site, Site):
                return NotImplemented
            if not self.main.index.kf <= site.index.kf:
                raise ValueError("Input to pool does not match main keyform")
            self.p[comp.name] = Atom()
            key = ~self.p[comp.name]
            self.inputs[key] = site
            self.scaled[key] = Site(site.index, {}, site.const, len(site.data)) 
            with self.params[0].mutable():
                self.params[0][key] = 1.0
        return self
        
    # def resolve(self, event: Event) -> None:
    #     updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
    #     if self.main.affected_by(*updates, grad=True):
    #         self.system.schedule(self.backward()) 

    def forward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        inputs = [s[0].scale(self.params[0][k]) 
            for k, s in self.inputs.items()]
        aggregate = self.agg(*inputs)
        if (post := self.post) is None:
            main = aggregate
        else:
            main = post(aggregate)
        return Event(self.forward, 
            (self.params.update(self.params[0].d),
             self.main.update(main), 
             self.aggregate.update(aggregate),
             *(site.update(data) 
               for site, data in zip(self.scaled.values(), inputs))), 
            dt, priority)
    
    # def backward(self, 
    #     dt: timedelta = timedelta(), 
    #     priority: Priority = Priority.LEARNING
    # ) -> Event:
    #     grad_agg = self.main.grad[0]
    #     if (post := self.post) is not None:
    #         grad_agg = grad_agg.mul(post.grad(self.aggregate[-1]))
    #     scaled = (self.main.new({}), *(s[-1] for s in self.scaled.values()))
    #     grads = [grad.mul(grad_agg) for grad in self.agg.grad(*scaled)]
    #     updates = []
    #     for (k, site), grad in zip(self.inputs.items(), grads[1:]):
    #         ud = site.update(grad.scale(self.params[-1][k]), grad=True)
    #         updates.append(ud)
    #     return Event(self.backward, updates, dt, priority)
