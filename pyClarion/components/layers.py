from typing import Self, Any, cast
from datetime import timedelta
from collections import deque

from .base import Parametric, Backpropagator, Component, V, DV
from .ops import cam
from ..knowledge import Atoms, Family, Atom
from ..system import Site, Priority, Event
from ..numdicts import Key, KeyForm, NumDict
from ..numdicts.ops.base import Unary, Aggregator
from ..numdicts.ops.tape import GradientTape


class Layer(Backpropagator):
    """
    A neural network layer.
    
    Implements forward propagation of activation signals and backward 
    propagation of error signals.
    """

    main: Site
    input: Site
    weights: Site
    bias: Site
    func: Unary[NumDict] | None
    fw_by: KeyForm
    bw_by: KeyForm
    tapes2: deque[tuple[GradientTape, NumDict, NumDict, NumDict, NumDict]]

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
        self.main = Site(idx_out, {}, 0.0)
        self.input = Site(idx_in, {}, 0.0)
        self.bias = Site(idx_out, {}, 0.0)
        self.weights = Site(idx_in * idx_out, {}, 0.0)
        self.tapes = deque([], maxlen=l)
        self.fw_by = idx_in.kf * idx_out.kf.agg
        self.bw_by = idx_in.kf.agg * idx_out.kf

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.system.schedule(self.forward())
        if len(self.tapes) == self.tapes.maxlen \
            and self.main.affected_by(*updates, grad=True):
            self.system.schedule(self.backward())

    def forward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        """Compute and propagate forward activations."""
        input, weights, bias = self.input[0], self.weights[0], self.bias[0]
        with GradientTape() as tape:
            main = (weights
                .mul(input, by=self.fw_by)
                .sum(by=self.bw_by)
                .sum(bias))
            if self.func:
                main = self.func(main)            
        return Event(self.forward, 
            [self.main.update(main),
             self.push_tape(tape, main, [input, weights, bias])],
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
        tape, main, args = self.tapes[-1]
        g_main = self.main.grad[0]
        g_i, g_w, g_b = tape.gradients(main, args, g_main) 
        return Event(self.backward,
            [self.input.update(g_i, grad=True),
             self.weights.update(g_w, Site.add_inplace, grad=True),
             self.bias.update(g_b, Site.add_inplace, grad=True)],
            dt, priority)


class Pool(Parametric, Backpropagator):
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
        self.tapes = deque([], maxlen=l)
        self.inputs = {}
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
            with self.params[0].mutable():
                self.params[0][key] = 1.0
        return self
        
    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.main.affected_by(*updates, grad=True):
            self.system.schedule(self.backward()) 

    def forward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        with GradientTape() as tape:
            inputs = [s[0].scale(self.params[0][k]) 
                for k, s in self.inputs.items()]
            aggregate = self.agg(*inputs)
            if (post := self.post) is None:
                main = aggregate
            else:
                main = post(aggregate)
        return Event(self.forward, 
            [self.main.update(main), 
             self.aggregate.update(aggregate),
             self.push_tape(tape, main, [s[0] for s in self.inputs.values()])], 
            dt, priority)
    
    def backward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> Event:
        tape, main, inputs = self.tapes[-1]
        g_agg = self.main.grad[0]
        grads = tape.gradients(main, inputs, g_agg)
        updates = []
        for (k, site), grad in zip(self.inputs.items(), grads):
            ud = site.update(grad, grad=True)
            updates.append(ud)
        return Event(self.backward, updates, dt, priority)
