from typing import Self, Any, cast
from datetime import timedelta
from collections import deque

from .base import Parametric, Backpropagator, Component, Priority
from .ops import cam
from ..knowledge import Atoms, Family, Atom, Nodes
from ..events import State, Site, Event, ForwardUpdate, BackwardUpdate
from ..numdicts import Key, KeyForm, NumDict
from ..numdicts.ops.base import Unary, Aggregator
from ..numdicts.ops.tape import GradientTape


class Mapping[I: Nodes, O: Nodes](Backpropagator):
    """
    Transforms an input signal according to a given function.
    
    Implements forward propagation of activation signals and backward 
    propagation of error signals.
    """

    i: I
    o: O
    main: Site = Site()
    input: Site = Site(lax=True)
    func: Unary[NumDict] | None
    fw_by: KeyForm
    bw_by: KeyForm

    def __init__(self, 
        name: str, 
        i: I,
        o: O,
        func: Unary[NumDict] | None = None, 
        *,
        l: int = 1
    ) -> None:
        super().__init__(name)
        idx_in, idx_out = self._init_indexes(i, o)
        self.i = i
        self.o = o
        self.func = func
        self.main = State(idx_out, {}, 0.0)
        self.input = State(idx_in, {}, 0.0)
        self.tapes = deque([], maxlen=l)

    def resolve(self, event: Event) -> None:
        forward = event.index(ForwardUpdate)
        backward = event.index(BackwardUpdate)
        if self.input in forward:
            self.system.schedule(self.forward())
        if len(self.tapes) == self.tapes.maxlen and self.main in backward:
            self.system.schedule(self.backward())

    def forward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        """Compute and propagate forward activations."""
        input = self.input[0]
        with GradientTape() as tape:
            main = self.main.new({}).sum(input)
            if self.func is not None:
                main = self.func(main)        
        self.push_tape(tape, main, [input])            
        return Event(self.forward, 
            [ForwardUpdate(self.main, main)],
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
        g_i, = tape.gradients(main, args, g_main) 
        return Event(self.backward,
            [BackwardUpdate(self.input, g_i)],
            dt, priority)
    

class Accumulator[D: Nodes](Backpropagator):
    """
    Transforms an input signal according to a given function.
    
    Implements forward propagation of activation signals and backward 
    propagation of error signals.
    """

    d: D
    main: Site = Site()
    input: Site = Site(lax=True)
    func: Unary[NumDict] | None
    fw_by: KeyForm
    bw_by: KeyForm

    def __init__(self, 
        name: str, 
        d: D,
        *,
        l: int = 1
    ) -> None:
        super().__init__(name)
        idx, = self._init_indexes(d)
        self.d = d
        self.main = State(idx, {}, 0.0)
        self.input = State(idx, {}, 0.0)
        self.tapes = deque([], maxlen=l)

    def resolve(self, event: Event) -> None:
        forward = event.index(ForwardUpdate)
        backward = event.index(BackwardUpdate)
        if self.input in forward:
            self.system.schedule(self.forward())
        if len(self.tapes) == self.tapes.maxlen and self.main in backward:
            self.system.schedule(self.backward())

    def forward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        """Compute and propagate forward activations."""
        input = self.input[0]
        with GradientTape() as tape:
            main = self.main[0].sum(input)
        self.push_tape(tape, main, [input])            
        return Event(self.forward, 
            [ForwardUpdate(self.main, main)],
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
        g_i, = tape.gradients(main, args, g_main) 
        return Event(self.backward,
            [BackwardUpdate(self.input, g_i)],
            dt, priority)


class Layer[I: Nodes, O: Nodes](Backpropagator):
    """
    A neural network layer.
    
    Implements forward propagation of activation signals and backward 
    propagation of error signals.
    """

    i: I
    o: O
    main: Site = Site()
    input: Site = Site()
    weights: Site = Site()
    bias: Site = Site()
    func: Unary[NumDict] | None
    fw_by: KeyForm
    bw_by: KeyForm

    def __init__(self, 
        name: str, 
        i: I,
        o: O,
        *, 
        func: Unary[NumDict] | None = None, 
        l: int = 1
    ) -> None:
        super().__init__(name)
        idx_in, idx_out = self._init_indexes(i, o)
        self.i = i
        self.o = o
        self.func = func
        self.main = State(idx_out, {}, 0.0)
        self.input = State(idx_in, {}, 0.0)
        self.bias = State(idx_out, {}, 0.0)
        self.weights = State(idx_in * idx_out, {}, 0.0)
        self.tapes = deque([], maxlen=l)
        self.fw_by = idx_in.kf * idx_out.kf.agg
        self.bw_by = idx_in.kf.agg * idx_out.kf

    def resolve(self, event: Event) -> None:
        forward = event.index(ForwardUpdate)
        backward = event.index(BackwardUpdate)
        if self.input in forward:
            self.system.schedule(self.forward())
        if len(self.tapes) == self.tapes.maxlen and self.main in backward:
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
        self.push_tape(tape, main, [input, weights, bias])            
        return Event(self.forward, 
            [ForwardUpdate(self.main, main)],
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
            [BackwardUpdate(self.input, g_i),
             BackwardUpdate(self.weights, g_w, "add"),
             BackwardUpdate(self.bias, g_b, "add")],
            dt, priority)


class Pool[D: Nodes](Parametric, Backpropagator):
    """
    An activation pooling process.

    Combines activation strengths from multiple sources.
    """

    class Params(Atoms):
        pass

    p: Params
    d: D
    main: Site = Site()
    aggregate: Site = Site()
    params: Site = Site()
    inputs: dict[Key, State]
    agg: Aggregator[NumDict]
    post: Unary[NumDict] | None

    def __init__(self, 
        name: str, 
        p: Family, 
        d: D, 
        *, 
        agg: Aggregator[NumDict] = cam, 
        post: Unary[NumDict] | None = None,
        l: int = 1
    ) -> None:
        super().__init__(name)
        index, = self._init_indexes(d)
        psort, psite = self._init_sort(p, type(self).Params, l=l)
        self.p = psort
        self.d = d
        self.params = psite
        self.main = State(index, {}, 0.0, l=l)
        self.aggregate = State(index, {}, 0.0, l=l)
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
            if not isinstance(site, State):
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
        gradients = event.index(BackwardUpdate)
        if len(self.tapes) == self.tapes.maxlen and self.main in gradients:
            self.system.schedule(self.backward()) 

    def forward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        with GradientTape() as tape:
            inputs = [s[0].scale(self.params[0][k]).reindex(self.main.index.kf) 
                for k, s in self.inputs.items()]
            aggregate = self.agg(*inputs)
            if (post := self.post) is None:
                main = aggregate
            else:
                main = post(aggregate)
        self.push_tape(tape, main, [s[0] for s in self.inputs.values()])
        return Event(self.forward, 
            [ForwardUpdate(self.main, main), 
             ForwardUpdate(self.aggregate, aggregate)], 
            dt, priority)
    
    def backward(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> Event:
        tape, main, inputs = self.tapes[-1]
        g_agg = self.main.grad[0]
        grads = tape.gradients(main, inputs, g_agg)
        updates = []
        for (k, state), grad in zip(self.inputs.items(), grads):
            ud = BackwardUpdate(state, grad)
            updates.append(ud)
        return Event(self.backward, updates, dt, priority)
