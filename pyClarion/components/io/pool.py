from typing import Final, Self

from ..base.backpropagator import Backpropagator, Backpropagation, PushTape
from ..base.parametric import Parametric

from ...events.system import Action, Process
from ...events.sites import Site, State
from ...events.updates import ForwardUpdate, BackwardUpdate
from ...knowledge import Nodes, Params, Param
from ...numdicts import keyform, NumDict, GradientTape, Key
from ...numdicts.ops.base import Unary, Aggregator


class Pool[P: Params, D: Nodes](Parametric[P], Backpropagator):
    """
    An activation pooling process.

    Combines activation strengths from multiple sources.
    """

    d: Final[D]
    output: Site = Site()
    inputs: dict[Key, State]

    def __init__(self, name: str, p: P, d: D) -> None:
        super().__init__(name, p)
        self.system.check_root(p)
        idx_d = self.system.get_index(keyform(d))
        self.d = d
        self.output = State(idx_d, {}, 0.0, self.system.depth)
        self.inputs = {}

    def __ilshift__(self: Self, other: Process) -> Self:
        state = getattr(other, "output", None)
        if not isinstance(other, Process):
            raise TypeError("Input to pool is not a process")
        elif not isinstance(state, State):
            raise TypeError("Input to pool does not have an output state.")
        elif not self.output.index.kf <= state.index.kf:
            raise ValueError("Input to pool does not match main keyform")
        self.p[other.name] = Param()
        key = ~self.p[other.name]
        self.inputs[key] = state
        with self.params[0].mutable():
            self.params[0][key] = 1.0
        return self


class PoolPropagation[P: Pool[Params, Nodes]](Action[P]):
    __slots__ = ()

    def __init__(self, 
        source: P, 
        agg: Aggregator[NumDict], 
        post: Unary[NumDict] | None = None
    ) -> None:
        params = source.params[0]
        kf_out = source.output.index.kf
        with GradientTape() as tape:
            aggregate = agg(*(
                s[0].scale(params[k]).reindex(kf_out) 
                for k, s in source.inputs.items()))
            output = aggregate if post is None else post(aggregate)
        variables = [params, *(s[0] for s in source.inputs.values())]
        super().__init__(source)
        self.updates.extend([
            ForwardUpdate(source.output, output),
            PushTape(source, tape, output, variables)
        ])


class PoolBackpropagation[P: Pool[Params, Nodes]](Backpropagation[P]):
    __slots__ = ()

    def __init__(self, source: P, index: int = -1) -> None:
        tape, output, variables = source.tapes[index]
        grads = tape.gradients(output, variables, source.output.grad[0])
        super().__init__(source, index)
        self.updates.append(BackwardUpdate(source.params, grads[0]))
        for state, grad in zip(source.inputs.values(), grads[1:]):
            self.updates.append(BackwardUpdate(state, grad))
   
