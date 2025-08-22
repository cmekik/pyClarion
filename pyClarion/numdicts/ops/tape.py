from typing import ClassVar, Self, Sequence, Iterator, Iterable, NamedTuple, Protocol, cast
from contextvars import ContextVar
from contextlib import contextmanager
from collections import deque
from inspect import BoundArguments, signature

from .. import numdicts as nd


class OpProto[**P, D: "nd.NumDict"](Protocol):
    __name__: str

    def __call__(self, d: D, /, *args: P.args, **kwargs: P.kwargs) -> D:
        ...

    def grad(self, g: D, r: D, d: D, /, *args: P.args, **kwargs: P.kwargs) -> D | Sequence[D]:
        ...


class GradientTape[D: "nd.NumDict"]:

    STACK: ClassVar[ContextVar["GradientTape | None"]] = ContextVar("STACK")
    STACK.set(None)

    gfuncs: dict["nd.NumDict", "Node"]

    def __init__(self):
        self.spent = False
        self.nodes = {}

    def __enter__(self: Self) -> Self:
        if self.spent:
            raise RuntimeError("Cannot enter same AutoDiff context twice")
        self.spent = True
        self.tok = type(self).STACK.set(self)
        return self
    
    def __exit__(self, *args):
        type(self).STACK.reset(self.tok)
        del self.tok
    
    @contextmanager
    def no_grad(self):
        tok = type(self).STACK.set(None)
        yield
        type(self).STACK.reset(tok)

    def gradients(self, 
        output: D,
        variables: list[D], 
        seed: D | None = None
    ) -> list[D]:
        grads = {}
        for current, node in self._iter_nodes(output, seed):
            g = (node.grads[0].sum(*node.grads[1:]) if 1 < len(node.grads)
                else node.grads[0] if 1 == len(node.grads)  
                else current.zeros())
            grads[current] = g
            node.grads.clear()
            if node.gspec is not None:
                args = node.gspec.sig.args
                kwargs = node.gspec.sig.kwargs
                gs = node.gspec.op.grad(g, current, *args, **kwargs)
                if isinstance(gs, tuple):
                    assert len(node.gspec.sig.args) == len(gs)
                    for d, g_d in zip(node.gspec.sig.args, gs):
                        self.nodes[d].grads.append(g_d)
                else:
                    d = node.gspec.sig.args[0]
                    self.nodes[d].grads.append(gs)
        result = []
        for v in variables:
            g = grads[v]
            result.append(g)
        return result

    def _iter_nodes(self, 
            output: D, seed: D | None = None
        ) -> Iterator[tuple[D, "GradientTape.Node"]]:
        if output not in self.nodes:
            raise ValueError("Not in graph")
        self.nodes[output].grads.append(seed or output.ones())
        queue: deque[D] = deque([output])
        while queue:
            current = queue.popleft()
            try:
                node = self.nodes[current]
            except KeyError:
                continue
            else:
                yield (current, node)
                if node.gspec is not None:
                    queue.extend(
                        # This is a hack. Necessary to handle positional args
                        # that are non-numdict. Convention is that numdicts come 
                        # first. Need to clean up in the future.
                        # E.g., by checking that we are only getting 
                        # positional-only arguments.
                        cast(Iterable[D], [arg for arg in node.gspec.sig.args 
                         if isinstance(arg, nd.NumDict)]))

    def record[**P](self, 
        f: OpProto[P, D], r: D, d: D, 
        *args: P.args, **kwargs: P.kwargs
    ) -> None:
        sig = signature(f).bind(d, *args, **kwargs)
        self.nodes[r] = self.Node([], self.OpData(f, sig))
        for d in sig.args:
            if d not in self.nodes:
                self.nodes[d] = self.Node([])

    class Node(NamedTuple):
        grads: list["nd.NumDict"]
        gspec: "GradientTape.OpData | None" = None

    class OpData(NamedTuple):
        op: OpProto
        sig: BoundArguments
