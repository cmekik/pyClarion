from typing import (SupportsFloat, overload, Self, Sequence, Callable, 
    Concatenate, ClassVar, Iterator, NamedTuple)
from inspect import Signature, Parameter, BoundArguments, signature
from contextlib import contextmanager
from contextvars import ContextVar
from collections import deque

import math
import random
import statistics as stats

from .base import NumDictBase, collect, group
from .indices import Index
from .keys import Key, KeyForm


def numdict(
    i: Index, 
    d: dict[Key, float] | dict[str, SupportsFloat], 
    c: SupportsFloat
) -> "NumDict":
    d = {Key(k): float(v) for k, v in d.items()}
    c = float(c)
    return NumDict(i, d, c)


class GradientTape:

    STACK: ClassVar[ContextVar["GradientTape | None"]] = ContextVar("STACK")
    STACK.set(None)

    gfuncs: dict["NumDict", "Node"]

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
        output: "NumDict",
        variables: list["NumDict"], 
        seed: "NumDict | None" = None
    ) -> list["NumDict"]:
        grads = {}
        for current, node in self._iter_nodes(output, seed):
            g = (NumDict.sum(*node.grads) if 0 < len(node.grads) 
                else current.zeros())
            grads[current] = g
            node.grads.clear()
            if node.gspec is not None:
                args = node.gspec.sig.args
                kwargs = node.gspec.sig.kwargs
                gs = node.gspec.op.grad(g, current, *args, **kwargs)
                if isinstance(gs, NumDict):
                    d = node.gspec.sig.args[0]
                    self.nodes[d].grads.append(gs)
                else:
                    assert len(node.gspec.sig.args) == len(gs)
                    for d, g_d in zip(node.gspec.sig.args, gs):
                        self.nodes[d].grads.append(g_d)
        result = []
        for v in variables:
            g = grads[v]
            result.append(g)
        return result

    def _iter_nodes(self, 
            output: "NumDict", seed: "NumDict | None" = None
        ) -> Iterator[tuple["NumDict", "GradientTape.Node"]]:
        if output not in self.nodes:
            raise ValueError("Not in graph")
        self.nodes[output].grads.append(seed or output.ones())
        queue: deque[NumDict] = deque([output])
        while queue:
            current = queue.popleft()
            try:
                node = self.nodes[current]
            except KeyError:
                continue
            else:
                yield (current, node)
                if node.gspec is not None:
                    queue.extend(node.gspec.sig.args)

    def record(self, f, r, d, *args, **kwargs) -> None:
        sig = signature(f).bind(*args, **kwargs)
        self.nodes[r] = self.Node([], self.OpData(f, sig))
        for d in sig.args:
            if d not in self.nodes:
                self.nodes[d] = self.Node([])

    class Node(NamedTuple):
        grads: list["NumDict"]
        gspec: "GradientTape.OpData | None" = None

    class OpData(NamedTuple):
        op: "Op"
        sig: BoundArguments


class Op:
    __name__ : str
    __qualname__: str
    __signature__: Signature
    __objclass__: type
    __call__: Callable
    grad: Callable

    def __set_name__(self, owner, name):
        self.__name__ = name
        self.__qualname__ = f"{owner.__name__}.{name}"
        self.__objclass__ = owner
        self.set_signature(owner)

    def set_signature(self, owner: type) -> None:
        self.__signature__ = signature(self.__call__)

class MethodType:
    __slots__ = ("__name__", "__self__", "__func__", "grad")
    __name__: str
    __self__: "NumDict"


class UnaryOp[**P](Op):

    class Method[**Q](MethodType):
        __slots__ = ()
        __func__: "UnaryOp[Q]"

        def __init__(self, op: "UnaryOp[Q]", obj: "NumDict") -> None:
            self.__name__ = op.__name__
            self.__func__ = op
            self.__self__ = obj
            self.grad = op.grad

        def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> "NumDict":
            func = self.__func__
            obj = self.__self__
            return func(obj, *args, **kwargs)

    @overload
    def __get__(self, obj: "NumDict", objtype=None) -> Method:
        ... 

    @overload
    def __get__(self, obj, objtype=None) -> Self:
        ...

    def __get__(self, obj, objtype=None):
        if isinstance(obj, NumDict):
            return UnaryOp.Method(self, obj)
        return self

    def __call__(self, d: "NumDict", /, *args: P.args, **kwargs: P.kwargs) -> "NumDict":
        ...

    def grad(self, g: "NumDict", r: "NumDict", d: "NumDict", /, *args: P.args, **kwargs: P.kwargs) -> "NumDict":
        raise NotImplementedError


class Constant(UnaryOp[[]]):
    c: float

    def __init__(self, c: float) -> None:
        self.c = c

    def __call__(self, d: "NumDict", /) -> "NumDict":
        r = type(d)(d._i, {}, self.c, False)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d)
        return r

    def set_signature(self, owner: type):
        param0 = Parameter(name="d", kind=Parameter.POSITIONAL_ONLY)
        self.__signature__ = Signature(parameters=(param0,), return_annotation=owner)

    
class Unary[**P](UnaryOp[P]):
    kernel: Callable[Concatenate[float, P], float]

    def __init__(self, kernel: Callable[Concatenate[float, P], float]) -> None:
        self.kernel = kernel

    def __call__(
        self, d: "NumDict", /, *args: P.args, **kwargs: P.kwargs
    ) -> "NumDict":
        new_c = float(self.kernel(d._c, *args, **kwargs))
        new_d = {k: float(new_v) for k, v in d._d.items() 
            if (new_v := self.kernel(v, *args, **kwargs)) != new_c}
        r = type(d)(d._i, new_d, new_c, False)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d, *args, **kwargs)
        return r

    def set_signature(self, owner: type):
        sig = signature(self.kernel)
        kernel_params = list(sig.parameters.values())
        param0 = Parameter(name="d", kind=Parameter.POSITIONAL_ONLY)
        self.__signature__ = Signature(
            parameters=(param0, *kernel_params[1:]), return_annotation=owner)


class Binary(Op):

    class Method(MethodType):
        __slots__ = ()
        __func__: "Binary"

        def __init__(self, op: "Binary", obj: "NumDict") -> None:
            self.__name__ = op.__name__
            self.__func__ = op
            self.__self__ = obj
            self.grad = op.grad

        def __call__(self, d2: "NumDict", /, by: KeyForm | None = None) -> "NumDict":
            func = self.__func__
            obj = self.__self__
            return func(obj, d2, by=by)

    kernel: Callable[[float, float], float]

    def __init__(self, kernel: Callable[[float, float], float]) -> None:
        self.kernel = kernel

    @overload
    def __get__(self, obj: "NumDict", objtype=None) -> Method:
        ... 

    @overload
    def __get__(self, obj, objtype=None) -> Self:
        ...

    def __get__(self, obj, objtype=None):
        if isinstance(obj, NumDict):
            return Binary.Method(self, obj)
        return self

    def __call__(self, d1: "NumDict", d2: "NumDict", /, by: KeyForm | None = None) -> "NumDict":
        it = collect(d1, d2, mode="match", branches=(by,))
        new_c = self.kernel(d1._c, d2._c)
        new_d = {k: v for k, (v1, v2) in it 
            if (v := self.kernel(v1, v2)) != new_c}
        r = type(d1)(d1._i, new_d, new_c, False)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d1, d2, by=by)
        return r

    def grad(self, g: "NumDict", r: "NumDict", d1: "NumDict", d2: "NumDict", /, by: KeyForm | None = None) -> "NumDict":
        raise NotImplementedError
    

class Ternary(Op):

    class Method(MethodType):
        __slots__ = ()
        __func__: "Ternary"

        def __init__(self, op: "Ternary", obj: "NumDict") -> None:
            self.__name__ = op.__name__
            self.__func__ = op
            self.__self__ = obj
            self.grad = op.grad

        def __call__(self, d2: "NumDict", d3: "NumDict", /, by: KeyForm | None = None) -> "NumDict":
            func = self.__func__
            obj = self.__self__
            return func(obj, d2, d3, by=by)

    kernel: Callable[[float, float, float], float]

    def __init__(self, kernel: Callable[[float, float, float], float]) -> None:
        self.kernel = kernel

    @overload
    def __get__(self, obj: "NumDict", objtype=None) -> Method:
        ... 

    @overload
    def __get__(self, obj, objtype=None) -> Self:
        ...

    def __get__(self, obj, objtype=None):
        if isinstance(obj, NumDict):
            return Ternary.Method(self, obj)
        return self

    def __call__(self, d1: "NumDict", d2: "NumDict", d3: "NumDict", /, by: KeyForm | tuple[KeyForm, KeyForm] | None = None) -> "NumDict":
        if isinstance(by, KeyForm):
            by = (by, by)
        it = collect(d1, d2, mode="match", branches=by)
        new_c = self.kernel(d1._c, d2._c, d3._c)
        new_d = {k: v for k, (v1, v2, v3) in it 
            if (v := self.kernel(v1, v2, v3)) != new_c}
        r = type(d1)(d1._i, new_d, new_c, False)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d1, d2, d3, by=by)
        return r

    def grad(self, g: "NumDict", r: "NumDict", d1: "NumDict", d2: "NumDict", d3: "NumDict", /, by: KeyForm | None = None) -> "NumDict":
        raise NotImplementedError


class Aggregator(Op):

    class Method(MethodType):
        __slots__ = ()
        __func__: "Aggregator"

        def __init__(self, op: "Aggregator", obj: "NumDict") -> None:
            self.__name__ = op.__name__
            self.__func__ = op
            self.__self__ = obj
            self.grad = op.grad

        def __call__(self, *ds: "NumDict", by: KeyForm | Sequence[KeyForm | None] | None = None, c: float | None = None) -> "NumDict":
            func = self.__func__
            obj = self.__self__
            return func(obj, *ds, by=by, c=c)

    kernel: Callable[[Sequence[float]], float]

    def __init__(self, kernel: Callable[[Sequence[float]], float], eye: float) -> None:
        self.kernel = kernel
        self.eye = eye

    @overload
    def __get__(self, obj: "NumDict", objtype=None) -> Method:
        ... 

    @overload
    def __get__(self, obj, objtype=None) -> Self:
        ...

    def __get__(self, obj, objtype=None):
        if isinstance(obj, NumDict):
            return Aggregator.Method(self, obj)
        return self

    def __call__(self, d: "NumDict", /, *ds: "NumDict", 
        by: KeyForm | Sequence[KeyForm | None] | None = None, c: float | None = None
    ) -> "NumDict":
        mode = "match" if ds else "self" if d._c == self.eye else "full"
        c = c or self.eye
        if len(ds) == 0 and by is None:
            by = d._i.kf.agg
            it = ()
            i = Index(d._i.root, by)
            assert mode != "match"
            new_c = self.kernel(group(d, by, mode=mode).get(Key(), (c,)))
        elif len(ds) == 0 and isinstance(by, KeyForm):
            if not by < d._i.kf:
                raise ValueError(f"Keyform {by.as_key()} cannot "
                    f"reduce {d._i.kf.as_key()}")
            assert mode != "match"
            it = group(d, by, mode=mode).items()
            i = Index(d._i.root, by)
            new_c = d._c if mode == "self" else c
        elif 0 < len(ds):
            if c is not None:
                ValueError("Unexpected float value for arg c")
            mode = "match"
            it = collect(d, *ds, branches=by, mode=mode)
            i = d._i
            new_c = self.kernel((d._c, *(other._c for other in ds)))
        else:
            assert False
        new_d = {k: v for k, vs in it if (v := self.kernel(vs)) != new_c}
        r = type(d)(i, new_d, new_c, False)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d, *ds, by, c)
        return r

    def grad(self, g: "NumDict", r: "NumDict", d: "NumDict", /, *ds: "NumDict", by: KeyForm | Sequence[KeyForm | None] | None = None, c: float | None = None) -> "NumDict":
        raise NotImplementedError


class UnaryRV(UnaryOp[[]]):
    kernel: Callable[[float], float]

    def __init__(self, kernel: Callable[[float], float]) -> None:
        self.kernel = kernel

    def __call__(self, d: "NumDict", /, c: float | None = None) -> "NumDict":
        it = collect(d, mode="full")
        c = c or d._c
        new_d = {k: v for k, vs in it if (v := self.kernel(*vs)) != c}
        r = type(d)(d._i, new_d, c, False)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d, c)
        return r

    def set_signature(self, owner: type):
        param0 = Parameter(name="d", kind=Parameter.POSITIONAL_ONLY, default=Parameter.empty)
        param1 = Parameter(name="c", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=float | None, default=None)
        self.__signature__ = Signature(parameters=(param0, param1), return_annotation=owner)


class BinaryRV(Op):

    class Method(MethodType):
        __slots__ = ()
        __func__: "BinaryRV"

        def __init__(self, op: "BinaryRV", obj: "NumDict") -> None:
            self.__name__ = op.__name__
            self.__func__ = op
            self.__self__ = obj
            self.grad = op.grad

        def __call__(self, d2: "NumDict", /, by: KeyForm | None = None, c: float | None = None) -> "NumDict":
            func = self.__func__
            obj = self.__self__
            return func(obj, d2, by=by, c=c)

    kernel: Callable[[float, float], float]

    def __init__(self, kernel: Callable[[float, float], float]) -> None:
        self.kernel = kernel

    @overload
    def __get__(self, obj: Method | None, objtype=None) -> Self:
        ...

    @overload
    def __get__(self, obj: "NumDict", objtype=None) -> Method:
        ... 

    def __get__(self, obj, objtype=None):
        if obj is None or isinstance(obj, BinaryRV.Method):
            return self
        return BinaryRV.Method(self, obj)

    def __call__(self, d1: "NumDict", d2: "NumDict", /, by: KeyForm | None = None, c: float | None = None) -> "NumDict":
        c = c or d1._c
        it = collect(d1, d2, branches=by, mode="full")
        new_d = {k: v for k, vs in it if (v := self.kernel(*vs)) != c}
        r = type(d1)(d1._i, new_d, c, False)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d1, d2, by=by, c=c)
        return r

    def grad(self, g: "NumDict", r: "NumDict", d1: "NumDict", d2: "NumDict", /, by: KeyForm | None = None, c: float | None = None) -> "NumDict":
        raise NotImplementedError


class NumDict(NumDictBase):
    zeros = Constant(0.0)
    ones = Constant(1.0)
    isfinite = Unary(math.isfinite)
    isnan = Unary(math.isnan)
    isinf = Unary(math.isinf)
    neg = Unary(float.__neg__)
    abs = Unary(abs)
    log = Unary(lambda x, /: -math.inf if x == 0.0 else math.log(x))
    log1p = Unary(math.log1p)
    exp = Unary(math.exp)
    expm1 = Unary(math.expm1)
    cos = Unary(math.cos)
    sin = Unary(math.sin)
    tan = Unary(math.tan)
    cosh = Unary(math.cosh)
    sinh = Unary(math.sinh)
    tanh = Unary(math.tanh)
    acos = Unary(math.acos)
    asin = Unary(math.asin)
    atan = Unary(math.atan)
    acosh = Unary(math.acosh)
    asinh = Unary(math.asinh)
    atanh = Unary(math.atanh)
    shift = Unary(float.__add__)
    scale = Unary(float.__mul__)
    pow = Unary(lambda __x, __y: __x ** __y)
    bound_max = Unary(lambda x, value, /: min(x, value))
    bound_min = Unary(lambda x, value, /: max(x, value))
    isclose = Binary(math.isclose)
    eq = Binary(float.__eq__)
    gt = Binary(float.__gt__)
    ge = Binary(float.__ge__)
    lt = Binary(float.__lt__)
    le = Binary(float.__le__)
    copysign = Binary(math.copysign)
    sub = Binary(float.__sub__)
    div = Binary(float.__truediv__)
    sum = Aggregator(math.fsum, 0.0)
    mul = Aggregator(math.prod, 1.0)
    max = Aggregator(max, -math.inf)
    min = Aggregator(min, math.inf)
    mean = Aggregator(stats.mean, math.nan)
    stdev = Aggregator(stats.stdev, math.nan)
    variance = Aggregator(stats.variance, math.nan)
    pstdev = Aggregator(stats.pstdev, math.nan)
    pvariance = Aggregator(stats.pvariance, math.nan)
    stduniformvariate = UnaryRV(lambda x, /: random.random())
    expovariate = UnaryRV(random.expovariate)
    paretovariate = UnaryRV(random.paretovariate)
    normalvariate = BinaryRV(random.normalvariate)
    lognormvariate = BinaryRV(random.lognormvariate)
    vonmisesvariate = BinaryRV(random.vonmisesvariate)
    gammavariate = BinaryRV(random.gammavariate)
