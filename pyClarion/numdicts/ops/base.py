from typing import ClassVar, Self, Sequence, Callable, overload
from inspect import Signature, signature

from .funcs import collect, unary, binary, variadic
from .tape import OpProto, GradientTape
from ..keys import KeyForm
from .. import numdicts as nd


class OpMethod[**P, D: "nd.NumDict"]:
    __slots__ = ("__name__", "__self__", "__func__")
    __name__: str
    __func__: OpProto[P, D]
    __self__: D

    def __init__(self, op: OpProto[P, D], obj: D) -> None:
        self.__name__ = op.__name__
        self.__func__ = op
        self.__self__ = obj
    
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> D:
        return self.__func__(self.__self__, *args, **kwargs)
    

class OpBase[D: "nd.NumDict"]:
    
    __slots__ = (
        "__name__", "__qualname__", "__objclass__", "__signature__", "__call__")
    __name__ : str
    __qualname__: str
    __signature__: Signature
    __objclass__: type[D]
    __call__: Callable
    grad: Callable

    def __set_name__(self, owner: type[D], name: str) -> None:
        self.__name__ = name
        self.__qualname__ = f"{owner.__name__}.{name}"
        self.__objclass__ = owner
        self.set_signature(owner)

    @overload
    def __get__[**P](self: OpProto[P, D], 
        obj: D, 
        objtype: type[D] | None = None
    ) -> OpMethod[P, D]:
        ... 
        
    @overload
    def __get__[T](self, obj: T, objtype: type[T] | None = None) -> Self:
        ...

    def __get__(self, obj, objtype=None):
        if isinstance(obj, self.__objclass__):
            return OpMethod(self, obj)
        return self

    def set_signature(self, owner: type) -> None:
        self.__signature__ = signature(self.__call__)


class Constant[D: "nd.NumDict"](OpBase[D]):
    __slots__ = ("c",)
    c: float

    def __init__(self, c: float) -> None:
        self.c = c

    def __call__(self, d: D, /) -> D:
        r = type(d)(d._i, {}, self.c, False)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d)
        return r

    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.zeros()

    
class Unary[D: "nd.NumDict"](OpBase[D]):
    kernel: ClassVar[Callable[[float], float]]

    def __call__(self, d: D, /) -> D:
        r = unary(d, type(self).kernel)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d)
        return r
    
    def grad(self, g: D, r: D, d: D, /) -> D:
        raise NotImplementedError()
    

class UnaryDiscrete[D: "nd.NumDict"](Unary[D]):
    def grad(self, g: D, r: D, d: D, /) -> D:
        return d.zeros()


class Binary[D: "nd.NumDict"](OpBase[D]):
    kernel: ClassVar[Callable[[float, float], float]]

    def __call__(self, d1: D, d2: D, /, by: KeyForm | None = None) -> D:
        r = binary(d1, d2, by, None, type(self).kernel)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d1, d2, by=by)
        return r

    def grad(self, g: D, r: D, d1: D, d2: D, /, by: KeyForm | None = None) -> tuple[D, D]:
        raise NotImplementedError()


class BinaryDiscrete[D: "nd.NumDict"](Binary[D]):
    def grad(self, g: D, r: D, d1: D, d2: D, /, by: KeyForm | None = None) -> tuple[D, D]:
        return d1.zeros(), d2.zeros()


class UnaryRV[D: "nd.NumDict"](OpBase[D]):
    kernel: ClassVar[Callable[[float], float]]

    def __call__(self, d: D, /, c: float | None = None) -> D:
        it = collect(d, mode="full")
        c = c if c is not None else d._c
        new_d = {k: v for k, vs in it if (v := self.kernel(*vs)) != c}
        r = type(d)(d._i, new_d, c, False)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d, c)
        return r

    def grad(self, g: D, r: D, d: D, /, c: float | None = None) -> D:
        raise NotImplementedError()


class BinaryRV[D: "nd.NumDict"](OpBase[D]):
    kernel: ClassVar[Callable[[float, float], float]]

    def __call__(self, d1: D, d2: D, /, by: KeyForm | None = None, c: float | None = None) -> D:
        c = c if c is not None else d1._c
        it = collect(d1, d2, branches=by, mode="full")
        new_d = {k: v for k, vs in it if (v := self.kernel(*vs)) != c}
        r = type(d1)(d1._i, new_d, c, False)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d1, d2, by=by, c=c)
        return r

    def grad(self, g: "nd.NumDict", r: D, d1: D, d2: D, /, by: KeyForm | None = None, c: float | None = None) -> tuple[D, D]:
        raise NotImplementedError()


class Aggregator[D: "nd.NumDict"](OpBase[D]):
    kernel: ClassVar[Callable[[Sequence[float]], float]]
    eye: ClassVar[float]

    def __call__(self, d: D, /, *ds: D, by: KeyForm | Sequence[KeyForm | None] | None = None, c: float | None = None) -> D:
        r = variadic(d, *ds, by=by, c=c, kernel=type(self).kernel, eye=type(self).eye)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, d, *ds, by=by, c=c)
        return r

    def grad(self, g: D, r: D, d: D, /, *ds: D, by: KeyForm | Sequence[KeyForm | None] | None = None, c: float | None = None) -> D | Sequence[D]:
        raise NotImplementedError()