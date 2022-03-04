
from __future__ import annotations

__all__ = ["NumDict"]

from . import basic_ops as bops
from . import dict_ops as dops
from . import vec_ops as vops
from . import nn_ops

from typing import Callable, Dict, Mapping, TypeVar, Iterator, Any, Optional
from typing_extensions import Concatenate, ParamSpec
from functools import wraps
from math import isnan, isinf


P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
T1, T2 = TypeVar("T1"), TypeVar("T2")


def inplace(
    f: Callable[Concatenate["NumDict[T]", P], R]
) -> Callable[Concatenate["NumDict[T]", P], R]:

    @wraps(f)
    def wrapper(d: "NumDict[T]", *args: P.args, **kwargs: P.kwargs) -> R:
        if d.prot: raise RuntimeError("Cannot mutate protected NumDict data.")
        return f(d, *args, **kwargs)

    return wrapper


class NumDict(Mapping[T, float]):
    """
    A numerical dictionary.

    Represents a mapping from symbolic keys to (real) numerical values with 
    support for mathematical operations, op chaining, and automatic 
    differentiation. 

    :param m: Symbol-value associations represented by the NumDict.
    :param c: Optional constant, default value for all keys not in the map given 
        by m.
    :param prot: Bool indicating whether the NumDict is protected. When True, 
        in-place operations are disabled.
    """

    __slots__ = ("_m", "_c", "_prot")

    _m: Dict[T, float]
    _c: float
    _prot: bool 

    def __init__(
        self, 
        m: Optional[Mapping[T, Any]] = None,
        c: Any = 0.0, 
        prot: bool = False
    ) -> None:
        self._m = {k: float(v) for k, v in m.items()} if m else {}
        self._c = float(c) 
        self._prot = prot

    @classmethod
    def _new(
        cls, 
        m: Optional[Dict[T, float]] = None, 
        c: float = 0.0, 
        prot: bool = False
    ) -> "NumDict[T]":
        # Fast instance constructor (omits checks); for use in op defs
        new = cls.__new__(cls)
        new._m = m if m is not None else {}
        new._c = c
        new._prot = prot
        return new

    @property
    def m(self) -> Dict[T, float]:
        return self._m.copy()

    @property
    def c(self) -> float:
        """The NumDict constant."""
        return self._c

    @c.setter
    @inplace
    def c(self, val: float) -> None:
        self._c = float(val)

    @property
    def prot(self) -> bool:
        """Bool indicating whether self is protected."""
        return self._prot

    @prot.setter
    def prot(self, val: bool) -> None:
        self._prot = bool(val)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, NumDict):
            return self._m == other._m and self.c == other.c
        else:
            return NotImplemented

    def __repr__(self) -> str:
        m = f"{repr(self._m)}" if self._m else ""
        c = f"c={self.c}"
        prot = f"prot={self.prot}" if self.prot else ""
        args = ", ".join(arg for arg in (m, c, prot) if arg)
        return f"{type(self).__name__}({args})"

    def __len__(self) -> int:
        return len(self._m)

    def __iter__(self) -> Iterator[T]:
        yield from iter(self._m)

    def __contains__(self, key: Any) -> bool:
        return key in self._m

    def __getitem__(self, key: Any) -> float:
        try:
            return self._m[key] # type: ignore
        except KeyError:
            return self._c

    @inplace
    def __setitem__(self, key: Any, val: float | int | bool) -> None:
        self._m[key] = float(val) # type: ignore

    @inplace
    def __delitem__(self, key: Any) -> None:
        del self._m[key]

    @inplace
    def clear(self) -> None:
        """Clear all key-value associations in self."""
        self._m = {}

    @inplace
    def update(
        self, m: Mapping[T, Any], clear: bool = False, strict: bool = False
    ) -> None:
        """
        Update self with new key-value pairs.
        
        Behaves like dict.update(). Will overwrite existing values.

        :param m: Mapping containing new key-value pairs.
        :param clear: If True, clear self prior to running the update. 
        :param strict: If True, will throw a ValueError if existing values are 
            overwritten.
        """
        if clear: self.clear()
        n_old = len(self)
        self._m.update({k: float(v) for k, v in m.items()})
        if strict and len(self._m) < n_old + len(m): 
            raise ValueError("Arg m not disjoint with self")

    def has_inf(self) -> bool:
        """Return True iff any key is mapped to inf or the constant is inf."""
        return any(map(isinf, self.values())) or isinf(self.c)

    def has_nan(self) -> bool:
        """Return True iff any key is mapped to nan or the constant is nan."""
        return any(map(isnan, self.values())) or isnan(self.c)
    
    def copy(self: "NumDict[T]") -> "NumDict[T]":
        """Return an unprotected copy of self."""
        return type(self)(m=self, c=self.c)

    def pipe(
        self, 
        f: Callable[Concatenate["NumDict[T]", P], "NumDict[T2]"], 
        *args: P.args, 
        **kwdargs: P.kwargs
    ) -> "NumDict[T2]":
        """Call a custom function as part of a NumDict method chain."""
        return f(self, *args, **kwdargs)

    ### Mathematical Dunder Methods ###

    __neg__ = bops.neg
    __abs__ = bops.absolute
    
    __lt__ = bops.less
    __gt__ = bops.greater
    __leq__ = bops.less_equal
    __geq__ = bops.greater_equal

    __add__ = bops.add
    __radd__ = bops.add
    __mul__ = bops.mul
    __rmul__ = bops.mul
    __sub__ = bops.sub
    __rsub__ = bops.rsub
    __truediv__ = bops.div
    __rtruediv__ = bops.rdiv
    __pow__ = bops.power
    __rpow__ = bops.rpow

    __or__ = bops.maximum
    __ror__ = bops.maximum
    __and__ = bops.minimum
    __rand__ = bops.minimum

    __matmul__ = vops.matmul
    __rmatmul__ = vops.matmul

    ### Mathematical Methods ###

    isfinite = bops.isfinite
    isnan = bops.isnan
    isinf = bops.isinf
    replace_inf = bops.replace_inf
    neg = bops.neg
    abs = bops.absolute
    sign = bops.sign
    log = bops.log
    exp = bops.exp
    
    isclose = bops.isclose
    less = bops.less
    greater = bops.greater
    less_equal = bops.less_equal
    greater_equal = bops.greater_equal

    add = bops.add
    mul = bops.mul
    sub = bops.sub
    rsub = bops.rsub
    div = bops.div
    rdiv = bops.rdiv
    pow = bops.power
    rpow = bops.rpow

    max = bops.maximum
    min = bops.minimum

    ### Dict Ops ###

    mask = dops.mask
    set_c = dops.set_c
    isolate = dops.isolate
    keep = dops.keep
    drop = dops.drop
    keep_less = dops.keep_less
    keep_greater = dops.keep_greater
    keep_if = dops.keep_if
    squeeze = dops.squeeze
    with_keys = dops.with_keys
    transform_keys = dops.transform_keys
    merge = dops.merge

    ### Aggregation Ops ###

    reduce_sum = vops.reduce_sum
    matmul = vops.matmul
    reduce_max = vops.reduce_max
    reduce_min = vops.reduce_min
    put = vops.put
    mul_from = vops.mul_from
    div_from = vops.div_from
    sum_by = vops.sum_by
    max_by = vops.max_by
    min_by = vops.min_by
    eltwise_max = vops.eltwise_max
    eltwise_min = vops.eltwise_min
    outer = vops.outer

    ### NN Ops ###

    sigmoid = nn_ops.sigmoid
    tanh = nn_ops.tanh
    boltzmann = nn_ops.boltzmann
    sample = nn_ops.sample
    cam_by = nn_ops.cam_by
    eltwise_cam = nn_ops.eltwise_cam
