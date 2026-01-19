from typing import (Mapping, Iterator, Callable, Concatenate, Self, 
    SupportsFloat, overload)
from functools import wraps
from contextlib import contextmanager

import math

from .keys import KeyForm, Key
from .indices import Index, IndexObserver
from .undefined import _Undefined
from .ops.base import Constant

from .ops import defs


def numdict(
    i: Index, 
    d: dict[Key, float] | dict[str, SupportsFloat], 
    c: SupportsFloat | _Undefined
) -> "NumDict":
    d = {Key(k): float(v) for k, v in d.items()}
    c = c if isinstance(c, _Undefined) else float(c) 
    return NumDict(i, d, c)


def inplace[D: "NumDict", **P, R](
    f: Callable[Concatenate[D, P], R]
) -> Callable[Concatenate[D, P], R]:
    @wraps(f)
    def wrapper(d: D, *args: P.args, **kwargs: P.kwargs) -> R:
        if d._p: 
            raise RuntimeError("Cannot mutate protected NumDict data.")
        return f(d, *args, **kwargs)
    return wrapper


class NumDictBase(IndexObserver):
    __slots__ = ("_i", "_d", "_c", "_p")

    _i: Index
    _d: dict[Key, float]
    _c: float | _Undefined
    _p: bool

    def __init__(
        self, 
        i: Index,
        d: dict[Key, float], 
        c: float | _Undefined,
        _v: bool = True
    ) -> None:
        if _v: 
            for key in d:
                if key not in i:
                    raise ValueError(f"Key {key} not a member of index")
        self._i = i
        self._d = d 
        self._c = c
        self._p = True
        self.register(i)

    @property
    def i(self) -> Index:
        return self._i

    @property
    def d(self) -> dict[Key, float]:
        return {k: self._d[k] for k in self._d}

    @property
    def c(self) -> float | _Undefined:
        return self._c

    def __len__(self) -> int:
        return len(tuple(self._i))

    def __iter__(self) -> Iterator[Key]:
        yield from self._i

    def __contains__(self, key: str | Key) -> bool:
        k = Key(key)
        return k in self._i
    
    def __getitem__(self, key: str | Key) -> float:
        k = Key(key)
        try:
            return self._d[k]
        except KeyError as e:
            if k in self:
                c = self._c
                if isinstance(c, _Undefined):
                    raise KeyError(f"Key '{k}' is undefined") from e
                else:
                    return c
            else:
                raise KeyError(f"Key '{k}' not a member") from e

    def __repr__(self) -> str:
        return (f"<{type(self).__qualname__} '{self.i.kf.as_key()}' "
            f"c={self.c} at {hex(id(self))}>")
    
    def __str__(self) -> str:
        data = [f"{type(self).__qualname__} '{self.i.kf.as_key()}' c={self.c}"]
        width = 0
        for k in self.d:
            width = max(width, len(str(k)))
        for k, v in self.d.items():
            data.append(f"{str(k):<{width}} {v}")
        return "\n    ".join(data)

    def copy(self: Self) -> Self:
        return type(self)(self._i, self.d, self._c)

    def pipe[**P](
        self: Self, 
        f: Callable[Concatenate[Self, P], Self], 
        *args: P.args, 
        **kwdargs: P.kwargs
    ) -> Self:
        """Call a custom function as part of a NumDict method chain."""
        return f(self, *args, **kwdargs)

    def valmax(self) -> float:
        kmax, vmax = None, -math.inf
        for k in self:
            if self[k] > vmax:
                kmax, vmax = k, self[k] 
        assert kmax is not None 
        return vmax

    def valmin(self) -> float:
        kmin, vmin = None, math.inf
        for k in self:
            if self[k] < vmin:
                kmin, vmin = k, self[k] 
        assert kmin is not None
        return vmin

    @overload
    def argmax(self) -> Key:
        ...

    @overload
    def argmax(self, *, by: str | Key | KeyForm) -> dict[Key, Key]:
        ...

    def argmax(
        self, *, by: str | Key | KeyForm | None = None
    ) -> Key | dict[Key, Key]:
        it = self._d if isinstance(self._c, _Undefined) else self._i
        match by:
            case None:
                kmax, vmax = None, -math.inf
                for k in self:
                    if self[k] > vmax:
                        kmax, vmax = k, self[k] 
                assert kmax is not None 
                return kmax
            case by:
                if isinstance(by, (str, Key)):
                    by = KeyForm.from_key(Key(by))
                reduce = by.reductor(self.i.kf)
                kmax, vmax = {}, {}
                for k in it:
                    group, v = reduce(k), self[k]
                    if vmax.setdefault(group, -math.inf) < v:
                        kmax[group] = k
                        vmax[group] = v
                return {k: v for k, v in kmax.items()}

    @overload
    def argmin(self) -> Key:
        ...

    @overload
    def argmin(self, *, by: str | Key | KeyForm) -> dict[Key, Key]:
        ...

    def argmin(
        self, *, by: str | Key | KeyForm | None = None
    ) -> Key | dict[Key, Key]:
        it = self._d if isinstance(self._c, _Undefined) else self._i
        match by:
            case None:
                kmin, vmin = None, math.inf
                for k in it:
                    if self[k] < vmin:
                        kmin, vmin = k, self[k] 
                assert kmin is not None
                return kmin
            case by:
                if isinstance(by, (str, Key)):
                    by = KeyForm.from_key(Key(by))
                reduce = by.reductor(self.i.kf)
                kmin, vmin = {}, {}
                for k in it:
                    group, v = reduce(k), self[k]
                    if v < vmin.setdefault(group, math.inf):
                        kmin[group] = k
                        vmin[group] = v
                return {k: v for k, v in kmin.items()}

    @contextmanager
    def mutable(self):
        self._p = False
        yield self
        self._p = True

    @inplace
    def __setitem__(self, key: str | Key, value: float) -> None:
        k = Key(key)
        if k in self:
            self._d[k] = float(value)
        else:
            raise ValueError(f"Key '{key}' not a member")

    @c.setter
    @inplace
    def c(self, c: float) -> None:
        self._c = c

    @inplace
    def reset(self) -> None:
        self._d.clear()
    
    @inplace
    def update(
        self: Self, 
        data: Mapping[Key, SupportsFloat] | Mapping[str, SupportsFloat]
    ) -> None:
        for k in data:
            if k not in self:
                raise ValueError(f"Key '{k}' not a member")
        self._d.update({Key(k): float(v) for k, v in data.items()})
    
    def on_del(self, index: Index, key: Key) -> None:
        self._d.pop(key, None)


class NumDict(NumDictBase):

    zeros = Constant[Self](0.0)
    ones = Constant[Self](1.0)

    isfinite = defs.IsFinite[Self]()
    isnan = defs.IsNaN[Self]()
    isinf = defs.IsInf[Self]()
    isbetween = defs.IsBetween[Self]()

    reindex = defs.Reindex[Self]()

    neg = defs.Neg[Self]()
    inv = defs.Inv[Self]()
    abs = defs.Abs[Self]()

    log = defs.Log[Self]() 
    log1p = defs.Log1p[Self]()
    exp = defs.Exp[Self]()
    expm1 = defs.Expm1[Self]()
    cos = defs.Cos[Self]()
    sin = defs.Sin[Self]()
    tan = defs.Tan[Self]()
    cosh = defs.Cosh[Self]()
    sinh = defs.Sinh[Self]()
    tanh = defs.Tanh[Self]()
    acos = defs.Acos[Self]()
    asin = defs.Asin[Self]()
    atan = defs.Atan[Self]()
    acosh = defs.Acosh[Self]()
    asinh = defs.Asinh[Self]()
    atanh = defs.Atanh[Self]()

    scale = defs.Scale[Self]()
    shift = defs.Shift[Self]()
    pow = defs.Pow[Self]()
    clip = defs.Clip[Self]()

    eq = defs.Eq[Self]()
    gt = defs.Gt[Self]()
    ge = defs.Ge[Self]()
    lt = defs.Lt[Self]()
    le = defs.Le[Self]()
    isclose = defs.IsClose[Self]()
    copysign = defs.Copysign[Self]()

    sub = defs.Sub[Self]()
    div = defs.Div[Self]()

    sum = defs.Sum[Self]()
    mul = defs.Mul[Self]()
    max = defs.Max[Self]()
    min = defs.Min[Self]()

    mean = defs.Mean[Self]()
    stdev = defs.Stdev[Self]() 
    variance = defs.Variance[Self]() 
    pstdev = defs.Pstdev[Self]() 
    pvariance = defs.Pvariance[Self]() 

    stduniformvariate = defs.UniformVariate[Self]()
    expovariate = defs.ExpoVariate[Self]() 
    paretovariate = defs.ParetoVariate[Self]() 
    normalvariate = defs.NormalVariate[Self]() 
    lognormvariate = defs.LogNormVariate[Self]() 
    vonmisesvariate = defs.VonMisesVariate[Self]()
    gammavariate = defs.GammaVariate[Self]()
