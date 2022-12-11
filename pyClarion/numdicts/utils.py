from __future__ import annotations

from . import numdict as nd

from typing import (Callable, Union, Iterable, TypeVar, Any, Set, Dict, List,   
    overload, Optional, Tuple)
from math import copysign, exp
from math import isclose as _isclose
from math import isinf as _isinf
from math import isnan as _isnan
from math import isfinite as _isfinite

T, T1, T2 = TypeVar("T"), TypeVar("T1"), TypeVar("T2")
NDLike = Union["nd.NumDict[T]", float, int, bool]


def coerce2(
    f: Callable[[nd.NumDict[T], nd.NumDict[T]], nd.NumDict[T]]
) -> Callable[[nd.NumDict[T], NDLike[T]], nd.NumDict[T]]:

    def wrapper(d1: nd.NumDict[T], d2: NDLike[T]) -> nd.NumDict[T]:
        d2 = nd.NumDict[T](c=d2) if isinstance(d2, (float, int, bool)) else d2
        return f(d1, d2)

    wrapper.__name__ = f.__name__
    wrapper.__qualname__ = f.__qualname__

    return wrapper


def op1(f: Callable[[float], float], d: nd.NumDict[T]) -> nd.NumDict[T]:
    return nd.NumDict._new(m={k: f(v) for k, v in d.items()}, c=f(d.c))


def op2(
    f: Callable[[float, float], float], d1: nd.NumDict[T], d2: nd.NumDict[T]
) -> nd.NumDict[T]:
    keys = set(d1) | set(d2)
    return nd.NumDict._new(
        m={k: f(d1[k], d2[k]) for k in keys}, 
        c=f(d1.c, d2.c))


### ABSTRACT AGGREGATION FUNCTIONS ###


def reduce(
    d: nd.NumDict[Any], 
    *,
    f: Callable[[Iterable[float]], float], 
    initial: float, 
    key: Optional[T] = None
) -> nd.NumDict[T]:
    values = [initial]; values.extend(d.values())
    result = f(values)
    if key is None: 
        return nd.NumDict[T]._new(m={}, c=result)
    else: 
        return nd.NumDict[T]._new(m={key: result})


def by(
    d: nd.NumDict[T1], *, 
    f: Callable[[Iterable[float]], float], 
    kf: Callable[[T1], T2],
) -> nd.NumDict[T2]:
    groups: Dict[T2, List[float]] = {}
    for k, v in d.items(): groups.setdefault(kf(k), []).append(v)
    return nd.NumDict._new(m={k: f(v) for k, v in groups.items()})


def eltwise(
    *ds: nd.NumDict[T], f: Callable[[Iterable[float]], float]
) -> nd.NumDict[T]:
    if len(ds) < 1: raise ValueError("At least one input is necessary.")
    ks: Set[T] = set(); ks = ks.union(*ds)
    return nd.NumDict._new(
        m={k: f([d[k] for d in ds]) for k in ks}, 
        c=f([d.c for d in ds]))


# ARITHMETIC HELPER FUNCTIONS #


def isinf(x: float) -> float:
    return float(_isinf(x))


def isnan(x: float) -> float:
    return float(_isnan(x))


def isfinite(x: float) -> float:
    return float(_isfinite(x))


def isclose(x: float, y: float) -> float:
    return float(_isclose(x, y))


def sign(x: float) -> float:
    return copysign(1.0, x) if x != 0.0 else 0.0


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x)) if x >= 0 else exp(x) / (1 + exp(x))


def tanh(x: float) -> float:
    return 2 * sigmoid(2 * x) - 1


def lt(x: float, y: float) -> float:
    return float(x < y)


def gt(x: float, y: float) -> float:
    return float(x > y)


def le(x: float, y: float) -> float:
    return float(x <= y)


def ge(x: float, y: float) -> float:
    return float(x >= y)


### OTHER UTILITIES ###


def first(pair: Tuple[T, Any]) -> T:
    """Return the first element in a pair."""
    return pair[0]


def second(pair: Tuple[Any, T]) -> T:
    """Return the second element in a pair."""
    return pair[1]