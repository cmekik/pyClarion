"""Functions on numerical dictionaries without autodiff support."""


__all__ = [
    "epsilon", "freeze", "unfreeze", "isclose", "keep", "drop", 
    "transform_keys", "threshold", "clip", "boltzmann", "draw", "by", 
    "elementwise", "ew_sum", "ew_mean", "ew_max", "ew_min", "valuewise", 
    "val_sum", "exponential_moving_avg", "tabulate"
]

from .numdicts import NumDict, MutableNumDict, D

from typing import (
    TypeVar, Callable, Hashable, Dict, Union, List, Any, Container
)
import random
import operator
import math


def epsilon():
    """A very small value (1e-07),"""

    return 1e-07


def freeze(d: MutableNumDict) -> NumDict:
    """Return a frozen copy of d."""

    return NumDict(d, d.default)


def unfreeze(d: NumDict) -> MutableNumDict:
    """Return a mutable copy of d."""

    return MutableNumDict(d, d.default)


def isclose(d1: D, d2: D) -> bool:
    """Return True if self is close to other in values."""
    
    _d = d1._binary(d2, math.isclose) # make this a numdict method

    return all(_d.values())


def keep(
    d: D, 
    func: Callable[..., bool] = None, 
    keys: Container = None,
    **kwds: Any
) -> NumDict:
    """
    Return a copy of d keeping only the desired keys. 
    
    Keys are kept iff func(key, **kwds) or key in container is True.
    """

    if func is None and keys is None:
        raise ValueError("At least one of func or keys must not be None.")

    mapping = {
        k: d[k] for k in d 
        if (
            (func is not None and func(k, **kwds)) or 
            (keys is not None and k in keys)
        )
    }

    return NumDict(mapping, d.default)


def drop(
    d: D, 
    func: Callable[..., bool] = None, 
    keys: Container = None,
    **kwds: Any
) -> NumDict:
    """
    Return a copy of d dropping unwanted keys. 
    
    Keys are dropped iff func(key, **kwds) or key in container is True.
    """

    if func is None and keys is None:
        raise ValueError("At least one of func or keys must not be None.")

    mapping = {
        k: d[k] for k in d 
        if (func is not None and not func(k, **kwds)) or 
        (keys is not None and k not in keys)
    }

    return NumDict(mapping, d.default)


def transform_keys(d: D, *, func: Callable[..., Hashable], **kwds) -> NumDict:
    """
    Return a copy of d where each key is mapped to func(key, **kwds).

    Warning: If function is not one-to-one wrt keys, will raise ValueError.
    """

    mapping = {func(k, **kwds): d[k] for k in d}

    if len(d) != len(mapping):
        raise ValueError("Func must be one-to-one on keys of arg d.")

    return NumDict(mapping, d.default)


def threshold(d: D, *, th: float) -> NumDict:
    """
    Return a copy of d containing only values above theshold.
    
    If the default is below threshold, it is set to None in the output.
    """

    mapping = {k: d[k] for k in d if th < d[k]}
    if d.default is not None:
        default = d.default if th < d.default else None 

    return NumDict(mapping, default)


def clip(d: D, low: float = None, high: float = None) -> NumDict:
    """
    Return a copy of d with values clipped.
    
    dtype must define +/- inf values.
    """

    low = low or float("-inf")
    high = high or float("inf")

    mapping = {k: max(low, min(high, d[k])) for k in d}

    return NumDict(mapping, d.default)


def boltzmann(d: D, t: float) -> NumDict:
    """Construct a boltzmann distribution from d with temperature t."""

    output = MutableNumDict(default=0.0)
    if len(d) > 0:
        numerators = (d / t).exp()
        denominator = val_sum(numerators)
        output.max(numerators / denominator)

    return NumDict(output)


def draw(d: D, k: int=1, val=1.0) -> NumDict:
    """
    Draw k keys from numdict without replacement.
    
    If k >= len(d), returns a selection of all elements in d. Sampled elements 
    are given a value of 1.0 by default. Output inherits type, dtype and 
    default values from d.
    """

    pr = MutableNumDict(d)
    output = MutableNumDict()
    if len(d) > k:
        while len(output) < k:
            cs, ws = tuple(zip(*pr.items()))
            choices = random.choices(cs, weights=ws)
            output.extend(choices, value=val)
            pr.keep(output.__contains__)
    else:
        output.extend(d, value=val)
    
    return NumDict(output, d.default)


def by(
    d: D, 
    op: Callable[..., float],
    keyfunc: Callable[..., Hashable], 
    **kwds: Any
) -> NumDict:
    """
    Compute op over elements grouped by keyfunc.
    
    Key should be a function mapping each key in self to a grouping key. New 
    keys are determined based on the result of keyfunc(k, **kwds), where 
    k is a key from d.
    """

    _d: Dict[Hashable, List[float]] = {}
    for k, v in d.items():
        _d.setdefault(keyfunc(k, **kwds), []).append(v)
    mapping = {k: op(v) for k, v in _d.items()}

    return NumDict(mapping, d.default)


def elementwise(op: Callable[..., float], *ds: D) -> NumDict:
    """
    Apply op elementwise to a sequence of numdicts.
    
    If any numdict in ds has None default, then default is None, otherwise the 
    new default is calculated by running op on all defaults.
    """

    keys: set = set()
    keys.update(*ds)

    grouped: dict = {}
    defaults: list = []
    for d in ds:
        defaults.append(d.default)
        for k in keys:
            grouped.setdefault(k, []).append(d[k])
    
    if any([d is None for d in defaults]):
        default = None
    else:
        default = op(defaults)

    return NumDict({k: op(grouped[k]) for k in grouped}, default)


def ew_sum(*ds: D) -> NumDict:
    """
    Elementwise sum of values in ds.
    
    Wraps elementwise().
    """

    return elementwise(sum, *ds)


def ew_mean(*ds: D) -> NumDict:
    """
    Elementwise sum of values in ds.
    
    Wraps elementwise().
    """

    return elementwise(sum, *ds) / len(ds)


def ew_max(*ds: D)  -> NumDict:
    """
    Elementwise maximum of values in ds.
    
    Wraps elementwise().
    """

    return elementwise(max, *ds)


def ew_min(*ds: D) -> NumDict:
    """
    Elementwise maximum of values in ds.
    
    Wraps elementwise().
    """

    return elementwise(min, *ds)


def valuewise(
    op: Callable[[float, float], float], d: D, initial: float
) -> float:
    """Recursively apply commutative binary op to values of d."""

    v = initial
    for item in d.values():
        v = op(v, item)

    return v


def val_sum(d: D) -> Any:
    """Return the sum of the values of d."""

    return valuewise(operator.add, d, 0.0)


def exponential_moving_avg(d: D, *ds: D, alpha: float) -> List[NumDict]:
    """Given a sequence of numdicts, return a smoothed sequence."""

    avg: List[NumDict] = [d]
    for _d in ds:
        avg.append(alpha * _d + (1 - alpha) * avg[-1])

    return avg


def tabulate(*ds: D) -> Dict[Hashable, List[float]]:
    """
    Tabulate data from a sequence of numdicts.

    Produces a dictionary inheriting its keys from ds, and mapping each key to 
    a list such that the ith value of the list is equal to ds[i][k].
    """

    tabulation: Dict[Hashable, List[float]] = {}
    for d in ds:
        for k, v in d.items():
            l = tabulation.setdefault(k, [])
            l.append(v)

    return tabulation
