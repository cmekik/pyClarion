"""Functions on numerical dictionaries without autodiff support."""
#functions to differentiate
#boltzmann
#clip, threshold
#keep, drop, transform_keys
__all__ = [
    "epsilon", "freeze", "unfreeze", "with_default", "isclose", "keep", "drop", 
    "squeeze", "transform_keys", "threshold", "clip", "boltzmann", "draw", "by", 
    "elementwise", "ew_sum", "ew_mean", "ew_max", "ew_min", "valuewise", 
    "val_sum", "val_max", "val_min", "all_val", "any_val", 
    "exponential_moving_avg", "tabulate"
]

from .numdicts import NumDict, MutableNumDict, D

from typing import (
    TypeVar, Callable, Hashable, Dict, Union, List, Any, Container, Optional
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


def squeeze(d: D, default: float = None) -> NumDict:
    """
    Return a copy of d dropping explicit members close to the default .
    
    :param default: Default value to assume if self.default is None. If 
        provided when self.default is defined, will be ignored.
    """

    if d.default is not None:
        default = d.default
    elif default is not None:
        pass
    else:
        raise ValueError("Cannot squeeze numdict with no default.")

    mapping = {k: v for k, v in d.items() if not math.isclose(v, default)}

    return NumDict(mapping, default)


def with_default(d: D, *, default: Optional[Union[float, int]]) -> NumDict:

    return NumDict(d, default=default)


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


def threshold(
    d: D, *, th: Union[float, int], keep_default: bool = False
) -> NumDict:
    """
    Return a copy of d containing only values above theshold.
    
    If the default is below threshold it is set to None in the output, unless 
    keep default is True.
    """

    mapping = {k: d[k] for k in d if th < d[k]}
    if d.default is not None:
        default = d.default if keep_default or th < d.default else None 

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


def boltzmann(d: D, t: Union[float, int]) -> NumDict:
    """
    Construct a boltzmann distribution from d with temperature t.

    If d has a default, the returned value will have a default of 0, and, if d 
    is empty, the return value will also be empty.
    """

    default = 0 if d.default is not None else None
    if len(d) > 0:
        x = d / t
        x = x - val_max(x) # softmax(x) = softmax(x + c)
        numerators = x.exp()
        denominator = val_sum(numerators)
        return with_default(numerators / denominator, default=default)
    else:
        return NumDict(default=default)


def draw(
    d: D, 
    n: int = 1, 
    val: Union[float, int] = 1.0, 
    default: Union[float, int] = 0.0
) -> NumDict:
    """
    Draw k keys from numdict without replacement.
    
    If k >= len(d), returns a selection of all elements in d. 
    
    Sampled elements are given a val of 1.0 by default. Output inherits its
    default value from d.
    """

    pr = MutableNumDict(d)
    output = MutableNumDict()
    if len(d) > n:
        while len(output) < n:
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
    """Recursively apply commutative binary op to explicit values of d."""

    if not 0 < len(d):
        raise ValueError("Arg d must be non-empty.")

    v = initial
    for item in d.values():
        v = op(v, item)

    return v


def val_sum(d: D) -> float:
    """Return the sum of the values of d."""

    return valuewise(operator.add, d, 0.0)


def val_max(d: D) -> float:
    """Return the maximum explicit value in d."""

    return valuewise(max, d, float("-inf"))


def val_min(d: D) -> float:
    """Return the minimum explicit value in d."""

    return valuewise(max, d, float("+inf"))

def all_val(d: D) -> bool:
    """Return True if all values, including the default, are truthy."""

    return all(v for v in d.values()) and bool(d.default)

def any_val(d: D) -> bool:
    """Return True if any values, including the default, are truthy."""

    return any(v for v in d.values()) or bool(d.default)


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
