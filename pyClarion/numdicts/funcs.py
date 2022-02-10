"""Functions on numerical dictionaries without autodiff support."""
__all__ = [
    "epsilon", "freeze", "unfreeze", "with_default", "isclose", 
    "squeeze",  "draw", "elementwise", "ew_sum", "ew_mean", "ew_max", "ew_min",
    "val_sum", "val_max", "val_min", "all_val", "any_val",
    "exponential_moving_avg", "tabulate"
]

from .numdicts import NumDict, MutableNumDict, D

from typing import (
    TypeVar, Callable, Hashable, Dict, Union, List, Any, Optional
)
import random
import math
import warnings

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

    _d = d1._binary(d2, math.isclose)  # make this a numdict method

    return all(_d.values())

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


def ew_max(*ds: D) -> NumDict:
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

def val_max(d: D) -> float:
    warnings.warn("val_max is Deprecated; Use reduce_max", DeprecationWarning)
    return max(d.values())

def val_sum(d: D) -> float:
    warnings.warn("val_sum is Deprecated; Use reduce_sum", DeprecationWarning)
    return sum(d.values(), 0)

def val_min(d: D) -> float:
    warnings.warn("val_min is Deprecated; Use reduce_min", DeprecationWarning)
    return min(d.values())

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

