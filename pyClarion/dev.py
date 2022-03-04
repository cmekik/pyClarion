"""Tools for developing new pyClarion components."""


from .base import feature, dimension, chunk, Process
from .base.uris import (FSEP, SEP, SUP, ID, ispath, join, split, split_head, 
    commonprefix, remove_prefix, relativize, prefix)
from typing import overload, Dict, Any, Tuple, Iterable, Callable, TypeVar, List


__all__ = ["FSEP", "SEP", "SUP", "ID", "Process", "lag", "first", "second", 
    "group_by", "group_by_dims", "ispath", "join", "split", "split_head", 
    "commonprefix", "remove_prefix", "relativize", "prefix"]


T = TypeVar("T")


def eye(x: T) -> T:
    """Return input x (identity function)."""
    return x


@overload
def lag(x: feature, val: int = 1) -> feature:
    ...

@overload
def lag(x: dimension, val: int = 1) -> dimension:
    ...

def lag(x, val = 1):
    """Return a copy of x with lag incremented by val."""
    if isinstance(x, dimension):
        return dimension(x.id, x.lag + val)
    elif isinstance(x, feature):
        return feature(x.d, x.v, x.l + val)
    else:
        raise TypeError(f"Expected 'feature' or 'dimension', got {type(x)}")


def first(pair: Tuple[T, Any]) -> T:
    """Return the first element in a pair."""
    return pair[0]


def second(pair: Tuple[T, Any]) -> T:
    """Return the second element in a pair."""
    return pair[1]


def cf2cd(key: Tuple[chunk, feature]) -> Tuple[chunk, dimension]:
    """Convert a chunk-feature pair to a chunk-dimension pair."""
    return key[0], key[1].dim


def group_by(iterable: Iterable, key: Callable) -> Dict[Any, Tuple]:
    """Return a dict grouping items in iterable by values of the key func."""
    groups: dict = {}
    for item in iterable:
        k = key(item)
        groups.setdefault(k, []).append(item)
    return {k: tuple(v) for k, v in groups.items()}


def group_by_dims(
    features: Iterable[feature]
) -> Dict[dimension, Tuple[feature, ...]]:
    """
    Construct a dict grouping features by their dimensions.
    
    Returns a dict where each dim is mapped to a tuple of features of that dim.
    Does not check for duplicate features.

    :param features: An iterable of features to be grouped by dimension.
    """
    return group_by(iterable=features, key=feature.dim.fget)
