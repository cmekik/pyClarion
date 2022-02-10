from ..base.symbols import feature, dimension, chunk
from ..base import uris

from typing import overload, Dict, Any, Tuple, Iterable, Callable, TypeVar, List


__all__ = ["lag", "first", "second", "expand_dim", "group_by", "group_by_dims", 
    "collect_dims"]


T = TypeVar("T")


def eye(x: T) -> T:
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
    return pair[0]


def second(pair: Tuple[T, Any]) -> T:
    return pair[1]


def cf2cd(key: Tuple[chunk, feature]) -> Tuple[chunk, dimension]:
    return key[0], key[1].dim


@overload
def expand_dim(x: str, prefix: str) -> str:
    ...

@overload
def expand_dim(x: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    ...

@overload
def expand_dim(x: List[str], prefix: str) -> List[str]:
    ...

@overload
def expand_dim(x: Tuple[str, ...], prefix: str) -> Tuple[str, ...]:
    ...

def expand_dim(x, prefix):
    if isinstance(x, str):
        return uris.FSEP.join([prefix, x]).strip(uris.FSEP)
    elif isinstance(x, dict):
        return {uris.FSEP.join([prefix, k]).strip(uris.FSEP): v 
            for k, v in x.items()}
    elif isinstance(x, list):
        return list(uris.FSEP.join([prefix, k]).strip(uris.FSEP) for k in x)
    else:
        assert isinstance(x, tuple)
        return tuple(uris.FSEP.join([prefix, k]).strip(uris.FSEP) for k in x)


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


def collect_dims(features):
    dims = []
    for f in features:
        if f.d not in dims:
            dims.append(f.d)
    return dims
