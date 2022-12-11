from typing import TypeVar, Tuple, Any, Iterable, Callable, Dict

from .base.symbols import F, D, V, C


T = TypeVar("T")


def eye(x: T) -> T:
    """Return input x (identity function)."""
    return x


def first(pair: Tuple[T, Any]) -> T:
    """Return the first element in a pair."""
    return pair[0]


def second(pair: Tuple[Any, T]) -> T:
    """Return the second element in a pair."""
    return pair[1]


T1, T2 = TypeVar("T1"), TypeVar("T2")
def ab2ba(pair: Tuple[T1, T2]) -> Tuple[T2, T1]:
    a, b = pair
    return b, a


def cf2cd(key: Tuple[C, V]) -> Tuple[C, D]:
    """Convert a chunk-feature pair to a chunk-dimension pair."""
    return key[0], key[1].dim


def group_by(iterable: Iterable, key: Callable) -> Dict[Any, Tuple]:
    """Return a dict grouping items in iterable by values of the key func."""
    groups: dict = {}
    for item in iterable:
        k = key(item)
        groups.setdefault(k, []).append(item)
    return {k: tuple(v) for k, v in groups.items()}


def group_by_dims(features: Iterable[F]) -> Dict[D, Tuple[F, ...]]:
    """
    Construct a dict grouping features by their dimensions.
    
    Returns a dict where each dim is mapped to a tuple of features of that dim.
    Does not check for duplicate features.

    :param features: An iterable of features to be grouped by dimension.
    """
    return group_by(iterable=features, key=F.dim.fget)
