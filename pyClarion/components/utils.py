"""Miscellaneous utility functions."""


__all__ = [
    "group_by", "group_by_ctype", "group_by_dims", "group_by_tags", 
    "group_by_vals", "group_by_lags"
]


from ..base import ConstructType, Symbol, feature

from typing import Tuple, Dict, Hashable, Iterable, Callable, TypeVar


T = TypeVar("T")
K = TypeVar("K")


#########################
### GROUPING FUNCTIONS ##
#########################


def group_by(
    iterable: Iterable[T], key: Callable[[T], K]
) -> Dict[K, Tuple[T, ...]]:
    """Return a dict grouping items in iterable by values of the key func."""

    groups: dict = {}
    for item in iterable:
        k = key(item)
        groups.setdefault(k, []).append(item)
    
    return {k: tuple(v) for k, v in groups.items()}


def group_by_ctype(
    symbols: Iterable[Symbol]
) -> Dict[ConstructType, Tuple[Symbol, ...]]:
    """
    Construct a dict grouping symbols by their construct types.
    
    Returns a dict where each construct type is mapped to a tuple of symbols of 
    that type. Does not check for duplicates.
    """

    # Ignore type of key due to mypy false alarm. - Can
    key = Symbol.ctype.fget # type: ignore 
    
    return group_by(iterable=symbols, key=key)


def group_by_dims(
    features: Iterable[feature]
) -> Dict[Tuple[Hashable, int], Tuple[feature, ...]]:
    """
    Construct a dict grouping features by their dimensions.
    
    Returns a dict where each dim is mapped to a tuple of features of that dim.
    Does not check for duplicate features.

    :param features: An iterable of features to be grouped by dimension.
    """

    # Ignore type of key due to mypy false alarm. - Can
    key = feature.dim.fget # type: ignore 
    
    return group_by(iterable=features, key=key)


def group_by_tags(
    features: Iterable[feature]
) -> Dict[Hashable, Tuple[feature, ...]]:
    """
    Construct a dict grouping features by their dimensions.
    
    Returns a dict where each dim is mapped to a tuple of features of that dim.
    Does not check for duplicate features.

    :param features: An iterable of features to be grouped by dimension.
    """

    # Ignore type of key due to mypy false alarm. - Can
    key = feature.tag.fget # type: ignore 
    
    return group_by(iterable=features, key=key)


def group_by_vals(
    features: Iterable[feature]
) -> Dict[Hashable, Tuple[feature, ...]]:
    """
    Construct a dict grouping features by their values.
    
    Returns a dict where each value is mapped to a tuple of features that have 
    that value. Does not check for duplicate features.

    :param features: An iterable of features to be grouped by value.
    """

    # Ignore type of key due to mypy false alarm. - Can
    key = feature.val.fget # type: ignore 
    
    return group_by(iterable=features, key=key)


def group_by_lags(
    features: Iterable[feature]
) -> Dict[Hashable, Tuple[feature, ...]]:
    """
    Construct a dict grouping features by their lags.
    
    Returns a dict where each lag value is mapped to a tuple of features of 
    that lag value. Does not check for duplicate features.

    :param features: An iterable of features to be grouped by lag.
    """

    # Ignore type of key due to mypy false alarm. - Can
    key = feature.lag.fget # type: ignore 
    
    return group_by(iterable=features, key=key)

