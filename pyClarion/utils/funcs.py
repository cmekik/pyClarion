
from ..base.symbols import feature
import random
import math

from typing import Iterable, Dict, Hashable, Tuple
from itertools import groupby
import logging


__all__ = ["collect_cmd_data"]


def group_by_dims(
    features: Iterable[feature]
) -> Dict[Hashable, Tuple[feature, ...]]:
    """
    Construct a dict grouping features by their dimensions.
    
    Returns a dict where each dim is mapped to a tuple of features of that dim.
    Does not check for duplicate features.

    :param features: An iterable of features to be grouped by dimension.
    """

    groups = {}
    # Ignore type of key due to mypy false alarm. - Can
    key = feature.dim.fget # type: ignore 
    s = sorted(features, key=key)
    for k, g in groupby(s, key):
        groups[k] = tuple(g)
    
    return groups


def collect_cmd_data(construct, inputs, controller):

    subsystem, terminus = controller
    try:
        data = inputs[subsystem][terminus]
    except KeyError:
        data = frozenset()
        msg = "Failed data pull from %s in %s."
        logging.warning(msg, controller, construct)
    
    return data
