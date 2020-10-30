
from pyClarion.base.symbols import feature
import random
import math

from typing import Iterable, Dict, Hashable, Tuple
from itertools import groupby
import logging


__all__ = [
    "group_by_dims", "collect_cmd_data", "eye", "inv", "max_strength", 
    "invert_strengths", "simple_junction", "max_junction", 
    "linear_rule_strength", "select", "boltzmann_distribution", 
    "multiplicative_filter", "scale_strengths"
]


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


def eye(strength):
    """Return strength."""

    return strength


def inv(strength):
    """Return inverted strength (i.e., 1 - strength)."""

    return 1.0 - strength


def max_strength(construct, packets):
    """
    Map construct to its maximum strength in packets.
    
    Assumes strength >= 0.
    """

    strength = 0.0
    for packet in packets:
        strength = max(packet.get(construct, 0.0), strength)
    return strength


def invert_strengths(strengths):

    inverted = {c: 1.0 - s for c, s in strengths.items()}
    return inverted


def simple_junction(packets):
    """Merge packets, assuming packet keys are disjoint."""

    d = {}
    for packet in packets:
        d.update(packet)
    return d


def max_junction(packets, min_val=0.0):
    """Map constructs in packets to their maximum strengths."""

    d = {}
    for packet in packets:
        for construct, strength in packet.items():
            d[construct] = max(d.get(construct, min_val), strength)
    return d


def linear_rule_strength(conditions, strengths, default=0.0):
    """Compute weighted sum of condition strengths."""

    return sum(w * strengths.get(ch, default) for ch, w in conditions.items())


def select(probabilities, k=1):
    """
    Sample k keys from probability dict without replacement.
    
    If probabilities is empty returns empty set.
    """

    selection = set()
    if len(probabilities) > 0:
        cs, ws = tuple(zip(*probabilities.items()))
        while len(selection) < k:
            selection.update(random.choices(cs, weights=ws))
    return selection


def boltzmann_distribution(strengths, temperature):
    """Construct and return a boltzmann distribution."""

    terms, divisor = {}, 0
    for construct, s in strengths.items():
        terms[construct] = math.exp(s / temperature)
        divisor += terms[construct]
    probabilities = {c: s / divisor for c, s in terms.items()}
    return probabilities


def multiplicative_filter(weights, strengths, fdefault=0.0):

    d = {
        node: strengths[node] * weights.get(node, fdefault) 
        for node in strengths
    }
    return d


def scale_strengths(weight, strengths):

    scaled_strengths = {
        construct: weight * strength 
        for construct, strength in strengths.items()
    }
    return scaled_strengths
