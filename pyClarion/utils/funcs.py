import random
import math


__all__ = [
    "max_strength", "simple_junction", "max_junction", "linear_rule_strength", 
    "select", "boltzmann_distribution", "multiplicative_filter"
]


def max_strength(construct, packets):
    """
    Map construct to its maximum strength in packets.
    
    Assumes strength >= 0.
    """

    strength = 0
    for packet in packets:
        strength = max(packet.get(construct, 0), strength)
    return {construct: strength}


def simple_junction(packets):
    """Merge packets, assuming packet keys are disjoint."""

    d = {}
    for packet in packets:
        d.update(packet)
    return d


def max_junction(packets, min_val=0):
    """Map constructs in packets to their maximum strengths."""

    d = {}
    for packet in packets:
        for construct, strength in packet.items():
            d[construct] = max(d.get(construct, min_val), strength)
    return d


def linear_rule_strength(conditions, strengths, default=0):
    """Compute weighted sum of condition strengths."""

    return sum(w * strengths.get(ch, default) for ch, w in conditions.items())


def select(probabilities, k=1):
    """Sample k keys from probability dict without replacement."""

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

def multiplicative_filter(filter_weights, strengths, fdefault=0):

    d = {
        node: strengths[node] * (1 - filter_weights.get(node, fdefault)) 
        for node in strengths
    }
    return d
