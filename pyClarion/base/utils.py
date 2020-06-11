"""Provides convenience functions."""


__all__ = [
    "feature", "chunk", "flow_bt", "flow_tb", "flow_tt", "flow_bb", 
    "response", "buffer", "subsystem", "agent"
]


from typing import Hashable 
from pyClarion.base.symbols import ConstructType, ConstructSymbol, FeatureSymbol
from pyClarion.base.packets import ActivationPacket
from pyClarion.base.propagators import PropagatorA
from pyClarion.base.realizers import MatchArg, UpdaterArg, Flow


##################################
### Construct Symbol Factories ###
##################################

# These are convenience functions for constructing construct symbols.
# They simply wrap the appropriate ConstructSymbol constructor.


def feature(dim: Hashable, val: Hashable) -> ConstructSymbol:
    """
    Return a new feature symbol.

    :param dim: Dimension of feature.
    :param val: Value of feature.
    """

    return FeatureSymbol(dim, val)


def chunk(name: Hashable) -> ConstructSymbol:
    """
    Return a new chunk symbol.

    :param cid: Chunk identifier.
    """

    return ConstructSymbol(ConstructType.chunk, name)


def flow_bt(name: Hashable) -> ConstructSymbol:
    """
    Return a new bottom-up flow symbol.

    :param cid: Name of flow.
    """

    return ConstructSymbol(ConstructType.flow_bt, name)


def flow_tb(name: Hashable) -> ConstructSymbol:
    """
    Return a new top-down flow symbol.

    :param cid: Name of flow.
    """

    return ConstructSymbol(ConstructType.flow_tb, name)


def flow_tt(name: Hashable) -> ConstructSymbol:
    """
    Return a new top-level flow symbol.

    :param cid: Name of flow.
    """

    return ConstructSymbol(ConstructType.flow_tt, name)


def flow_bb(name: Hashable) -> ConstructSymbol:
    """
    Return a new bottom-level flow symbol.

    :param cid: Name of flow.
    """

    return ConstructSymbol(ConstructType.flow_bb, name)


def response(name: Hashable) -> ConstructSymbol:
    """
    Return a new response symbol.

    :param name: Name of response.
    """

    return ConstructSymbol(ConstructType.response, name)


def buffer(name: Hashable) -> ConstructSymbol:
    """
    Return a new buffer symbol.

    :param name: Name of buffer.
    """

    return ConstructSymbol(ConstructType.buffer, name)


def subsystem(name: Hashable) -> ConstructSymbol:
    """
    Return a new subsystem symbol.

    :param name: Name of subsystem.
    """

    return ConstructSymbol(ConstructType.subsystem, name)


def agent(name: Hashable) -> ConstructSymbol:
    """
    Return a new agent symbol.

    :param name: Name of agent.
    """

    return ConstructSymbol(ConstructType.agent, name)
