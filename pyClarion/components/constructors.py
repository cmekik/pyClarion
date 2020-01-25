"""
Convenience functions for constructing construct symbols and realizers.
"""

from typing import Hashable 
from pyClarion.base import ActivationPacket, MatchArg, UpdaterArg, Proc, Node, Flow
from pyClarion.base.symbols import *


####################################
### Construct Realizer Factories ###
####################################


def Feature(
    dim: Hashable,
    val: Hashable, 
    matches: MatchArg = None,
    proc: Proc[ActivationPacket, ActivationPacket] = None,
    updaters: UpdaterArg[Node] = None,
) -> Node:

    construct = feature(dim=dim, val=val)
    obj = Node(name=construct, matches=matches, proc=proc, updaters=updaters)
    return obj

def Chunk(
    name: Hashable,
    matches: MatchArg = None,
    proc: Proc[ActivationPacket, ActivationPacket] = None,
    updaters: UpdaterArg[Node] = None,
) -> Node:

    construct = chunk(name=name)
    obj = Node(name=construct, matches=matches, proc=proc, updaters=updaters)
    return obj

def _construct_ftype(
    name: Hashable,
    ftype: ConstructType,  
    matches: MatchArg = None, 
    proc: Proc[ActivationPacket, ActivationPacket] = None, 
    updaters: UpdaterArg[Flow] = None
) -> Flow:

    name = ConstructSymbol(ftype, name)
    return Flow(name=name, matches=matches, proc=proc, updaters=updaters)

def FlowTT(
    name: Hashable,
    matches: MatchArg = None, 
    proc: Proc[ActivationPacket, ActivationPacket] = None, 
    updaters: UpdaterArg[Flow] = None
) -> Flow:

    return _construct_ftype(
        name=name, 
        ftype=ConstructType.flow_tt, 
        matches=matches, 
        proc=proc, 
        updaters=updaters
    )

def FlowBB(
    name: Hashable,
    matches: MatchArg = None, 
    proc: Proc[ActivationPacket, ActivationPacket] = None, 
    updaters: UpdaterArg[Flow] = None
) -> Flow:

    return _construct_ftype(
        name=name, 
        ftype=ConstructType.flow_bb, 
        matches=matches, 
        proc=proc, 
        updaters=updaters
    )

def FlowTB(
    name: Hashable,
    matches: MatchArg = None, 
    proc: Proc[ActivationPacket, ActivationPacket] = None, 
    updaters: UpdaterArg[Flow] = None
) -> Flow:

    return _construct_ftype(
        name=name, 
        ftype=ConstructType.flow_tb, 
        matches=matches, 
        proc=proc, 
        updaters=updaters
    )

def FlowBT(
    name: Hashable,
    matches: MatchArg = None, 
    proc: Proc[ActivationPacket, ActivationPacket] = None, 
    updaters: UpdaterArg[Flow] = None
) -> Flow:

    return _construct_ftype(
        name=name, 
        ftype=ConstructType.flow_bt,
        matches=matches, 
        proc=proc, 
        updaters=updaters
    )
