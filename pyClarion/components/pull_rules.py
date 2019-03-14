from pyClarion.base.symbols import *
from pyClarion.base.realizers import *
from typing import List, cast


def node_pull_rule(node: ConstructRealizer, source: ConstructSymbol) -> bool:
    """Check if self may pull data from given construct."""

    possibilities: List[bool] = [
        source.ctype == ConstructType.Buffer,
        (
            source.ctype == ConstructType.Flow and
            node.csym.ctype == ConstructType.Feature and
            bool(
                cast(FlowID, source.cid).ftype & (FlowType.BB | FlowType.TB)
            )
        ),
        (
            source.ctype == ConstructType.Flow and
            node.csym.ctype == ConstructType.Chunk and
            bool(
                cast(FlowID, source.cid).ftype & (FlowType.TT | FlowType.BT)
            )
        )
    ] 

    return any(possibilities)


def flow_pull_rule(flow: ConstructRealizer, source: ConstructSymbol) -> bool:
    """Check if self may pull data from given construct."""

    possibilities: List[bool] = [
        (
            source.ctype == ConstructType.Feature and
            bool(
                cast(FlowID, flow.csym.cid).ftype & (FlowType.BB | FlowType.BT)
            )
        ),
        (
            source.ctype == ConstructType.Chunk and
            bool(
                cast(FlowID, flow.csym.cid).ftype & (FlowType.TT | FlowType.TB)
            )
        ),
    ]

    return any(possibilities)


def response_pull_rule(
    response: ConstructRealizer, source: ConstructSymbol
) -> bool:
    """Check if self may pull data from given construct."""

    possibilities: List[bool] = [
        bool(source.ctype & cast(ResponseID, response.csym.cid).itype)
    ]

    return any(possibilities)


def behavior_pull_rule(
    behavior: ConstructRealizer, source: ConstructSymbol
) -> bool:
    """Check if self may pull data from given construct."""

    possibilities: List[bool] = [
        source == cast(BehaviorID, behavior.csym.cid).response
    ]

    return any(possibilities)


def subsystem_pull_rule(
    subsystem: ConstructRealizer, source: ConstructSymbol
) -> bool:

    possibilities = [
        (
            source.ctype == ConstructType.Buffer and
            subsystem.csym in cast(BufferID, source.cid).outputs
        )
    ]

    return any(possibilities)
