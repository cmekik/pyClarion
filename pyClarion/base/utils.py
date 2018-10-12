from typing import Iterable, Set, Type, Any
from pyClarion.base.enums import FlowType
from pyClarion.base.symbols import (
    ConstructSymbol, Node, Microfeature, Chunk, Flow, Appraisal, Buffer, 
    Subsystem, Agent, Behavior
)


def get_nodes(*node_iterables: Iterable[Node]) -> Set[Node]:
    """
    Construct the set of all nodes in a set of node containers.

    :param node_iterables: A sequence of iterables containing nodes.
    """

    node_set = set()
    for node_iterable in node_iterables:
        for node in node_iterable:
            node_set.add(node)
    return node_set


def check_construct(construct: ConstructSymbol, type_: Type):
    """Check if construct matches given type."""

    if not isinstance(construct, type_):
        raise TypeError("Unexpected construct type {}".format(str(construct)))


def may_contain(container: Any, element: Any) -> bool:
    """Return true if container may contain element."""
    
    possibilities = [
        (
            isinstance(container, Subsystem) and
            (
                isinstance(element, Node) or
                isinstance(element, Flow) or
                isinstance(element, Appraisal) or
                isinstance(element, Behavior)
            )
        ),
        (
            isinstance(container, Agent) and
            (
                isinstance(element, Subsystem) or
                isinstance(element, Buffer)
            )
        )
    ]
    return any(possibilities)


def may_connect(source: Any, target: Any) -> bool:
    """Return true if source may send output to target."""
    
    possibilities = [
        isinstance(source, Node) and isinstance(target, Appraisal),
        (
            isinstance(source, Microfeature) and 
            isinstance(target, Flow) and
            (
                target.flow_type == FlowType.Bot2Top or
                target.flow_type == FlowType.Bot2Bot
            )
        ),
        (
            isinstance(source, Chunk) and 
            isinstance(target, Flow) and
            (
                target.flow_type == FlowType.Top2Bot or
                target.flow_type == FlowType.Top2Top
            )
        ),
        (
            isinstance(source, Flow) and
            isinstance(target, Microfeature) and
            (
                source.flow_type == FlowType.Top2Bot or 
                source.flow_type == FlowType.Bot2Bot
            )
        ),
        (
            isinstance(source, Flow) and
            isinstance(target, Chunk) and
            (
                source.flow_type == FlowType.Bot2Top or 
                source.flow_type == FlowType.Top2Top
            )
        ),
        (
            isinstance(source, Appraisal) and 
            isinstance(target, Behavior)
        ),
        (
            isinstance(source, Buffer) and
            isinstance(target, Subsystem)
        )
    ]
    return any(possibilities)