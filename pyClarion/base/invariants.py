from typing import Any
from pyClarion.base.enums import FlowType
from pyClarion.base.symbols import Node, Microfeature, Chunk, Flow, Appraisal


def subsystem_may_contain(key: Any) -> bool:
    
    value = (
        isinstance(key, Node) or
        isinstance(key, Flow) or
        isinstance(key, Appraisal)
    )
    return value


def subsystem_may_connect(source: Any, target: Any) -> bool:
    
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
        )
    ]
    return any(possibilities)