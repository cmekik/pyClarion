"""
Tools for representing knowledge and action structures.

Structures generally completely specify the behavior of some element when 
embedded in a Clarion network.

Knowledge structures completely specify the behavior of a knowledge construct 
when embedded in a Clarion network.

Actuator structures completely specify the behavior of an actuator when embedded 
in a Clarion network.
"""

import dataclasses
from typing import Union
from pyClarion.base.knowledge import Node, Flow, Appraisal
from pyClarion.base.processor import Channel, Junction, Selector, Effector

@dataclasses.dataclass()
class Structure(object):

    pass

@dataclasses.dataclass()
class KnowledgeStructure(Structure):

    construct: Union[Node, Flow, Appraisal]

@dataclasses.dataclass()
class NodeStructure(KnowledgeStructure):

    construct: Node
    junction: Junction        

@dataclasses.dataclass()
class FlowStructure(KnowledgeStructure):

    construct: Flow
    junction: Junction
    channel: Channel

@dataclasses.dataclass()
class AppraisalStructure(KnowledgeStructure):

    construct: Appraisal
    junction: Junction
    selector: Selector

@dataclasses.dataclass()
class ActuatorStructure(Structure):

    junction: Junction
    effector: Effector