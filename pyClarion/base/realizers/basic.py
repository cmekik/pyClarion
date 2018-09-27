from dataclasses import dataclass
from pyClarion.base.symbols import Node, Flow, Appraisal, Activity, Memory
from pyClarion.base.processors import Channel, Junction, Selector, Buffer, Effector
from pyClarion.base.realizers.abstract import BasicConstructRealizer


#################################
### BASIC CONSTRUCT REALIZERS ###
#################################


@dataclass()
class NodeRealizer(BasicConstructRealizer[Node]):

    construct: Node
    junction: Junction        


@dataclass()
class FlowRealizer(BasicConstructRealizer[Flow]):

    construct: Flow
    junction: Junction
    channel: Channel


@dataclass()
class AppraisalRealizer(BasicConstructRealizer[Appraisal]):

    construct: Appraisal
    junction: Junction
    selector: Selector


@dataclass()
class ActivityRealizer(BasicConstructRealizer[Activity]):

    construct: Activity
    effector: Effector
 

@dataclass()
class MemoryRealizer(BasicConstructRealizer[Memory]):

    construct: Memory
    buffer: Buffer
 