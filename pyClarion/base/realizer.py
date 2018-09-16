import dataclasses
from typing import Union, TypeVar, Generic, MutableMapping
from pyClarion.base.symbols import (
    ConstructSymbol, BasicConstructSymbol, CompositeConstructSymbol, 
    Node, Flow, Appraisal, Activity, Memory, Subsystem, Agent
)
from pyClarion.base.processor import Channel, Junction, Selector, Effector


Ct = TypeVar('Ct', bound=ConstructSymbol)
Bt = TypeVar('Bt', bound=BasicConstructSymbol)
Xt = TypeVar('Xt', bound=CompositeConstructSymbol)


####################
### ABSTRATCIONS ###
####################


class Realizer(Generic[Ct], object):
    
    construct: Ct

    def __init__(self, construct: Ct) -> None:

        self.construct = construct


@dataclasses.dataclass()
class BasicConstructRealizer(Realizer[Bt]):

    construct: Bt


class CompositeConstructRealizer(MutableMapping[Ct, Realizer], Realizer[Xt]):
    pass


#################################
### BASIC CONSTRUCT REALIZERS ###
#################################


@dataclasses.dataclass()
class NodeRealizer(BasicConstructRealizer[Node]):

    junction: Junction        


@dataclasses.dataclass()
class FlowRealizer(BasicConstructRealizer[Flow]):

    junction: Junction
    channel: Channel


@dataclasses.dataclass()
class AppraisalRealizer(BasicConstructRealizer[Appraisal]):

    junction: Junction
    selector: Selector


@dataclasses.dataclass()
class ActivityRealizer(BasicConstructRealizer[Activity]):

    effector: Effector


@dataclasses.dataclass()
class MemoryRealizer(BasicConstructRealizer[Memory]):

    junction: Junction
    channel: Channel
