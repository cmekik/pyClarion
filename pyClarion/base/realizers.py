"""
Tools for defining the behavior of theoretcally relevant constructs.
"""


import dataclasses
from typing import Union, TypeVar, Generic, MutableMapping, Dict, Callable
from pyClarion.base.symbols import (
    ConstructSymbol, BasicConstructSymbol, CompositeConstructSymbol, 
    Node, Flow, Appraisal, Activity, Memory, Subsystem, Agent
)
from pyClarion.base.packet import ActivationPacket
from pyClarion.base.processor import Channel, Junction, Selector, Effector
from pyClarion.base.link import (
    Observer, Observable, NodePropagator, FlowPropagator, AppraisalPropagator, 
    ActivityDispatcher
)


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


class CompositeConstructRealizer(
    MutableMapping[ConstructSymbol, Realizer], Realizer[Xt]
):
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


#####################################
### COMPOSITE CONSTRUCT REALIZERS ###
#####################################

class SubsystemRealizer(CompositeConstructRealizer[Subsystem]):
    """A network of interconnected nodes and flows linked to an actuator."""

    def __init__(self, construct) -> None:

        self.construct = construct
        self._inputs: Dict[Memory, Callable[..., ActivationPacket]] = dict()
        self._nodes: Dict[Node, NodePropagator] = dict()
        self._flows: Dict[Flow, FlowPropagator] = dict()
        self._appraisal: Dict[Appraisal, AppraisalPropagator] = dict()
        self._activity: Dict[Activity, ActivityDispatcher] = dict()

    def __getitem__(self, key):

        if isinstance(key, Node):
            out = self.nodes[key]
        elif isinstance(key, Flow):
            out = self.flows[key]
        elif isinstance(key, Appraisal):
            out = self._appraisal[key]
        elif isinstance(key, Activity):
            out = self._activity[key]
        else:
            raise TypeError("Unexpected type {}".format(type(key)))
        return out

    def __setitem__(self, key, value):

        if isinstance(key, Node):
            self._add_node(key, value)
        elif isinstance(key, Flow):
            self._add_flow(key, value)
        elif isinstance(key, Appraisal):
            self._add_appraisal(key, value)
        elif isinstance(key, Activity):
            self._add_activity(key, value)
        else:
            raise TypeError("Unexpected type {}".format(type(key)))

    def __delitem__(self, key, value):

        if isinstance(key, Node):
            self._remove_node(key)
        elif isinstance(key, Flow):
            self._remove_flow(key)
        elif isinstance(key, Appraisal):
            self._remove_appraisal(key)
        elif isinstance(key, Activity):
            self._remove_activity(key)
        else:
            raise TypeError("Unexpected type {}".format(type(key)))

    def _add_node(self, node_realizer: NodeRealizer) -> None:
        
        node = node_realizer.construct
        node_connector = NodePropagator(node_realizer)
        self._nodes[node] = node_connector
        for memory, pull_method in self.inputs.items():
            node_connector.watch(memory, pull_method)
        for flow, flow_connector in self._flows.items():
            flow_connector.watch(node, node_connector.get_pull_method())
            node_connector.watch(flow, flow_connector.get_pull_method())
        for appraisal, appraisal_connector in self._appraisal.items():
            appraisal_connector.watch(node, node_connector.get_pull_method())

    def _remove_node(self, node: Node) -> None:
        
        for appraisal, appraisal_connector in self._appraisal.items():
            appraisal_connector.drop(node)
        for flow_connector in self._flows.values():
            flow_connector.drop(node)
        del self._nodes[node]
            
    def _add_flow(self, flow_realizer: FlowRealizer) -> None:

        flow = flow_realizer.construct
        flow_connector = FlowPropagator(flow_realizer)
        self._flows[flow] = flow_connector
        for node, node_connector in self._nodes.items():
            node_connector.watch(flow, flow_connector.get_pull_method())
            flow_connector.watch(node, node_connector.get_pull_method())

    def _remove_flow(self, flow: Flow) -> None:
        
        try:
            for node_connector in self._nodes.values():
                node_connector.drop(flow)
            del self._flows[flow]
        except KeyError:
            pass

    def _add_appraisal(
        self, appraisal: Appraisal, appraisal_realizer: AppraisalRealizer
    ):

        if self._appraisal:
            raise Exception("Appraisal already set")
        elif (
            isinstance(appraisal, Appraisal) and 
            isinstance(appraisal_realizer, AppraisalRealizer)
        ):
            self._appraisal[appraisal] = AppraisalPropagator(appraisal_realizer) 
        else:
            # Print informative type error message.
            raise TypeError()

    def _add_activity(
        self, activity: Activity, activity_realizer: ActivityRealizer
    ):

        if self._activity:
            raise Exception("Activity already set")
        elif (
            isinstance(activity, Activity) and 
            isinstance(activity_realizer, ActivityRealizer)
        ):
            self._activity[activity] = ActivityDispatcher(activity_realizer) 
        else:
            # Print informative type error message.
            raise TypeError()

    @property
    def inputs(self) -> Dict[Memory, Callable[..., ActivationPacket]]:
        """External inputs to this network"""

        return self._inputs