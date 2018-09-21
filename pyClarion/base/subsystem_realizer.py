"""
Tools for representing networks of chunks and microfeatures.

.. warning::
    Highly experimental.

"""

import abc
import enum
from typing import Dict, Union, Set, Tuple, Optional, Hashable, Callable
from pyClarion.base.knowledge import (
    Flow, Node, Chunk, Microfeature, Appraisal, Memory, Activity
)
from pyClarion.base.packet import ActivationPacket
from pyClarion.base.processor import Channel, Junction, Selector, Effector
from pyClarion.base.realizer import (
    NodeRealizer, FlowRealizer, AppraisalRealizer, ActivityRealizer
)
from pyClarion.base.connector import (
    Observer, Observable, NodePropagator, FlowPropagator, AppraisalPropagator, 
    ActivityDispatcher
)


class SubsystemRealizer(object):
    """A network of interconnected nodes and flows linked to an actuator."""

    def __init__(self, construct) -> None:

        self.construct = construct
        self._inputs: Dict[Memory, Callable[..., ActivationPacket]] = dict()
        self._nodes: Dict[Node, NodePropagator] = dict()
        self._flows: Dict[Flow, FlowPropagator] = dict()
        # Not sure if implementation should be dict here.
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
            self.add_node(key, value)
        elif isinstance(key, Flow):
            self.add_flow(key, value)
        elif isinstance(key, Appraisal):
            self._add_appraisal(key, value)
        elif isinstance(key, Activity):
            self._add_activity(key, value)
        else:
            raise TypeError("Unexpected type {}".format(type(key)))
        return out


    def add_node(self, node_realizer: NodeRealizer) -> None:
        
        node = node_realizer.construct
        node_connector = NodePropagator(node_realizer)
        self.nodes[node] = node_connector
        for memory, pull_method in self.inputs.items():
            node_connector.add_link(memory, pull_method)
        for flow, flow_connector in self.flows.items():
            flow_connector.add_link(node, node_connector.get_pull_method())
            node_connector.add_link(flow, flow_connector.get_pull_method())
        for appraisal, appraisal_connector in self.appraisal.items():
            appraisal_connector.add_link(node, node_connector.get_pull_method())

    def remove_node(self, node: Node) -> None:
        
        for appraisal, appraisal_connector in self.appraisal.items():
            appraisal_connector.drop_link(node)
        for flow_connector in self.flows.values():
            flow_connector.drop_link(node)
        del self.nodes[node]
            
    def add_flow(self, flow_realizer: FlowRealizer) -> None:

        flow = flow_realizer.construct
        flow_connector = FlowPropagator(flow_realizer)
        self.flows[flow] = flow_connector
        for node, node_connector in self.nodes.items():
            node_connector.add_link(flow, flow_connector.get_pull_method())
            flow_connector.add_link(node, node_connector.get_pull_method())

    def remove_flow(self, flow: Flow) -> None:
        
        try:
            for node_connector in self.nodes.values():
                node_connector.drop_link(flow)
            del self.flows[flow]
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
            # Print informate type error message.
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
            # Print informate type error message.
            raise TypeError()

    @property
    def inputs(self) -> Dict[Hashable, Callable[..., ActivationPacket]]:
        """External inputs to this network"""

        return self._inputs

    @property
    def nodes(self) -> Dict[Node, NodePropagator]:
        '''Nodes known to this network.'''
        
        return self._nodes

    @property
    def flows(self) -> Dict[Flow, FlowPropagator]:
        '''Activation flows defined for this network.'''

        return self._flows

    @property
    def appraisal(self) -> Dict[Appraisal, AppraisalPropagator]:
        '''Action selector for this network.'''

        return self._appraisal

    @property
    def activity(self) -> Dict[Activity, ActivityDispatcher]:

        return self._activity
