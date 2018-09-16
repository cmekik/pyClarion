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
    NodeRealizer, FlowRealizer, AppraisalRealizer
)
from pyClarion.base.connector import (
    Observer, Observable, NodePropagator, FlowPropagator, AppraisalPropagator, 
    ActivityDispatcher
)


class ActivationNetwork(object):
    """A network of interconnected nodes and flows linked to an actuator."""

    def __init__(self) -> None:

        self._inputs: Dict[Memory, Callable[..., ActivationPacket]] = dict()
        self._nodes: Dict[Node, NodePropagator] = dict()
        self._flows: Dict[Flow, FlowPropagator] = dict()
        self._appraisal: Dict[Appraisal, AppraisalPropagator] = dict()
        self._activity: Dict[Activity, ActivityDispatcher] = dict()

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
