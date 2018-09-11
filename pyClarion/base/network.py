"""
Tools for representing networks of chunks and microfeatures.

.. warning::
    Highly experimental.

"""

import abc
import enum
from typing import Dict, Union, Set, Tuple, Optional
from pyClarion.base.knowledge import Flow, Node, Chunk, Microfeature
from pyClarion.base.channel import Channel
from pyClarion.base.junction import Junction
from pyClarion.base.selector import Selector
from pyClarion.base.effector import Effector
from pyClarion.base.connector import NodeConnector, FlowConnector, Actuator


class ActuatorNetwork(abc.ABC):
    """A network of interconnected nodes and flows linked to an actuator."""

    def __init__(self,
        selector: Selector, 
        effector: Effector,
        selector_junction: Junction, 
    ) -> None:

        self._nodes: Dict[Node, NodeConnector] = dict()
        self._flows: Dict[Flow, FlowConnector] = dict()
        self._selector: Actuator = Actuator(
            selector, effector, selector_junction
        ) 

    def add_node(self, node: Node, junction: Junction) -> None:
        
        node_connector = NodeConnector(node, junction)
        self.nodes[node] = node_connector
        for flow, flow_connector in self.flows.items():
            flow_connector.add_link(node, node_connector.get_pull_method())
            node_connector.add_link(flow, flow_connector.get_pull_method())
        self.selector.add_link(node, node_connector.get_pull_method())

    def remove_node(self, node: Node) -> None:
        
        self.selector.drop_link(node)
        for flow_connector in self.flows.values():
            flow_connector.drop_link(node)
        del self.nodes[node]
            
    def add_flow(self, flow: Flow, channel: Channel, junction: Junction) -> None:

        flow_connector = FlowConnector(flow, channel, junction)
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

    def run_activation_cycle(self) -> None:

        # Initialization
        for node_connector in self.nodes.values():
            node_connector()
        
        # Cycle body
        chosen: Optional[Set[Chunk]] = None
        while not chosen:
            for flow_connector in self.flows.values():
                flow_connector()
            for node_connector in self.nodes.values():
                node_connector()            
            self.selector()
            chosen = self.selector.get_output().chosen


    @property
    def nodes(self) -> Dict[Node, NodeConnector]:
        '''Nodes known to this network.'''
        
        return self._nodes

    @property
    def flows(self) -> Dict[Flow, FlowConnector]:
        '''Activation flows defined for this network.'''

        return self._flows

    @property
    def selector(self) -> Actuator:
        '''Action selector for this network.'''

        return self._selector