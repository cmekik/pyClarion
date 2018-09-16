"""
Tools for constructing Clarion components.

.. warning::
   Module experimental.

"""

import abc
from typing import Union, List, Tuple, Generic, TypeVar, Callable, Optional, Set
from pyClarion.base.knowledge import Node, Flow
from pyClarion.base.processor import Channel, Junction
from pyClarion.base.realizer import (
    BasicConstructRealizer, NodeRealizer, FlowRealizer
)
from pyClarion.base.network import ActivationNetwork


Ct = TypeVar('Ct', bound=BasicConstructRealizer)


class ConstructAdministrator(Generic[Ct], abc.ABC):
    """
    Manages some class of realizers.

    Administrators are intended to be objects implementing learning and 
    forgetting routines. They monitor the activity of the subsystem to which 
    they belong and modify its members (channels, junctions and/or parameters).
    """

    @abc.abstractmethod
    def update_knowledge(self, *args, **kwargs) -> None:
        """
        Updates knowledge given result of current activation cycle.

        The API for this is under development. A more specific signature is 
        forthcoming.
        """
        pass
    
    @abc.abstractmethod
    def initialize_knowledge(self) -> List[Ct]:
        '''Create and return channel(s) managed by ``self``.'''

        pass

    @abc.abstractmethod
    def attach_to_network(self, network: ActivationNetwork) -> None:
        pass

    @abc.abstractmethod
    def get_known_nodes(self) -> Set[Node]:
        pass


class NodeAdministrator(ConstructAdministrator[NodeRealizer]):

    node_adder: Optional[Callable] = None
    node_remover: Optional[Callable] = None
    
    def attach_to_network(self, network: ActivationNetwork) -> None:

        self.node_adder = network.add_node
        self.node_remover = network.remove_node


class FlowAdministrator(ConstructAdministrator[FlowRealizer]):

    flow_adder: Optional[Callable] = None
    flow_remover: Optional[Callable] = None
    
    def attach_to_network(self, network: ActivationNetwork) -> None:

        self.flow_adder = network.add_flow
        self.flow_remover = network.remove_flow
