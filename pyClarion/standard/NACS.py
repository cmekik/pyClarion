'''
Implementation of the non-action-centered subsystem in standard Clarion.
'''


from typing import Dict, Hashable, Set, Tuple, List
from pyClarion.base.knowledge import Node, Chunk, Microfeature, Flow, Plicity
from pyClarion.standard.common import (
    ActivationPacket, TopLevelPacket, TopDownPacket, BottomUpPacket, Channel, UpdateJunction
)
from pyClarion.base.structure import FlowStructure
from pyClarion.base.component import NodeComponent, FlowComponent
from pyClarion.base.agent import Subsystem


###########################
### TOPLEVEL CONSTRUCTS ###
###########################


class AssociativeRuleChannel(Channel):

    def __init__(self, associations : Dict[Chunk, Dict[Chunk, float]]) -> None:
        
        self.associations = associations

    def __call__(self, input_map : ActivationPacket) -> TopLevelPacket:

        output = TopLevelPacket()
        for conclusion_chunk, weight_map in self.associations.items():
            for condition_chunk, weight in weight_map.items():
                output[conclusion_chunk] += (
                    weight * input_map[condition_chunk]
                )
        return output


class GKS(FlowComponent):
    
    def __init__(self) -> None:

        flow_name = 'GKS'
        self.flow = Flow(flow_name, Plicity.Explicit) 
        self.associations : Dict[Chunk, Dict[Chunk, float]] = dict()

    def update_knowledge(self, *args, **kwargs):
        pass

    def initialize_knowledge(self) -> List[FlowStructure]:

        return [
            FlowStructure(
                self.flow, 
                UpdateJunction(),
                AssociativeRuleChannel(self.associations)
            )
        ]

    def get_known_nodes(self) -> Set[Node]:

        output: Set[Node] = set(self.associations.keys())
        for mapping in self.associations.values():
            output.update(mapping.keys())
        return output


#############################
### INTERLEVEL CONSTRUCTS ###
#############################

    
class _InterLevelChannel(Channel):

    def __init__(
        self, 
        links : Dict[Chunk, Set[Microfeature]],
        weights : Dict[Chunk, Dict[Hashable, float]]
    ) -> None:

        self.links = links
        self.weights = weights


class TopDownChannel(_InterLevelChannel):

    def __call__(self, input_map : ActivationPacket) -> TopDownPacket:
        
        output = TopDownPacket()
        for nd, strength in input_map.items():
            if nd in self.links:
                for mf in self.links[nd]:
                    val = self.weights[nd][mf.dim] * strength
                    if output[mf] < val:
                        output[mf] = val
        return output


class BottomUpChannel(_InterLevelChannel):

    def __call__(self, input_map : ActivationPacket) -> BottomUpPacket:

        output = BottomUpPacket()
        for nd in self.links:
            dim_activations : Dict[Hashable, float] = dict()
            for mf in self.links[nd]:
                if (
                    (mf.dim not in dim_activations) or
                    (dim_activations[mf.dim] < input_map[mf])
                ):
                    dim_activations[mf.dim] = input_map[mf]
            for dim, strength in dim_activations.items():
                output[nd] += self.weights[nd][dim] * strength
        return output


class InterLevelComponent(FlowComponent):

    def __init__(
        self, 
        links : Dict[Chunk, Set[Microfeature]], 
        weights : Dict[Chunk, Dict[Hashable, float]]
    ) -> None:
        '''
        Initialize an InterLevelComponent.

        Must have:
            for every chunk, microfeature if ``microfeature in links[chunk]``
            then ``microfeature.dim in weights[chunk] and
            ``weights[chunk][microfeature.dim] != 0
        '''

        flow_name = 'Interlevel'
        self.flows = {
            Plicity.Abplicit: Flow(flow_name, Plicity.Abplicit),
            Plicity.Deplicit: Flow(flow_name, Plicity.Deplicit)
        }
        self.links = links
        self.weights = weights

    def update_knowledge(self, *args, **kwargs):
        pass

    def initialize_knowledge(self) -> List[FlowStructure]:

        return [
            FlowStructure(
                self.flows[Plicity.Abplicit], 
                UpdateJunction(),
                TopDownChannel(self.links, self.weights)
            ),
            FlowStructure(
                self.flows[Plicity.Deplicit],
                UpdateJunction(),
                BottomUpChannel(self.links, self.weights)
            )
        ]


############################
### SUBSYSTEM DEFINITION ###
############################


class NACS(Subsystem):
    """Ensures smooth functioning of components."""

    def __init__(self, external_inputs, external_outputs, actuator_structure, node_component, *components):

        self._selector = selector
        self._effector = effector
        self._node_component = node_component
        self._flow_components = set(components)
        self._network = ActuatorNetwork(
            external_inputs, actuator_structure
        )

    def init_links(self):
        """Link up and sync components and network at initialization time."""

        initial_nodes = set()
        for component in self.components:
            component.attach_to_network(self.network)
            component.add_knowledge_to_network()
            initial_nodes.update(component.get_known_nodes())
        self.node_component.attach_to_network(self.network)
        self.node_component.add_initial_nodes(initial_nodes)
        self.node_component.add_knowledge_to_network()

    @property
    def node_component(self):
        return self._node_component

    @property
    def flow_components(self):
        return self._flow_components

    @property
    def network(self):
        return self._network
