'''
Implementation of the non-action-centered subsystem in standard Clarion.
'''


from typing import Dict, Hashable, Set, Tuple, List
from pyClarion.base.symbol import Node, Chunk, Microfeature, Flow, FlowType
from pyClarion.base.packets import Level
from pyClarion.base.processors import Channel
from pyClarion.standard.common import (
    ActivationPacket, UpdateJunction, default_factory
)
from pyClarion.base.realizer import FlowRealizer, SubsystemRealizer
from pyClarion.base.administrator import FlowAdministrator


###########################
### TOPLEVEL CONSTRUCTS ###
###########################


class AssociativeRules(Channel[float]):

    def __init__(self, associations : Dict[Chunk, Dict[Chunk, float]]) -> None:
        
        self.associations = associations

    def __call__(self, input_map : ActivationPacket[float]) -> ActivationPacket[float]:

        output = ActivationPacket(
            default_factory=default_factory, origin=Level.TopLevel
        )
        for conclusion_chunk, weight_map in self.associations.items():
            for condition_chunk, weight in weight_map.items():
                output[conclusion_chunk] += (
                    weight * input_map[condition_chunk]
                )
        return output


class GKSAdministrator(FlowAdministrator):
    
    def __init__(self) -> None:

        flow_name = 'GKS'
        self.flow = Flow(flow_name, FlowType.TopLevel) 
        self.associations : Dict[Chunk, Dict[Chunk, float]] = dict()

    def update_knowledge(self, *args, **kwargs):
        pass

    def initialize_knowledge(self) -> List[FlowRealizer]:

        return [
            FlowRealizer(
                self.flow, 
                UpdateJunction(),
                AssociativeRules(self.associations)
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

    def __call__(self, input_map : ActivationPacket) -> ActivationPacket:
        
        output = ActivationPacket(flow_type=FlowType.TopDown)
        for nd, strength in input_map.items():
            if nd in self.links:
                for mf in self.links[nd]:
                    val = self.weights[nd][mf.dim] * strength
                    if output[mf] < val:
                        output[mf] = val
        return output


class BottomUpChannel(_InterLevelChannel):

    def __call__(self, input_map : ActivationPacket) -> ActivationPacket:

        output = ActivationPacket(flow_type=FlowType.BottomUp)
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


class InterLevelFlowAdministrator(FlowAdministrator):

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
            Flow(flow_name, FlowType.TopDown), 
            Flow(flow_name, FlowType.BottomUp)
        }
        self.links = links
        self.weights = weights

    def update_knowledge(self, *args, **kwargs):
        pass

    def initialize_knowledge(self) -> List[FlowRealizer]:

        return [
            FlowRealizer(
                Flow("Interlevel", FlowType.TopDown), 
                UpdateJunction(),
                TopDownChannel(self.links, self.weights)
            ),
            FlowRealizer(
                Flow("Interlevel", FlowType.BottomUp),
                UpdateJunction(),
                BottomUpChannel(self.links, self.weights)
            )
        ]


class NACSRealizer(SubsystemRealizer):

    def __call__(self):

        for node, node_propagator in self._nodes.items():
            if isinstance(node, Chunk):
                node_propagator()
        for flow, flow_propagator in self._flows.values():
            if flow.flow_type == FlowType.TopDown:
                flow_propagator()
        for node, node_propagator in self._nodes.items():
            if isinstance(node, Microfeature):
                node_propagator()
        for flow, flow_propagator in self._flows.values():
            if flow.flow_type in (FlowType.TopLevel, FlowType.BottomLevel):
                flow_propagator()
        for node, node_propagator in self._nodes.items():
            node_propagator()
        for flow, flow_propagator in self._flows.values():
            if flow.flow_type == FlowType.BottomUp:
                flow_propagator()
        for node, node_propagator in self._nodes.items():
            if isinstance(node, Chunk):
                node_propagator()
        for appraisal, appraisal_propagator in self._appraisal.items():
            appraisal_propagator()
        for activity, activity_dispatcher in self._activity.items():
            activity_dispatcher()
