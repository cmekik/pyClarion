'''
Implementation of the non-action-centered subsystem in standard Clarion.
'''


from typing import Dict, Hashable, Set, Tuple
from pyClarion.base.knowledge import Chunk, Microfeature
from pyClarion.standard.common import (
    ActivationPacket, TopLevelPacket, TopDownPacket, BottomUpPacket, Channel
)
from pyClarion.base.agent import Component, Subsystem


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


class GKS(Component):
    
    def __init__(self) -> None:

        self.associations : Dict[Chunk, Dict[Chunk, float]] = dict()

    def spawn_channels(self) -> AssociativeRuleChannel:

        return AssociativeRuleChannel(self.associations)


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


class InterLevelComponent(Component):

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

        self.links = links
        self.weights = weights

    def spawn_channels(self) -> Tuple[TopDownChannel, BottomUpChannel]:

        return (
            TopDownChannel(self.links, self.weights), 
            BottomUpChannel(self.links, self.weights)
        )


############################
### SUBSYSTEM DEFINITION ###
############################

class NACS(Subsystem):
    pass
