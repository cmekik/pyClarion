'''
Implementation of the non-action-centered subsystem in standard Clarion.
'''


from typing import Dict, Hashable, Set, Tuple, List, Sequence, Any
from pyClarion.base import *


AssociativeRuleSequence = (
    List[
        Tuple[
            # Conclusion chunk
            Chunk, 
            # Condition chunks and corresponding weights
            Dict[Chunk, float]
        ]
    ]
) 


InterlevelAssociation =(
    Dict[
        Chunk,
        Tuple[
            # Dimensional weights
            Dict[Hashable, float],
            # Microfeatures 
            Set[Microfeature]
        ]
    ]
)


class AssociativeRulesChannel(Channel[float]):

    def __init__(
        self, 
        assoc: AssociativeRuleSequence, 
        default_activation: DefaultActivation
    ) -> None:

        self.assoc = assoc
        self.default_activation = default_activation

    def __call__(self, input_map):
        
        output = ActivationPacket()
        for conclusion, conditions in self.assoc:
            strength = 0.
            for cond in conditions: 
                strength += (
                    conditions[cond] * 
                    input_map.get(cond, self.default_activation(cond))
                )
            try:
                activation = max(output[conclusion], strength)
            except KeyError:
                activation = strength
            output[conclusion] = activation
        return output


class TopDownChannel(Channel[float]):

    def __init__(self, assoc: InterlevelAssociation) -> None:

        self.assoc = assoc

    def __call__(self, input_map):

        output = ActivationPacket()
        for node in input_map:
            if isinstance(node, Chunk) and node in self.assoc:
                weights, mfs = self.assoc[node]
                for mf in mfs:
                    new_activation = weights[mf.dim] * input_map[node]
                    try:
                        activation = max(output[mf], new_activation)
                    except KeyError:
                        activation = new_activation
                    output[mf] = activation
        return output


class BottomUpChannel(Channel[float]):

    def __init__(self, assoc: InterlevelAssociation) -> None:

        self.assoc = assoc

    def __call__(self, input_map):

        output = ActivationPacket()
        for chunk in self.assoc:
            weights, microfeatures = self.assoc[chunk]
            dim_activations = dict()
            for mf in microfeatures:
                dim_activations[mf.dim] = max(
                    dim_activations.get(mf.dim, self.default_activation(mf)),
                    input_map.get(mf, self.default_activation(mf))
                )
            output[chunk] = self.default_activation()
            for dim in dim_activations:
                output[chunk] += (
                    weights[dim] * dim_activations[dim]
                )
            output[chunk] /= len(weights) ** 1.1
        return output

    @staticmethod
    def default_activation(node: Node = None):

        return 0.


def may_connect(source: Any, target: Any) -> bool:
    """Return true if source may send output to target."""
    
    possibilities = [
        isinstance(source, Node) and isinstance(target, Appraisal),
        (
            isinstance(source, Microfeature) and 
            isinstance(target, Flow) and
            (
                target.flow_type == FlowType.BT or
                target.flow_type == FlowType.BB
            )
        ),
        (
            isinstance(source, Chunk) and 
            isinstance(target, Flow) and
            (
                target.flow_type == FlowType.TB or
                target.flow_type == FlowType.TT
            )
        ),
        (
            isinstance(source, Flow) and
            isinstance(target, Microfeature) and
            (
                source.flow_type == FlowType.TB or 
                source.flow_type == FlowType.BB
            )
        ),
        (
            isinstance(source, Flow) and
            isinstance(target, Chunk) and
            (
                source.flow_type == FlowType.BT or 
                source.flow_type == FlowType.TT
            )
        ),
        (
            isinstance(source, Appraisal) and 
            isinstance(target, Behavior)
        ),
        (
            isinstance(source, Buffer) and
            isinstance(target, Node)
        )
    ]
    return any(possibilities)


def nacs_propagation_cycle(realizer: SubsystemRealizer) -> None:
    """Execute NACS activation propagation cycle on given realizer."""

    # Update Chunks
    for node in realizer.nodes:
        if isinstance(node, Chunk):
            realizer[node].propagate()

    # Propagate Top-down Flows
    for flow in realizer.flows:
        if flow.flow_type == FlowType.TB:
            realizer[flow].propagate()
    
    # Update Microfeatures
    for node in realizer.nodes:
        if isinstance(node, Microfeature):
            realizer[node].propagate()
    
    # Simultaneously Process at Both Top and Bottom Levels
    for flow in realizer.flows:
        if flow.flow_type in (FlowType.TT, FlowType.BB):
            realizer[flow].propagate()
    
    # Update All Nodes
    for node in realizer.nodes:
        realizer[node].propagate()
    
    # Propagate Bottom-up Links
    for flow in realizer.flows:
        if flow.flow_type == FlowType.BT:
            realizer[flow].propagate()
    
    # Update Chunks
    for node in realizer.nodes:
        if isinstance(node, Chunk):
            realizer[node].propagate()
    
    # Update Appraisal(s)
    for appraisal in realizer.appraisals:
        realizer[appraisal].propagate()
