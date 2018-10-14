'''
Implementation of the non-action-centered subsystem in standard Clarion.
'''


from typing import Dict, Hashable, Set, Tuple, List, Sequence
from pyClarion.base.enums import FlowType, Level
from pyClarion.base.symbols import Node, Chunk, Microfeature, Flow
from pyClarion.base.packets import ActivationPacket
from pyClarion.base.processors import Channel
from pyClarion.base.realizers.subsystem import SubsystemRealizer
from pyClarion.standard.common import default_activation


AssociativeRuleSequence = (
    Sequence[
        Tuple[
            # Conclusion chunk
            Chunk, 
            # Condition chunks and corresponding weights
            Sequence[Tuple[Chunk, float]]
        ]
    ]
) 


class AssociativeRulesChannel(Channel[float]):

    def __init__(self, assoc: AssociativeRuleSequence) -> None:

        self.assoc = [[chunk, dict(weights)] for chunk, weights in assoc]

    def __call__(self, input_map):
        
        output = ActivationPacket(origin=Level.Top)
        for conclusion, conditions in self.assoc:
            strength = 0.
            for cond in conditions: 
                strength += (
                    conditions[cond] * 
                    input_map.get(cond, default_activation(cond))
                )
            try:
                activation = max(output[conclusion], strength)
            except KeyError:
                activation = strength
            output[conclusion] = activation
        return output


class TopDownChannel(Channel[float]):

    def __init__(self, assoc):

        self.assoc = assoc

    def __call__(self, input_map):

        output = ActivationPacket(origin=Level.Top)
        for node in input_map:
            if isinstance(node, Chunk) and node in self.assoc:
                mfs = self.assoc[node]["microfeatures"]
                weights = self.assoc[node]["weights"]
                for mf in mfs:
                    new_activation = weights[mf.dim] * input_map[node]
                    try:
                        activation = max(output[mf], new_activation)
                    except KeyError:
                        activation = new_activation
                    output[mf] = activation
        return output


class BottomUpChannel(Channel[float]):

    def __init__(self, assoc):

        self.assoc = assoc

    def __call__(self, input_map):

        output = ActivationPacket(origin=Level.Bot)
        for chunk in self.assoc:
            microfeatures = self.assoc[chunk]["microfeatures"]
            weights = self.assoc[chunk]["weights"]
            dim_activations = dict()
            for mf in microfeatures:
                dim_activations[mf.dim] = max(
                    dim_activations.get(mf.dim, default_activation(mf)),
                    input_map.get(mf, default_activation(mf))
                )
            output[chunk] = default_activation(chunk)
            for dim in dim_activations:
                output[chunk] += (
                    weights[dim] * dim_activations[dim]
                )
            output[chunk] /= len(weights) ** 1.1
        return output


class NACSRealizer(SubsystemRealizer):

    def propagate(self):

        # Update Chunks
        for node in self.nodes:
            if isinstance(node, Chunk):
                self[node].propagate()

        # Propagate Top-down Flows
        for flow in self.flows:
            if flow.flow_type == FlowType.Top2Bot:
                self[flow].propagate()
        
        # Update Microfeatures
        for node in self.nodes:
            if isinstance(node, Microfeature):
                self[node].propagate()
        
        # Simultaneously Process at Both Top and Bottom Levels
        for flow in self.flows:
            if flow.flow_type in (FlowType.Top2Top, FlowType.Bot2Bot):
                self[flow].propagate()
        
        # Update All Nodes
        for node in self.nodes:
            self[node].propagate()
        
        # Propagate Bottom-up Links
        for flow in self.flows:
            if flow.flow_type == FlowType.Bot2Top:
                self[flow].propagate()
        
        # Update Chunks
        for node in self.nodes:
            if isinstance(node, Chunk):
                self[node].propagate()
        
        # Update Appraisal
        self.appraisal.propagate()
