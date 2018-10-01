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


class AssociativeRules(Channel[float]):

    def __init__(self, assoc: AssociativeRuleSequence) -> None:

        self.assoc = [[chunk, dict(weights)] for chunk, weights in assoc]

    def __call__(self, input_map):
        
        output = ActivationPacket(
            default_factory=default_activation, origin=Level.Top
        )
        for conclusion, conditions in self.assoc:
            strength = 0.
            for cond in conditions: 
                strength += (
                    conditions[cond] * 
                    input_map.get(cond, default_activation(cond))
                )
            output[conclusion] = max(output[conclusion], strength)
        return output


class TopDownChannel(Channel[float]):

    def __init__(self, assoc):

        self.assoc = assoc

    def __call__(self, input_map):

        output = ActivationPacket(
            default_factory=default_activation, origin=Level.Top
        )
        for node in input_map:
            if isinstance(node, Chunk) and node in self.assoc:
                mfs = self.assoc[node]["microfeatures"]
                weights = self.assoc[node]["weights"]
                for mf in mfs:
                    output[mf] = max(
                        output[mf],
                        weights[mf.dim] * input_map[node]
                    )
        return output


class BottomUpChannel(Channel[float]):

    def __init__(self, assoc):

        self.assoc = assoc

    def __call__(self, input_map):

        output = ActivationPacket(
            default_factory=default_activation, origin=Level.Bot
        )
        for chunk in self.assoc:
            microfeatures = self.assoc[chunk]["microfeatures"]
            weights = self.assoc[chunk]["weights"]
            dim_activations = dict()
            for mf in microfeatures:
                dim_activations[mf.dim] = max(
                    dim_activations.get(mf.dim, default_activation(mf)),
                    input_map.get(mf, default_activation(mf))
                )
            for dim in dim_activations:
                output[chunk] += (
                    weights[dim] * dim_activations[dim]
                )
            output[chunk] /= len(weights) ** 1.1
        return output


class NACSRealizer(SubsystemRealizer):

    def do(self):

        # Update Chunks
        for node in self.nodes:
            if isinstance(node, Chunk):
                self[node].do()

        # Propagate Top-down Flows
        for flow in self.flows:
            if flow.flow_type == FlowType.Top2Bot:
                self[flow].do()
        
        # Update Microfeatures
        for node in self.nodes:
            if isinstance(node, Microfeature):
                self[node].do()
        
        # Simultaneously Process at Both Top and Bottom Levels
        for flow in self.flows:
            if flow.flow_type in (FlowType.Top2Top, FlowType.Bot2Bot):
                self[flow].do()
        
        # Update All Nodes
        for node in self.nodes:
            self[node].do()
        
        # Propagate Bottom-up Links
        for flow in self.flows:
            if flow.flow_type == FlowType.Bot2Top:
                self[flow].do()
        
        # Update Chunks
        for node in self.nodes:
            if isinstance(node, Chunk):
                self[node].do()
        
        # Update Appraisal
        self[self.appraisal].do()
