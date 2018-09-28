'''
Implementation of the non-action-centered subsystem in standard Clarion.
'''


from typing import Dict, Hashable, Set, Tuple, List
from pyClarion.base.symbols import Node, Chunk, Microfeature, Flow, FlowType
from pyClarion.base.packets import ActivationPacket, Level
from pyClarion.base.processors import Channel
from pyClarion.base.realizers.composite import SubsystemRealizer
from pyClarion.standard.common import get_default_activation


class AssociativeRules(Channel[float]):

    def __init__(self, assoc):

        self.assoc = assoc

    def __call__(self, input_map):
        
        output = ActivationPacket(
            default_factory=get_default_activation, origin=Level.TopLevel
        )
        for rule in self.assoc:
            conditions = rule["conditions"]
            conclusion = rule["conclusion"]
            strength = 0.
            for cond in conditions: 
                strength += (
                    conditions[cond] * 
                    input_map.get(cond, get_default_activation(cond))
                )
            output[conclusion] = max(output[conclusion], strength)
        return output


class TopDownChannel(Channel[float]):

    def __init__(self, assoc):

        self.assoc = assoc

    def __call__(self, input_map):

        output = ActivationPacket(
            default_factory=get_default_activation, origin=Level.TopLevel
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
            default_factory=get_default_activation, origin=Level.BottomLevel
        )
        for chunk in self.assoc:
            microfeatures = self.assoc[chunk]["microfeatures"]
            weights = self.assoc[chunk]["weights"]
            dim_activations = dict()
            for mf in microfeatures:
                dim_activations[mf.dim] = max(
                    dim_activations.get(mf.dim, get_default_activation(mf)),
                    input_map.get(mf, get_default_activation(mf))
                )
            for dim in dim_activations:
                output[chunk] += (
                    weights[dim] * dim_activations[dim]
                )
            output[chunk] /= len(weights) ** 1.1
        return output


class NACSRealizer(SubsystemRealizer):

    def __call__(self):

        # Update Chunks
        for node in self.nodes:
            if isinstance(node, Chunk):
                self[node]()

        # Propagate Top-down Flows
        for flow in self.flows:
            if flow.flow_type == FlowType.TopDown:
                self[flow]()
        
        # Update Microfeatures
        for node in self.nodes:
            if isinstance(node, Microfeature):
                self[node]()
        
        # Simultaneously Process at Both Top and Bottom Levels
        for flow in self.flows:
            if flow.flow_type in (FlowType.TopLevel, FlowType.BottomLevel):
                self[flow]()
        
        # Update All Nodes
        for node in self.nodes:
            self[node]()
        
        # Propagate Bottom-up Links
        for flow in self.flows:
            if flow.flow_type == FlowType.BottomUp:
                self[flow]()
        
        # Update Chunks
        for node in self.nodes:
            if isinstance(node, Chunk):
                self[node]()
        
        # Update Appraisal
        self[self.appraisal]()
