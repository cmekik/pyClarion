"""Provides components for implementing NACS in standard Clarion."""


__all__ = ["AssociativeRulePropagator", "TopDown", "BottomUp", "NACSCycle"]


from typing import Dict
from pyClarion.base import PropagatorA, ActivationPacket, Subsystem, ConstructType
from pyClarion.utils.funcs import simple_junction, linear_rule_strength
from pyClarion.components.datastructures import Chunks, AssociativeRules 


class AssociativeRulePropagator(PropagatorA):
    """
    Propagates activations among chunks through associative rules.
    
    The strength of the conclusion is calculated as a weighted sum of condition 
    strengths. In cases where there are multiple rules with the same conclusion, 
    the maximum is taken. 

    Implementation based on p. 73-74 of Anatomy of the Mind.
    """

    def __init__(self, rules: AssociativeRules, default=0.0):

        self.rules = rules
        self.default = default

    def call(self, construct, inputs, **kwds):

        if len(kwds) > 0:
            raise ValueError(
                (
                    "Unexpected keyword arguments passed to {}.call(): '{}'."
                ).format(self.__class__.__name__, next(iter(kwds.keys())))
            )
        
        d = {}
        packets = inputs.values()
        strengths = simple_junction(packets)
        for conc, cond_dict in self.rules:
            s = linear_rule_strength(cond_dict, strengths, self.default) 
            d[conc] = max(s, d.get(conc, self.default))
        
        return d


class TopDown(PropagatorA):
    """
    Computes a top-down activations in NACS.

    During a top-down cycle, chunk strengths are multiplied by dimensional 
    weights to get dimensional strengths. Dimensional strengths are then 
    distributed evenly among features of the corresponding dimension that 
    are linked to the source chunk.

    Implementation is based on p. 77-78 of Anatomy of the Mind.
    """

    def __init__(self, chunks=None, default=0):

        self.chunks: Chunks = chunks if chunks is not None else Chunks()
        self.default = default

    def call(self, construct, inputs, **kwds):
        """
        Execute a top-down activation cycle.

        :param construct: Construct symbol for client construct.
        :param inputs: Dictionary mapping input constructs to their pull 
            methods.
        """

        if len(kwds) > 0:
            raise ValueError(
                (
                    "Unexpected keyword arguments passed to {}.call(): '{}'."
                ).format(self.__class__.__name__, next(iter(kwds.keys())))
            )

        d = {}
        packets = inputs.values()
        strengths = simple_junction(packets)
        for ch, dim_dict in self.chunks.items():
            for _, data in dim_dict.items():
                s = data["weight"] * strengths.get(ch, self.default)
                for feat in data["values"]:
                    d[feat] = max(s, d.get(feat, self.default))

        return d


class BottomUp(PropagatorA):
    """
    Computes a bottom-up activations in NACS.

    During a bottom-up cycle, chunk strengths are computed as a weighted sum 
    of the maximum activation of linked features within each dimension. The 
    weights are simply top-down weights normalized over dimensions. 

    Implementation is based on p. 77-78 of Anatomy of the Mind.
    """

    def __init__(self, chunks=None, default=0):

        self.chunks: Chunks = chunks if chunks is not None else Chunks()
        self.default = default

    def call(self, construct, inputs, **kwds): 
        """
        Execute a bottom-up activation cycle.

        :param construct: Construct symbol for client construct.
        :param inputs: Dictionary mapping input constructs to their pull 
            methods.
        """

        if len(kwds) > 0:
            raise ValueError(
                (
                    "Unexpected keyword arguments passed to {}.call(): '{}'."
                ).format(self.__class__.__name__, next(iter(kwds.keys())))
            )

        d = {}
        packets = inputs.values()
        strengths = simple_junction(packets)
        for ch, dim_dict in self.chunks.items():
            divisor = sum(data["weight"] for data in dim_dict.values())
            for _, data in dim_dict.items():
                s = max(strengths.get(f, self.default) for f in data["values"])
                d[ch] = d.get(ch, self.default) + data["weight"] * s / divisor
        
        return d


class NACSCycle(object):

    def __call__(self, nacs: Subsystem, args: Dict = None) -> None:
        """
        Execute NACS activation cycle on given subsystem realizer.
        
        Not designed for use with flow_bt, flow_tb Flow objects.
        """

        if args is None: args = dict()

        # Update chunk strengths
        for chunk_node in nacs.chunks.values():
            chunk_node.propagate(args=args.get(chunk_node.construct))

        # Propagate chunk strengths to bottom level
        for flow in nacs.flows.values():
            if flow.construct.ctype == ConstructType.flow_tb:
                flow_args = args.get(flow.construct, dict())
                flow.propagate(args=flow_args)
        
        # Update feature strengths
        for feature_node in nacs.features.values():
            feature_node.propagate(args=args.get(feature_node.construct))
        
        # Propagate strengths within levels
        for flow in nacs.flows.values():
            if flow.construct.ctype in ConstructType.flow_h:
                flow.propagate(args=args.get(flow.construct))
        
        # Update feature strengths (account for signal from any 
        # bottom-level flows)
        for feature_node in nacs.features.values():
            feature_node.propagate(args=args.get(feature_node.construct))
        
        # Propagate feature strengths to top level
        for flow in nacs.flows.values():
            if flow.construct.ctype == ConstructType.flow_bt:
                flow_args = args.get(flow.construct, dict())
                flow.propagate(args=flow_args)
        
        # Update chunk strengths (account for bottom-up signal)
        for chunk_node in nacs.chunks.values():
            chunk_node.propagate(args=args.get(chunk_node.construct))
        
        # Select response
        for response_realizer in nacs.responses.values():
            response_realizer.propagate(
                args=args.get(response_realizer.construct)
            )
