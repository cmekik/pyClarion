"""Provides propagators for standard Clarion subsystems."""


__all__ = ["NACSCycle"]


from typing import Dict
from pyClarion.base import Subsystem, ConstructType


class NACSCycle(object):

    def __call__(self, nacs: Subsystem, args: Dict = None) -> None:
        """
        Execute NACS activation cycle on given subsystem realizer.
        
        Not designed for use with flow_bt, flow_tb Flow objects.
        """

        if args is None: args = dict()

        # Update chunk strengths
        for chunk_node in nacs.chunks.values(): # type: ignore
            chunk_node.propagate(args=args.get(chunk_node.construct))

        # Propagate chunk strengths to bottom level
        for flow in nacs.flows.values(): # type: ignore
            if flow.construct.ctype == ConstructType.flow_tb:
                flow_args = args.get(flow.construct, dict())
                flow.propagate(args=flow_args)
        
        # Update feature strengths
        for feature_node in nacs.features.values(): # type: ignore
            feature_node.propagate(args=args.get(feature_node.construct))
        
        # Propagate strengths within levels
        for flow in nacs.flows.values(): # type: ignore
            if flow.construct.ctype in ConstructType.flow_h:
                flow.propagate(args=args.get(flow.construct))
        
        # Update feature strengths (account for signal from any 
        # bottom-level flows)
        for feature_node in nacs.features.values(): # type: ignore
            feature_node.propagate(args=args.get(feature_node.construct))
        
        # Propagate feature strengths to top level
        for flow in nacs.flows.values(): # type: ignore
            if flow.construct.ctype == ConstructType.flow_bt:
                flow_args = args.get(flow.construct, dict())
                flow.propagate(args=flow_args)
        
        # Update chunk strengths (account for bottom-up signal)
        for chunk_node in nacs.chunks.values(): # type: ignore
            chunk_node.propagate(args=args.get(chunk_node.construct))
        
        # Select response
        for response_realizer in nacs.responses.values(): # type: ignore
            response_realizer.propagate(
                args=args.get(response_realizer.construct)
            )
