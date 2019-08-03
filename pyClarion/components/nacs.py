'''
Implementation of the non-action-centered subsystem in standard Clarion.
'''


import typing as typ
from pyClarion.base import *
from pyClarion.components.utils import *


AssociativeRuleDict = (
    typ.Dict[
        # Conclusion chunk:
        ConstructSymbol, 
        # Condition chunks and corresponding weights for each rule associated 
        # with given conclusion:
        typ.List[typ.Dict[ConstructSymbol, typ.Any]]
    ]
) 
InterlevelAssociation = (
    typ.Dict[
        ConstructSymbol,
        typ.Dict[
            typ.Hashable, # Dimension 
            typ.Tuple[
                typ.Any, # Dimensional weight
                typ.Set[ConstructSymbol] # Dimensional microfeatures
            ]
        ],
     ]
)


class AssociativeRuleCollection(Proc):
    """
    Propagates activations among chunks through associative rules.
    
    Each rule has the form 
        conclusion <- condition_1, condition_2, ..., condition_n
    The strength of the conclusion is calculated as a weighted sum of condition 
    strengths. In cases when there are multiple rules with the same conclusion, 
    the maximum is taken. By default, weights are set to 1 / n, where n is the 
    number of conditions.

    Implementation based on p. 73-74 of Anatomy of the Mind.
    """

    def __init__(self, assoc = None, default = 0):

        self.assoc: AssociativeRuleDict = assoc or dict()
        self.default = default

    def call(self, construct, inputs, **kwargs):

        if len(kwargs) > 0:
            raise ValueError(
                (
                    "Unexpected keyword arguments passed to {}.call(): '{}'."
                ).format(self.__class__.__name__, next(iter(kwargs.keys())))
            )
        
        d = dict()
        packets = (pull_func() for pull_func in inputs.values())
        strengths = simple_junction(packets)
        for conc, cond_list in self.assoc.items():
            for cond in cond_list:
                s = linear_rule_strength(cond, strengths, self.default) 
                d[conc] = max(s, d.get(conc, self.default))
        return ActivationPacket(strengths=d)

    def add_rule(self, conclusion, *conditions, weights=None):

        if weights is not None:
            if len(weights) != len(conditions):
                raise ValueError(
                    (
                        "Number of weights ({}) and conditions ({}) do not "
                        "match."
                    ).format(len(weights), len(conditions))
                )
            rule_body = dict(zip(conditions, weights))
        else:
            rule_body = {condition: 1. for condition in conditions}

        rule_list = self.assoc.setdefault(conclusion, list())
        rule_list.append(rule_body)


class InterlevelLinkCollection(Proc):
    """
    Propagates activations between chunks and features.

    Features are linked to chunks by dimensional weights ranging in (0, 1] 
    (same dimension = same weight). 
    
    During a top-down cycle, chunk strengths are multiplied by dimensional 
    weights to get dimensional strengths. Dimensional strengths are then 
    distributed evenly among features of the corresponding dimension that 
    are linked to the source chunk. 
    
    During a bottom-up cycle, chunk strengths are computed as a weighted sum 
    of the maximum activation of linked features within each dimension. The 
    weights are simply top-down weights normalized over dimensions. 

    Implementation is based on p. 77-78 of Anatomy of the Mind.
    """

    def __init__(self, assoc=None, default=0):

        self.assoc: InterlevelAssociation = assoc or {}
        self.default = default

    def call(
        self, 
        construct: ConstructSymbol, 
        inputs: ActivationPacket, 
        mode: ConstructType = None,
        **kwargs: Any
    ) -> ActivationPacket:

        if len(kwargs) > 0:
            raise ValueError(
                (
                    "Unexpected keyword arguments passed to {}.call(): '{}'."
                ).format(self.__class__.__name__, next(iter(kwargs.keys())))
            )

        packet: ActivationPacket
        if mode == ConstructType.flow_bt:
            packet = self.bottom_up(construct, inputs)
        elif mode == ConstructType.flow_tb:
            packet = self.top_down(construct, inputs)
        else:
            if isinstance(mode, ConstructType):
                raise ValueError("Unexpected mode value {}.".format(mode))
            elif mode is None:
                raise TypeError(
                    "{}.__call__() expects a 'mode' option.".format(type(self))
                )
            else:
                raise TypeError(
                    "Mode must be of type ConstructType, "
                    "got {} instead.".format(type(mode))
                )
        return packet

    def top_down(self, construct, inputs):
        """
        Execute a top-down activation cycle.

        :param construct: Construct symbol for client construct.
        :param inputs: Dictionary mapping input constructs to their pull 
            methods.
        """

        d = {}
        packets = (pull_func() for pull_func in inputs.values())
        strengths = simple_junction(packets)
        for ch, dim_dict in self.assoc.items():
            for dim, (weight, feats) in dim_dict.items():
                for feat in feats:
                    s = weight * strengths.get(ch, self.default)
                    d[feat] = max(s, d.get(feat, self.default))
        return ActivationPacket(strengths=d)

    def bottom_up(self, construct, inputs): 
        """
        Execute a bottom-up activation cycle.

        :param construct: Construct symbol for client construct.
        :param inputs: Dictionary mapping input constructs to their pull 
            methods.
        """

        d = {}
        packets = (pull_func() for pull_func in inputs.values())
        strengths = simple_junction(packets)
        for ch, dim_dict in self.assoc.items():
            divisor = sum(w for w, _ in dim_dict.values())
            for dim, (w, feats) in dim_dict.items():
                s = max(strengths.get(feat, self.default) for feat in feats)
                d[ch] = d.get(ch, self.default) + w * s / divisor
        return ActivationPacket(strengths=d)

    def link(self, chunk_, *features, weights=None):
        """
        Link a chunk with some microfeatures.

        :param chunk_: Construct symbol for the chunk.
        :param features: Construct symbols for the features.
        :param weights: Dictionary of dimensional weights.
        """

        link_dict = self.assoc.setdefault(chunk_, dict())
        for feat in features:
            w = weights[feat.dim] if weights is not None else 1
            _, feature_set = link_dict.setdefault(feat.dim, (w, set()))
            feature_set.add(feat)


def nacs_propagation_cycle(nacs: Subsystem, args: Dict = None) -> None:
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
        if flow.construct.ctype == ConstructType.flow_v:
            flow_args = args.get(flow.construct, dict())
            flow_args["mode"] = ConstructType.flow_tb
            flow.propagate(args=flow_args)
    
    # Update feature strengths
    for feature_node in nacs.features.values():
        feature_node.propagate(args=args.get(feature_node.construct))
    
    # Propagate strengths within levels
    for flow in nacs.flows.values():
        if flow.construct.ctype in ConstructType.flow_h:
            flow.propagate(args=args.get(flow.construct))
    
    # Update feature strengths (account for signal from any bottom-level flows)
    for feature_node in nacs.features.values():
        feature_node.propagate(args=args.get(feature_node.construct))
    
    # Propagate feature strengths to top level
    for flow in nacs.flows.values():
        if flow.construct.ctype == ConstructType.flow_v:
            flow_args = args.get(flow.construct, dict())
            flow_args["mode"] = ConstructType.flow_bt
            flow.propagate(args=flow_args)
    
    # Update chunk strengths (account for bottom-up signal)
    for chunk_node in nacs.chunks.values():
        chunk_node.propagate(args=args.get(chunk_node.construct))
    
    # Select response
    for response_realizer in nacs.responses.values():
        response_realizer.propagate(args=args.get(response_realizer.construct))
