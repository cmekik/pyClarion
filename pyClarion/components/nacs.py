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


class AssociativeRuleCollection(object):
    """Propagates activations among chunks."""

    def __init__(self, assoc = None, default = 0):

        self.assoc: AssociativeRuleDict = assoc or dict()
        self.default = default

    def __call__(self, construct, inputs, **kwargs):
        
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
            rule_body = dict(zip(condition, weights))
        else:
            rule_body = {condition: 1. for condition in conditions}

        rule_list = self.assoc.setdefault(conclusion, list())
        rule_list.append(rule_body)


class InterlevelLinkCollection(object):
    """Propagates activations in a top-down manner."""

    def __init__(self, assoc=None, default=0):

        self.assoc: InterlevelAssociation = assoc or {}
        self.default = default

    def __call__(
        self, 
        construct: ConstructSymbol, 
        inputs: ActivationPacket, 
        mode: ConstructType = None,
        **kwargs: Any
    ) -> ActivationPacket:

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

        d = {}
        packets = (pull_func() for pull_func in inputs.values())
        strengths = simple_junction(packets)
        for ch, dim_dict in self.assoc.items():
            divisor = sum(w for w, _ in dim_dict.values())
            for dim, (w, feats) in dim_dict.items():
                s_dim = max(strengths.get(feat, self.default) for feat in feats)
                d[ch] = d.get(ch, self.default) + ((w * s_dim) / divisor)
        return ActivationPacket(strengths=d)

    def link(self, chunk_, *features, weights=None):

        link_dict = self.assoc.setdefault(chunk_, dict())
        for feat in features:
            w_default = weights.get(feat, 1) if weights is not None else 1
            w, feature_set = link_dict.setdefault(feat.dim, (1, set()))
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
