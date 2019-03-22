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

    def __call__(self, construct, inputs):
        
        d = dict()
        packets = (pull_func() for pull_func in inputs.values())
        strengths = simple_junction(packets)
        for conc, cond_list in self.assoc.items():
            for cond in cond_list:
                s = linear_rule_strength(cond, strengths, self.default) 
                d[conc] = max(s, d.get(conc, self.default))
        return ActivationPacket(strengths=d)


class InterlevelLinks(object):
    """Propagates activations in a top-down manner."""

    def __init__(self, assoc=None, default=0):

        self.assoc: InterlevelAssociation = assoc or {}
        self.default = default

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


def nacs_propagation_cycle(nacs: SubsystemRealizer) -> None:
    """Execute NACS activation cycle on given subsystem realizer."""

    # Update chunk strengths
    for node in nacs.nodes:
        if node.ctype == ConstructType.chunk:
            nacs[node].propagate()

    # Propagate chunk strengths to bottom level
    for flow in nacs.flows:
        if flow.ctype == ConstructType.flow_tb:
            nacs[flow].propagate()
    
    # Update feature strengths
    for node in nacs.nodes:
        if node.ctype == ConstructType.feature:
            nacs[node].propagate()
    
    # Propagate strengths within levels
    for flow in nacs.flows:
        if flow.ctype in ConstructType.flow_h:
            nacs[flow].propagate()
    
    # Update feature strengths (account for signal from any bottom-level flows)
    for node in nacs.nodes:
        if node.ctype == ConstructType.feature:
            nacs[node].propagate()
    
    # Propagate feature strengths to top level
    for flow in nacs.flows:
        if flow.ctype == ConstructType.flow_bt:
            nacs[flow].propagate()
    
    # Update chunk strengths (account for bottom-up signal)
    for node in nacs.nodes:
        if node.ctype == ConstructType.chunk:
            nacs[node].propagate()
    
    # Select response
    for resp in nacs.responses:
        nacs[resp].propagate()
