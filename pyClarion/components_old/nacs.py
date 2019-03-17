'''
Implementation of the non-action-centered subsystem in standard Clarion.
'''


import typing as typ
from pyClarion.base import *


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

    def __init__(self, assoc = None, default_strength = None):

        self.assoc: AssociativeRuleDict = assoc or dict()
        self.default_strength = default_strength

    def __call__(self, strengths):
        
        d = dict()
        for conc, conds in self._iter_assoc():
            s = sum(
                w * strengths.get(c, self.default_strength(c)) 
                for c, w in conds.items()
            )
            if d.get(conc, self.default_strength(conc)) < s:
                d[conc] = s
        return d

    def _iter_assoc(self):

        for conc, cond_list in self.assoc.items():
            for conds in cond_list:
                yield conc, conds


class TopDownLinks(object):
    """Propagates activations in a top-down manner."""

    def __init__(self, assoc = None, default_strength = None):

        self.assoc: InterlevelAssociation = assoc or {}
        self.default_strength = default_strength

    def __call__(self, strengths):

        d = {}
        for chunk, weight, mf in self._iter_assoc():
            s = weight * strengths.get(chunk, self.default_strength(chunk))
            d[mf] = max(s, d.get(mf, self.default_strength(mf)))
        return d

    def _iter_assoc(self):

        for chunk, dim_dict in self.assoc.items():
            for dim, (weight, mfs) in dim_dict.items():
                for mf in mfs:
                    yield chunk, weight, mf


class BottomUpLinks(object):
    """Propagates activations in a bottom-up manner."""

    def __init__(self, assoc = None, default_strength = None):

        self.assoc: InterlevelAssociation = assoc or {}
        self.default_strength = default_strength

    def __call__(self, strengths):

        d = {}
        for chunk, n_dim, weight, mfs in self._iter_assoc():
            s_mf = max(
                strengths.get(mf, self.default_strength(mf)) for mf in mfs
            )
            d[chunk] = (
                d.get(chunk, self.default_strength(chunk)) + 
                (weight * s_mf) / (n_dim ** 1.1)
            )
        return d

    def _iter_assoc(self):

        for chunk, dim_dict in self.assoc.items():
            n_dim = len(dim_dict)
            for dim, (weight, mfs) in dim_dict.items():
                yield chunk, n_dim, weight, mfs


def nacs_propagation_cycle(realizer: SubsystemRealizer) -> None:
    """Execute NACS activation cycle on given subsystem realizer."""

    for node in realizer.nodes:
        if node.ctype == ConstructType.chunk:
            realizer[node].propagate()

    for flow in realizer.flows:
        if flow.ctype == ConstructType.flow_tb:
            realizer[flow].propagate()
    
    for node in realizer.nodes:
        if node.ctype == ConstructType.feature:
            realizer[node].propagate()
    
    for flow in realizer.flows:
        if flow.ctype in ConstructType.flow_tt | ConstructType.flow_tt:
            realizer[flow].propagate()
    
    for node in realizer.nodes:
        realizer[node].propagate()
    
    for flow in realizer.flows:
        if flow.ctype == ConstructType.flow_bt:
            realizer[flow].propagate()
    
    for node in realizer.nodes:
        if node.ctype == ConstructType.chunk:
            realizer[node].propagate()
    
    for appraisal in realizer.responses:
        realizer[appraisal].propagate()
