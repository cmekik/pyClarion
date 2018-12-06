'''
Implementation of the non-action-centered subsystem in standard Clarion.
'''


import typing as typ
from pyClarion.base import *


AssociativeRuleDict = (
    typ.Dict[
        # Conclusion chunk
        ConstructSymbol, 
        # Condition chunks and corresponding weights for each rule associated 
        # with given conclusion
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

    def __init__(self, assoc = None, default_strength = None):

        self.assoc: AssociativeRuleDict = assoc or dict()
        self.default_strength = default_strength

    def __call__(self, strengths):
        
        d = dict()
        for conc, conds in self.iter_assoc():
            s = sum(
                w * strengths.get(c, self.default_strength(c)) 
                for c, w in conds.items()
            )
            if d.get(conc, self.default_strength(conc)) < s:
                d[conc] = s
        return d

    def iter_assoc(self):

        for conc, cond_list in self.assoc.items():
            for conds in cond_list:
                yield conc, conds


class TopDownLinks(object):

    def __init__(self, assoc = None, default_strength = None):

        self.assoc: InterlevelAssociation = assoc or {}
        self.default_strength = default_strength

    def __call__(self, strengths):

        d = {}
        for chunk, weight, mf in self.iter_assoc():
            s = weight * strengths.get(chunk, self.default_strength(chunk))
            d[mf] = max(s, d.get(mf, self.default_strength(mf)))
        return d

    def iter_assoc(self):

        for chunk, dim_dict in self.assoc.items():
            for dim, (weight, mfs) in dim_dict.items():
                for mf in mfs:
                    yield chunk, weight, mf


class BottomUpLinks(object):

    def __init__(self, assoc = None, default_strength = None):

        self.assoc: InterlevelAssociation = assoc or {}
        self.default_strength = default_strength

    def __call__(self, strengths):

        d = {}
        for chunk, n_dim, weight, mfs in self.iter_assoc():
            s_mf = max(
                strengths.get(mf, self.default_strength(mf)) for mf in mfs
            )
            d[chunk] = (
                d.get(chunk, self.default_strength(chunk)) + 
                (weight * s_mf) / (n_dim ** 1.1)
            )
        return d

    def iter_assoc(self):

        for chunk, dim_dict in self.assoc.items():
            n_dim = len(dim_dict)
            for dim, (weight, mfs) in dim_dict.items():
                yield chunk, n_dim, weight, mfs


def may_connect(source: ConstructSymbol, target: ConstructSymbol) -> bool:
    """Return true if source may send output to target."""
    
    possibilities = [
        (
            source.ctype in ConstructType.Node and 
            target.ctype is ConstructType.Appraisal
        ),
        (
            source.ctype is ConstructType.Microfeature and
            target.ctype is ConstructType.Flow and
            typ.cast(FlowID, target.cid).ftype in FlowType.BB | FlowType.BT
        ),
        (
            source.ctype is ConstructType.Chunk and
            target.ctype is ConstructType.Flow and
            typ.cast(FlowID, target.cid).ftype in FlowType.TT | FlowType.TB
        ),
        (
            source.ctype is ConstructType.Flow and
            target.ctype is ConstructType.Microfeature and
            typ.cast(FlowID, source.cid).ftype in FlowType.BB | FlowType.TB
        ),
        (
            source.ctype is ConstructType.Flow and
            target.ctype is ConstructType.Chunk and
            typ.cast(FlowID, source.cid).ftype in FlowType.TT | FlowType.BT
        ),
        (
            source.ctype is ConstructType.Appraisal and
            target.ctype is ConstructType.Behavior 
        )
    ]
    return any(possibilities)


def nacs_propagation_cycle(realizer: SubsystemRealizer) -> None:
    """Execute NACS activation propagation cycle on given realizer."""

    for node in realizer.nodes:
        if node.ctype is ConstructType.Chunk:
            realizer[node].propagate()

    for flow in realizer.flows:
        if typ.cast(FlowID, flow.cid).ftype is FlowType.TB:
            realizer[flow].propagate()
    
    for node in realizer.nodes:
        if node.ctype is ConstructType.Microfeature:
            realizer[node].propagate()
    
    for flow in realizer.flows:
        if typ.cast(FlowID, flow.cid).ftype in FlowType.TT | FlowType.BB:
            realizer[flow].propagate()
    
    for node in realizer.nodes:
        realizer[node].propagate()
    
    for flow in realizer.flows:
        if typ.cast(FlowID, flow.cid).ftype is FlowType.BT:
            realizer[flow].propagate()
    
    for node in realizer.nodes:
        if node.ctype is ConstructType.Chunk:
            realizer[node].propagate()
    
    for appraisal in realizer.appraisals:
        realizer[appraisal].propagate()
