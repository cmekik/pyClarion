"""Tools for creating, managing, and processing rules."""


__all__ = ["Rules", "AssociativeRules"]


from ..base import ConstructType, Symbol, Propagator
from ..base import numdicts as nd
from ..utils.str_funcs import pstr_iterable, pstr_iterable_cb
from types import MappingProxyType
from collections.abc import MutableMapping


class Rules(MutableMapping):
    """
    A simple rule database.

    Rules are stored in a dict of the form:

    {
        rule(1): Rule(
            conc=chunk("conc1"),
            weights={cond1: w1, ..., condn: wn}
        ),
        ... # other rules
    }            
    """

    # Dev Note:
    # This is a fairly naive datastructure with naive methods. Consequently, 
    # finding a rule form has complexity O(N) where N is the number of rules 
    # with a given conclusion. Hopefully the naive implementation makes the 
    # whole thing easily intelligible. -CSM       

    _format = {"indent": 4, "digits": 3}

    class Rule(object):
        """Represents a rule form."""

        __slots__ = ("_conc", "_weights")

        def __init__(self, conc, *conds, weights=None):
            """
            Initialize a new rule.

            If conditions contains items that do not appear in weights, these 
            weights is extend to map each of these items to a weight of 1. If 
            weights is None, it is assumed to be an empty weight dict. 

            At the end of initialization, weights is renormalized such that 
            each weight w is mapped to w / sum(weights.values())

            :param conclusion: A chunk symbol for the rule conclusion.
            :param conditions: A sequence of chunk symbols representing rule 
                conditions.
            :param weights: An optional mapping from condition chunk symbols 
                to condition weights.
            """
            
            ws = nd.NumDict()
            if weights is not None:
                ws.update(weights)
            ws.extend(conds, value=1.0)

            ws /= sum(ws.values())

            self._conc = conc
            self._weights = nd.FrozenNumDict(ws)

        def __repr__(self):

            return "Rule(weights={})".format(self.weights)

        def __eq__(self, other):

            if isinstance(other, Rules.Rule):
                b = (
                    self.conc == other.conc and 
                    nd.isclose(self.weights, other.weights)
                )
                return b
            else:
                return NotImplemented

        @property
        def conc(self):
            """Conclusion of rule."""

            return self._conc

        @property
        def weights(self):
            """Conditions and condition weights of rule."""

            return self._weights

    def __init__(self):

        self._data = {} 

    def __repr__(self):

        repr_ = "{}({})".format(type(self).__name__, repr(self._data))
        return repr_

    def __len__(self):

        return len(self._data)

    def __iter__(self):

        yield from iter(self._data)

    def __getitem__(self, key):

        return self._data[key]

    def __setitem__(self, key, val):

        self._data[key] = val

    def __delitem__(self, key):

        del self._data[key]

    def link(self, r, conc, *conds, weights=None):
        """Add a new rule."""

        form = self.Rule(conc, *conds, weights=weights)
        self[r] = form

    def contains_form(self, form):
        """
        Check if the rule set contains a given rule form.
        
        See Rules.Rule for details on rule forms.
        """

        return any(form == entry for entry in self.values())

    def pstr(self):
        """Return a pretty string representation of self."""

        body = pstr_iterable(
            iterable=self._data, 
            cb=pstr_iterable_cb, 
            cbargs={"digits": self._format["digits"]}, 
            indent=self._format["indent"], 
            level=1
        )
        size = len(self._data)
        head = " " * (size > 0) * self._format["indent"] + "data = "      
        content = head + body
        s = "{cls}({nl}{content}{nl})".format(
            cls=type(self).__name__, content=content, nl="\n" * bool(size)
        )
        return s

    def pprint(self):
        """Pretty print self."""

        print(self.pstr())


class AssociativeRules(Propagator):
    """
    Propagates activations among chunks through associative rules.
    
    The strength of the conclusion is calculated as a weighted sum of condition 
    strengths. In cases where there are multiple rules with the same conclusion, 
    the maximum is taken. 

    Implementation based on p. 73-74 of Anatomy of the Mind.
    """

    _serves = ConstructType.flow_tt

    def __init__(self, source: Symbol, rules: Rules):

        self.source = source
        self.rules = rules

    def expects(self, construct):

        return construct == self.source

    def call(self, inputs):

        d = nd.NumDict()
        strengths = inputs[self.source]
        for r, form in self.rules.items():
            cd = nd.restrict(strengths, form.weights)
            cd *= form.weights 
            s_r = sum(cd.values())
            d[form.conc] = max(d[form.conc], s_r)
            d[r] = s_r
        
        return d
