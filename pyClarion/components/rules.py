"""Tools for creating, managing, and processing rules."""


__all__ = ["Rules", "AssociativeRules", "ActionRules"]


from ..base import ConstructType, Symbol, Propagator, UpdaterS, rule
from ..base import numdicts as nd

from types import MappingProxyType
from typing import Mapping
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

    class Rule(object):
        """Represents a rule form."""

        # TODO: Updater and update requests need testing. - Can 

        __slots__ = ("_conc", "_weights")

        def __init__(self, conc, *conds, weights=None):
            """
            Initialize a new rule.

            If conditions contains items that do not appear in weights, these 
            weights is extend to map each of these items to a weight of 1. If 
            weights is None, it is assumed to be an empty weight dict. 

            At the end of initialization, if the weights sum to more than 1.0, 
            weights is renormalized such that each weight w is mapped to 
            w / sum(weights.values())

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

            w_sum = nd.val_sum(ws)
            if w_sum > 1.0: 
                ws /= w_sum

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

    class Updater(UpdaterS):
        """
        Applies requested updates to a client Rules instance.
        
        Assumes any updates will be issued by constructs subordinate to 
        self.client.
        """

        _serves = ConstructType.container_construct

        def __init__(self, rules: "Rules") -> None:
            """Initialize a Rules.Updater instance."""

            self.rules = rules

        @property
        def expected(self):

            return frozenset()

        def __call__(self, inputs, output, update_data):
            """Resolve all outstanding rule database update requests."""

            self.rules.resolve_update_requests()

    def __init__(
        self, 
        data: Mapping[rule, "Rules.Rule"] = None,
        max_conds: int = None
    ) -> None:

        if data is None:
            data = dict()
        else:
            data = dict(data)

        self._data = data
        self.max_conds = max_conds

        self._promises: MutableMapping[rule, Rules.Rule] = dict()
        self._promises_proxy = MappingProxyType(self._promises)
        self._updater = type(self).Updater(self)

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

        self._validate_rule_form(val)
        self._data[key] = val

    def __delitem__(self, key):

        del self._data[key]

    @property
    def updater(self):
        """Updater object for bound to self."""

        return self._updater

    @property
    def promises(self):
        """A view of promised updates."""

        return self._promises_proxy

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

    def request_update(self, r, form):
        """
        Inform self of a new rule to be applied at a later time.
        
        Adds (r, form) to an internal future update dict. Upon call to 
        self.resolve_update_requests(), the update dict will be passed as an 
        argument to self.update(). 
        
        Will overwrite existing rule if r is already member of self. Does 
        not check for duplicate forms. Will throw an error if an update is 
        already registered for rule r.

        Does not validate the rule form before registering the request. 
        Validation occurs at update time. 
        """

        if ch in self._promises:
            msg = "Rule {} already registered for a promised update."
            raise ValueError(msg.format(r))
        else:
            self._promises[r] = form

    def resolve_update_requests(self):
        """
        Apply any promised updates (see Rules.request_update()).
        
        Clears promised update dict upon completion.
        """

        self.update(self._promises)
        self._promises.clear()

    def _validate_rule_form(self, form):

        if self.max_conds is not None and len(form.weights) > self.max_conds:
            msg = "Received rule with {} conditions; maximum allowed is {}."
            raise ValueError(msg.format(len(form.weights), self.max_conds))


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

    @property
    def expected(self):

        return frozenset((self.source,))

    def call(self, inputs):

        d = nd.NumDict()
        strengths = inputs[self.source]
        for r, form in self.rules.items():
            cd = nd.restrict(strengths, form.weights)
            cd *= form.weights 
            s_r = nd.val_sum(cd)
            d[form.conc] = max(d[form.conc], s_r)
            d[r] = s_r
        
        return d


class ActionRules(Propagator):
    """
    Propagates activations among chunks through action rules.
    
    Action rules compete to be selected based on their rule strengths, which is 
    equal to the product of an action rule's weight and the strength of its 
    condition chunk. The rule strength of the selected action is then 
    propagated to its conclusion. 
    """

    _serves = ConstructType.flow_tt

    def __init__(
        self, source: Symbol, rules: Rules, temperature: float = .01
    ) -> None:

        if rules.max_conds is None or rules.max_conds > 1:
            msg = "Rule database must not accept multiple condition rules."
            raise ValueError(msg)

        self.source = source
        self.rules = rules
        self.temperature = temperature

    @property
    def expected(self):

        return frozenset((self.source,))

    def call(self, inputs):

        strengths = inputs[self.source]

        s_r = nd.NumDict()
        for r, form in self.rules.items():
            _s_r = nd.restrict(strengths, form.weights) * form.weights
            s_r[r] = nd.val_sum(_s_r)
        prs = nd.boltzmann(s_r, self.temperature)
        selection = nd.draw(prs, 1)
        s_a = s_r * selection

        d = nd.NumDict()
        for r in s_a:
            d[self.rules[r].conc] = max(d[self.rules[r].conc], s_a[r])
        d |= selection

        return d
