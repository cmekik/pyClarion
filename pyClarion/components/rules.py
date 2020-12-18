"""Tools for creating, managing, and processing rules."""


__all__ = ["Rule", "Rules", "AssociativeRules", "ActionRules"]


from ..base import ConstructType, Symbol, Propagator, UpdaterS, rule
from .. import numdicts as nd

from typing import Mapping, TypeVar, Generic, Type, overload
from types import MappingProxyType
from collections.abc import MutableMapping


class Rule(object):
    """Represents a rule form."""

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
        
        ws = nd.MutableNumDict()
        if weights is not None:
            ws.update(weights)
        ws.extend(conds, value=1.0)

        w_sum = nd.val_sum(ws)
        if w_sum > 1.0: 
            ws /= w_sum

        self._conc = conc
        self._weights = nd.freeze(ws)

    def __repr__(self):

        return "Rule(weights={})".format(self.weights)

    def __eq__(self, other):

        if isinstance(other, Rule):
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

    def strength(self, strengths):
        """
        Compute rule strength given condition strengths.
        
        The rule strength is computed as the weighted sum of the condition 
        strengths in strengths.

        Implementation based on p. 60 and p. 73 of Anatomy of the Mind.
        """

        weighted = nd.keep(strengths, keys=self.weights) * self.weights
        
        return nd.val_sum(weighted)


Rt = TypeVar("Rt", bound="Rule")
class Rules(MutableMapping, Generic[Rt]):
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

    # TODO: Updater and update requests need testing. - Can 

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

    @overload
    def __init__(self: "Rules[Rule]") -> None:
        ...

    @overload
    def __init__(self: "Rules[Rule]", *, max_conds: int) -> None:
        ...

    @overload
    def __init__(self, *, rule_type: Type[Rt]) -> None:
        ...

    @overload
    def __init__(self, *, max_conds: int, rule_type: Type[Rt]) -> None:
        ...

    @overload
    def __init__(
        self, data: Mapping[rule, Rt], max_conds: int, rule_type: Type[Rt]
    ) -> None:
        ...

    def __init__(
        self, 
        data: Mapping[rule, Rt] = None,
        max_conds: int = None,
        rule_type: Type[Rt] = None
    ) -> None:

        if data is None:
            data = dict()
        else:
            data = dict(data)

        self._data = data
        self.max_conds = max_conds
        self.Rule = rule_type if rule_type is not None else Rule

        self._promises: MutableMapping[rule, Rule] = dict()
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
        if isinstance(val, self.Rule):
            self._data[key] = val
        else:
            msg = "This rule database expects rules of type '{}'." 
            TypeError(msg.format(type(self.Rule.__name__)))

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
        
        See Rule for details on rule forms.
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

        d = nd.MutableNumDict(default=0.0)
        strengths = inputs[self.source]
        for r, form in self.rules.items():
            s_r = form.strength(strengths)
            d[form.conc] = max(d[form.conc], s_r)
            d[r] = s_r
        
        return d


class ActionRules(Propagator):
    """
    Propagates activations from condition to action chunks using action rules.
    
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

        s_r = nd.MutableNumDict(default=0)
        for r, form in self.rules.items():
            s_r[r] = form.strength(strengths)

        probabilities = nd.boltzmann(s_r, self.temperature)
        selection = nd.draw(probabilities, n=1)

        s_a = s_r * selection
        s_a = nd.threshold(s_a, th=0, keep_default=True)
        d = nd.transform_keys(s_a, func=lambda r: self.rules[r].conc)
        d += s_a

        return d
