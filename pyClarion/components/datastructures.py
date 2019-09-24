"""
Some basic datastructures for building pyClarion agents.

Includes definitions for:
    - Chunk databases
    - Rule databases
    - Complex match specifications for controlling construct connectivity
    - TBD
"""

from pyClarion.base.symbols import *
from pyClarion.utils.str_funcs import pstr_iterable, pstr_iterable_cb
from types import MappingProxyType


class Chunks(object):
    """
    A simple chunk database.

    This object provides methods for constructing, maintaining, and inspecting a 
    database of links between chunk nodes and corresponding features.

    It is important to distinguish chunks from chunk nodes. Put simply, the 
    difference is that 'chunk' refers to a labeled chunk node along with its 
    links to feature nodes (these links may be empty). A chunk form, on the 
    other hand, refers to the pattern of connections between a labeled chunk 
    node and some feature nodes that define a chunk (excluding the labeled chunk 
    node itself).

    Chunks forms are stored in a dict of the form:
    _data = {
        chunk(1): {
            "dim1": {
                "weight": weight_1_1,
                "values": {
                    feature("dim1", "val1")
                    feature("dim2", "val2")
                    ... # other values
                }
            },
            ... # other dimensions
        },
        ... # other chunks
    } 

    Under normal cricumstances, this dict should not be directly accessed and/or 
    modified. However, a dict of this form may be passed at initialization time 
    to set initial rules.
    """

    _format = {"indent": 4, "digits": 3}

    def __init__(self, data=None):

        self.validate_init_data(data)
        self._data = dict(data) if data is not None else dict()

    def __repr__(self):

        repr_ = "{}(data={})".format(type(self).__name__, repr(self._data))
        return repr_

    def __contains__(self, ch):
        """Return True if self contains given chunk."""

        return ch in self._data

    def __len__(self):

        return len(self._data)

    def get_form(self, ch, default=None):
        """Return the form of given chunk."""

        return self._data.get(ch, default)

    def chunks(self):
        """Return a view of chunks in self."""

        return self._data.keys()

    def forms(self):
        """Return a view of chunk forms in self."""

        return self._data.values()

    def items(self):
        """Return a view of chunk_node, chunk_form pairs in self."""

        return self._data.items()
    
    def link(self, ch, *features, weights=None):
        """Link chunk to features."""

        # If feature sequence contains duplicates, they will be ignored upon 
        # conversion to a set in update_form().
        d = self._data.setdefault(ch, dict())
        self.update_form(d, *features, weights=weights)

    def remove_chunk(self, ch):
        """Remove chunk from database."""

        del self._data[ch]
    
    def unlink_dim(self, ch, dim):
        """Unlink all features of a given dimension from chunk."""

        del self._data[ch][dim]

    def unlink_feature(self, ch, feat):
        """
        Unlink feature from chunk.

        If no features of the same dim ar linked to chunk after operation, also 
        removes the dimension.
        """

        features = self._data[ch][feat.dim]["values"]
        features.remove(feat)
        if len(features) == 0:
            self.unlink_dim(ch, feat.dim)

    def set_weight(self, ch, dim, weight):
        """Set weight associated with a dimension of chunk."""

        self._data[ch][dim]["weight"] = weight

    def contains_form(self, *features, weights=None):
        """Return true if given chunk form matches at least one chunk."""

        test_form = self.update_form(dict(), *features, weights=weights)
        return any([form == test_form for form in self._data.values()]) 

    def pstr(self):
        """Return a pretty string a representation of self."""

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

    @staticmethod
    def update_form(form, *features, weights=None):
        """
        Update given chunk form.

        :param form: A chunk form (i.e., an unlabeled chunk).
        :param features: A sequence of feature construct symbols.
        :param weights: A mapping from dimensions to corresponding weights.
        """
        
        for feat in features:
            w = weights[feat.dim] if weights is not None else 1.0
            dim_data = form.setdefault(
                feat.dim, {"weight": w, "values": set()}
            )
            dim_data["values"].add(feat)
        
        return form

    @staticmethod
    def validate_init_data(data):
        """
        Check if initial data dict has valid form.

        Enforces
            - dict values must be dimension dicts
            - dimension dicts contain keys "weight", "values"
            - the "values" key returns an object of type set

        See class header for more information on expected data structure.
        """        

        if data is not None:
            for chunk_form in data.values():
                if not isinstance(chunk_form, dict):
                    raise TypeError("Chunk form must be of type dict.")
                if "weight" not in chunk_form:
                    raise ValueError("Dimension data must contain weight info.")
                if "values" not in chunk_form:
                    raise ValueError("Dimension data must contain value info.")
                if not isinstance(chunk_form["values"], set):
                    raise TypeError("Value info must be of type set.")


class AssociativeRules(object):
    """
    A simple associative rule database.

    This object provides methods for constructing, maintaining, and inspecting a 
    database of associative links from condition chunk nodes to conclusion chunk 
    nodes.

    Each rule has the form 
        conclusion <- condition_1, condition_2, ..., condition_n
    Each condition may be associated with a weight.

    Rules are stored in a dict of the form
    _data = {
        chunk("conclusion-1"): [
            {chunk("condition-1"): w1, ...}, # a first set of conditions
            ... # other condition sets for conclusion 1
        ],
        ...
    }
    
    Under normal cricumstances, this dict should not be directly accessed and/or 
    modified. However, a dict of this form may be passed at initialization time 
    to set initial rules.

    The key `chunk("conclusion-1")` represents a conclusion chunk, the value 
    associated with this key is a list of dicts. Each dict in this list 
    represents the condition chunks and weights for a different rule whose 
    conclusion is the chunk represented in the top-level key.

    The conclusion and condition chunk symbols together are called the rule 
    form. Rules methods enforce the uniqueness of rule forms within the 
    database.  
    """

    # Dev Note:
    # This is a fairly naive datastructure with naive methods. Consequently, 
    # finding a rule form has complexity O(N) where N is the number of rules 
    # with a given conclusion. Hopefully the naive implementation makes the 
    # whole thing easily intelligible. -CSM       

    _format = {"indent": 4, "digits": 3}

    def __init__(self, data=None):

        self.validate_init_data(data)
        self._data = data if data is not None else dict()

    def __repr__(self):

        repr_ = "{}(data={})".format(type(self).__name__, repr(self._data))
        return repr_

    def __iter__(self):
        """
        Iterate over each conclusion, condition_dict pair. 
        """

        for conclusion, condition_dicts in self._data.items():
            for condition_dict in condition_dicts:
                # Dev note:
                # Yield mapping proxy type to prevent database corruption 
                # through in-place modification. A more efficient way to do 
                # this, if necessary, would be to have persistent proxies. -CSM
                yield conclusion, MappingProxyType(condition_dict)

    def __len__(self):
        """Return the number of rules in self."""

        # To compute the size of the rule set, we compute the sum of the number 
        # of rules associated with each condition. This corresponds to the 
        # length of the list of condition dicts associated with each conclusion 
        # chunk.
        return sum(map(len, self._data.values()))

    def link(self, conclusion, *conditions, weights=None):
        """
        Add a rule linking the condition chunk to the conclusion chunk.

        If conclusion, *conditions specify a preexisting rule-form, throw an 
        error.

        :conditions: Symbols representing condition chunks (duplicates will be 
            ignored).
        :param weight_map: A dict mapping conditions to their weights.
        """

        # Check if condition and weight specifications are valid.
        self.validate_condition_data(*conditions, weights=weights)

        # Check if this rule-form already exists
        exists = self.contains_form(conclusion, *conditions)
        if exists:
            raise ValueError("A rule of this form already exists.")

        # by default, weights are set to 1/n where n is num conditions
        # Dev note:
        # In the future we may provide hooks for customizing this behavior. -CSM
        n = len(conditions)
        weights = weights if weights is not None else [1 / n] * n 

        new_condition_dict = dict(zip(conditions, weights))
        condition_dicts = self._data.setdefault(conclusion, list())
        condition_dicts.append(new_condition_dict)

    def contains_form(self, conclusion, *conditions):
        """
        Check if the rule set contains a given rule form.
        
        A rule form is simply the conclusion chunk together with a specific set 
        of condition chunks. 
        """

        # Check that conditions do not contain duplicates.
        self.validate_condition_data(*conditions)
        
        condition_set = set(conditions)
        try:
            form_index = self._get_index(conclusion, condition_set)
        except ValueError:
            return False
        else:
            return True

    def pop_form(self, conclusion, *conditions):
        """
        Remove given form from self and return associated data.
        
        If conclusion or the form is not found in self, throws.
        """

        # Check that conditions do not contain duplicates.
        self.validate_condition_data(*conditions)

        condition_set = set(conditions)
        form_index = self._get_index(conclusion, condition_set)
        return conclusion, conditions, self[conclusion].pop(index=form_index)

    def pstr(self):
        """Return a pretty string a representation of self."""

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

    def _get_index(self, conclusion, condition_set):
        """
        Return the list index pointing to a given chunk form.

        Returns index i, s.t. self._data[conclusion][i] is the condition dict 
        for the rule-form specified by conlusion and conditions. If no such 
        condition dict is found, throws an error.
        """

        try:
            rule_bodies = self._data[conclusion]
        except KeyError:
            raise ValueError("No rule exists with given conclusion.")
        for i, body in enumerate(rule_bodies):
            if condition_set == set(body.keys()):
                index = i
        try:
            return index
        except NameError:
            raise ValueError("No rule exists with given conditions.") 

    @staticmethod
    def validate_condition_data(*conditions, weights=None):
        """
        Check if conditions and weights form a valid condition dict.

        Enforces
            - uniqueness of conditions
            - equality of number of conditions and weights, if weights are given

        Does not check and/or enforce data types, though conditions are expected 
        to be construct symbols naming chunks and weights are expected to be 
        numeric values.

        See class header for more information on expected data structure.
        """

        if len(conditions) > len(set(conditions)):
            # Conditions contain duplicates. This is not allowed because, 
            # conceptually, rule conditions form a set. 
            raise ValueError("Rule conditions may not contain duplicates.")
        if weights is not None and len(conditions) != len(weights):
            raise ValueError(
                "Condition and weight sequences are of different lengths."
            )

    @staticmethod
    def validate_init_data(data=None):
        """
        Check if initial data dict has valid form.

        Enforces
            - dict values must be lists of condition dicts
            - condition sets must be unique within each list

        Does not check and/or enforce data types, though conclusions and 
        conditions are expected to be construct symbols naming chunks and 
        weights are expected to be numeric values.

        See class header for more information on expected data structure.
        """

        if data is not None:
            for condition_dicts in data.values():
                # Enforce that dict values are of type list
                if not isinstance(condition_dicts, list):
                    raise TypeError(
                    "Values of initial data dict must be of type list."
                )
                # Enforce that condition_dicts contains only dicts
                for item in condition_dicts:
                    if not isinstance(item, dict):
                        raise TypeError(
                        "Value lists must only contain dicts."
                    )
                # Enforce uniqueness of condition sets for current conclusion
                condition_sets = set()
                for condition_dict in condition_dicts:
                    conditions, weights = zip(*condition_dict.items())
                    condition_sets.add(frozenset(conditions))
                if len(condition_dicts) > len(condition_sets):
                    raise ValueError(
                    "Rule database may not contain multiple rules with "
                    "identical rule-forms." 
                ) 


class MatchSpec(object):
    """
    A unary predicate that applies to construct symbols.

    MatchSpec objects are intended to facilitate checking if constructs satisfy 
    complex conditions. Such checks may be required, for example, to decide 
    whether or not to connect two construct realizers (see pyClarion.realizers). 
    In general, MatchSpec objects may be used at any point where a (potentially 
    complex) predicate must be applied to construct symbols. 

    MatchSpec objects support definition with respect to construct types or 
    arbitrary predicates, and by enumeration of matching constructs. They are 
    set-like in that they support __contains__, may be extended (through 
    addition) or contracted (through removal). However, unlike sets, MatchSpec 
    objects do not support algebraic operators such as union, intersection, 
    difference etc.
    """

    # TODO: __repr__

    def __init__(
        self, 
        ctype: ConstructType = None, 
        constructs: Iterable[ConstructSymbol] = None,
        predicates: Iterable[Callable[[ConstructSymbol], bool]] = None
    ) -> None:
        """
        Initialize a new Matcher instance.

        :param ctype: Acceptable construct type(s).
        :param constructs: Acceptable construct symbols.
        :param predicates: Custom custom predicates indicating acceptable 
            constructs. 
        """

        self.ctype = ConstructType.null_construct
        self.constructs: MutableSet[ConstructSymbol] = set()
        self.predicates: MutableSet[Callable[[ConstructSymbol], bool]] = set()
        self.add(ctype, constructs, predicates)

    def __contains__(self, key: ConstructSymbol) -> bool:
        """
        Return true if construct is in the match set.
        
        A construct is considered to be in the match set if:
            - Its construct type is in self.ctype OR
            - It is equal to a member of self.constructs OR
            - A predicate in self.predicates returns true when called on its 
              construct symbol.
        """

        val = False
        val |= key.ctype in self.ctype
        val |= key in self.constructs
        for predicate in self.predicates:
                val |= predicate(key)
        return val

    def add(
        self, 
        ctype: ConstructType = None, 
        constructs: Iterable[ConstructSymbol] = None,
        predicates: Iterable[Callable[[ConstructSymbol], bool]] = None
    ) -> None:
        """
        Extend the set of accepted constructs.
        
        See Predicate.__init__() for argument descriptions.
        """

        if ctype is not None:
            self.ctype |= ctype
        if constructs is not None:
            self.constructs |= set(constructs)
        if predicates is not None:
            self.predicates |= set(predicates)

    def remove(
        self, 
        ctype: ConstructType = None, 
        constructs: Iterable[ConstructSymbol] = None,
        predicates: Iterable[Callable[[ConstructSymbol], bool]] = None
    ) -> None:
        """
        Contract the set of accepted constructs.
        
        See Predicate.__init__() for argument descriptions.
        """

        if ctype is not None:
            self.ctype ^= ctype
        if constructs is not None:
            self.constructs ^= set(constructs)
        if predicates is not None:
            self.predicates ^= set(predicates)
