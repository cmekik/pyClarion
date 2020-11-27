"""
Class and function definitions for numerical dictionaries.

Numerical dictionaries (numdicts, for short) are, simply, dictionaries that map 
keys to numerical values. They may be viewed as a generalization of the 
dictionary-of-keys representation for sparse matirces in that no underlying 
array structure is asssumed.

This module exports the NumDict and FrozenNumDict classes and various functions 
operating over these classes. NumDict and FrozenNumDict are both derived from 
the helper class BaseNumDict and differ in that NumDict instances are mutable 
while FrozenNumDict instances are not.

All numdicts support several basic numerical operations like +, -, *, /, etc. 
Some boolean operations are also supported. These are modeled after fuzzy set 
theory: 
    - ~ computes 1 - d[k] for each k in d, 
    - & is min(d1[k], d2[k]) for each k in d1 or d2, 
    - | is max(d1[k], d2[k]) for each k in d1 or d2 

For binary ops applied to two numdicts, the resulting numdict inherits the type 
of the left operand (similar to set/frozenset). If an operand in a binary op 
is a constant numerical value, then that value is broadcast to all keys in the 
other operand.

Numdicts may have default values (and they do by default). Defaults are updated 
appropriately when mathematical ops are applied. Querying a missing key simply 
returns the default value (if it is defined), but the missing key is not added 
to the queried numdict. If a numdict is mutable and missing keys should be 
added when queried, the setdefault() method may be used.

From an implementation point of view, numdicts are MutableMappings wrapping 
plain dicts. When a key occurs in the wrapped dict, it is considered an 
explicit member of the numdict. Containment checks against numdicts will only 
succeed for explicit members. Note that d[key] may succeed for non-explicit 
members if a default value is defined, in this case it is dangerous to use 
d[key] to check for (explicit) membership.
"""


__all__ = [
    "NumDict", "FrozenNumDict", "restrict", "keep", "drop", "transform_keys", 
    "threshold", "clip", "isclose", "valuewise", "val_sum", "elementwise", 
    "ew_sum", "ew_max", "boltzmann", "draw"
]


from collections.abc import MutableMapping, Mapping
from typing import TypeVar, Container, Callable, Hashable
from itertools import chain
import operator
import numbers
import math
import random


class BaseNumDict(Mapping):
    """
    Base class for numerical dictionaries.

    Supports various mathematical ops. For details, see numdicts module 
    description.
    """

    __slots__ = ("_dtype", "_dict", "_default")

    def __init__(self, data=None, dtype=float, default=0.0):

        if data is None:
            data = {}

        self._dtype = dtype
        self._dict = {k: self._cast(data[k]) for k in data}
        self._default = self._cast(default) if default is not None else None

    @property
    def dtype(self):

        return self._dtype

    @property
    def default(self):

        return self._default

    def __str__(self):

        fmtargs = type(self).__name__, str(self._dict), self._default

        return "{}({}, default={})".format(*fmtargs)

    def __repr__(self):

        fmtargs = type(self).__name__, repr(self._dict), self._default

        return "{}({}, default={})".format(*fmtargs)

    def __len__(self):

        return len(self._dict)

    def __iter__(self):

        yield from iter(self._dict)

    def __contains__(self, key):
        """
        Return True iff key is explicitly set in self.

        Warning, if a self.default is not None, self[key] may return a value 
        when `key in self` returns False. Do not use self[key] to check for 
        containment.
        """

        return key in self._dict

    def __getitem__(self, key):

        try:
            return self._dict[key]
        except KeyError:
            if self._default is None:
                raise
            else:
                return self._default

    def __neg__(self):

        return self.apply_unary_op(operator.neg)

    def __abs__(self):

        return self.apply_unary_op(operator.abs)

    def __invert__(self):

        return self.apply_unary_op(self._inv)

    def __lt__(self, other):

        return self.apply_binary_op(operator.lt)

    def __le__(self, other):

        return self.apply_binary_op(operator.le)

    def __gt__(self, other):

        return self.apply_binary_op(operator.gt)

    def __ge__(self, other):

        return self.apply_binary_op(operator.ge)
   
    def __add__(self, other):

        return self.apply_binary_op(other, operator.add)

    def __sub__(self, other):

        return self.apply_binary_op(other, operator.sub)

    def __mul__(self, other):

        return self.apply_binary_op(other, operator.mul)

    def __truediv__(self, other):

        return self.apply_binary_op(other, operator.truediv)

    def __pow__(self, other):

        return self.apply_binary_op(other, operator.pow)

    def __and__(self, other):

        return self.apply_binary_op(other, min)

    def __or__(self, other):

        return self.apply_binary_op(other, max)

    def __radd__(self, other):

        return self.apply_binary_op(other, operator.add)

    def __rsub__(self, other):

        return operator.neg(self) + other

    def __rmul__(self, other):

        return self.apply_binary_op(other, operator.mul)

    def __rtruediv__(self, other):

        return self.apply_binary_op(other, self._rtruediv)

    def __rpow__(self, other):

        return self.apply_binary_op(other, self._rpow)

    def __rand__(self, other):

        return self.apply_binary_op(other, min)

    def __ror__(self, other):

        return self.apply_binary_op(other, max)

    def by(self, keyfunc, op):
        """
        Compute op over elements grouped by keyfunc.
        
        Key should be a function mapping each key in self to a grouping key.
        """

        d = {}
        for k, v in self.items():
            d.setdefault(keyfunc(k), []).append(v)
        mapping = {k: op(v) for k, v in d.items()}

        return type(self)(mapping, self.dtype, self.default)

    def apply_unary_op(self, op):
        """
        Apply op to each element of self.

        Returns a new numdict.        
        """

        mapping = {k: op(self._dict[k]) for k in self}
        
        if self.default is not None:
            default = op(self.default)
        else:
            default = None

        return type(self)(mapping, self.dtype, default)

    def apply_binary_op(self, other, op):
        """
        Apply binary op to each element of self and other.

        Returns a new numdict.
        
        If other is a constant c, acts as if other[key] = c.

        If both self and other define defaults, the new default is equal to
        op(self_default, other_default). Otherwise no default is defined.
        """

        if isinstance(other, BaseNumDict):
            if self._dtype_matches(other.dtype): 
                keys = set(self.keys()) | set(other.keys())
                mapping = {k: op(self[k], other[k]) for k in keys}
                if self.default is None:
                    default = None
                elif other.default is None:
                    default = None
                else:
                    default = op(self.default, other.default)
                return type(self)(mapping, self.dtype, default)
            else:
                return NotImplemented
        elif self._dtype_matches(type(other)):
            keys = self.keys()
            mapping = {k: op(self[k], other) for k in keys}
            if self.default is None:
                default = None
            else: 
                default = op(self.default, other)
            return type(self)(mapping, self.dtype, default)
        else:
            return NotImplemented

    def _cast(self, val):

        if type(val) != self.dtype:
            return self.dtype(val)
        else:
            return val

    def _dtype_matches(self, dtype):

        if self.dtype == dtype:
            return True
        else:
            return False
        
    @staticmethod
    def _inv(x):

        return 1.0 - x

    @staticmethod
    def _rtruediv(a, b):

        return b / a

    @staticmethod
    def _rpow(a, b):

        return b ** a


class FrozenNumDict(BaseNumDict, Mapping):
    """
    A frozen numerical dictionary.

    Supports various mathematical ops, but not in-place opreations. For 
    details, see numdicts module description.
    """

    __slots__ = ()


class NumDict(BaseNumDict, MutableMapping):
    """
    A mutable numerical dictionary.

    Supports various mathematical ops, including in-place operations. For 
    details, see numdicts module description.
    """

    __slots__ = ()

    @property
    def dtype(self):

        return self._dtype

    @dtype.setter
    def dtype(self, dtype):

        self._dtype = dtype
        for k, v in self._dict.items():
            self._dict[k] = self._cast(v)
        if self._default is not None:
            self._default = self._cast(self._default)

    @property
    def default(self):

        return self._default

    @default.setter
    def default(self, val):

        if val is None:
            self._default = None 
        else: 
            self._default = self._cast(val)

    def __setitem__(self, key, val):

        self._dict[key] = self._cast(val)

    def __delitem__(self, key):

        del self._dict[key]

    def __iadd__(self, other):

        return self.apply_iop(other, operator.add)

    def __isub__(self, other):

        return self.apply_iop(other, operator.sub)

    def __imul__(self, other):

        return self.apply_iop(other, operator.mul)

    def __itruediv__(self, other):

        return self.apply_iop(other, operator.truediv)

    def __ipow__(self, other):

        return self.apply_iop(other, operator.pow)

    def __iand__(self, other):

        return self.apply_iop(other, min)

    def __ior__(self, other):

        return self.apply_iop(other, max)

    def setdefault(self, key, default=None):
        """
        Return self[key]; on failure, return default and set it as value of key. 
        
        If default is None, but self.default is defined, will return 
        self.default and set self[key] = self.default. If no default is 
        available, will throw a value error.
        """

        if key in self:
            return self[key]
        else:
            if default is not None:
                self[key] = default
            elif default is None and self.default is not None:
                default = self.default
                self[key] = default
            else: 
                raise ValueError("No default value specified.")
            return default

    def squeeze(self):
        """
        Drop values that are close to self.default.
        
        Inplace operation.
        """

        if self.default is None:
            raise ValueError("Cannot squeeze numdict with no default.")

        keys = list(self.keys())
        for k in keys:
            if math.isclose(self[k], self.default):
                del self[k]

    def extend(self, *iterables, value=None):
        """
        Add each new item found in the union of iterables as a key to self.
        
        Inplace operation.

        For each item in the union of iterables, if the item does not already 
        have a set value in self, sets the item to have value `value`, which, 
        if None is passed, defaults to self.default, if defined, and 
        self.dtype() otherwise.
        """

        if value is not None:
            v = value
        elif self.default is not None:
            v = self.default
        else:
            v = self.dtype()

        for k in chain(*iterables):
            if k not in self:
                self[k] = v

    def keep(self, func):
        """
        Keep only the keys that are mapped to True by func.
        
        Inplace operation.

        Keys are kept iff func(key) is True.
        """

        keys = list(self.keys())
        for key in keys:
            if not func(key):
                del self[key]

    def drop(self, func):
        """
        Drop all keys that are mapped to True by func.
        
        Inplace operation.

        Keys are dropped iff func(key) is True.
        """

        keys = list(self.keys())
        for key in keys:
            if func(key):
                del self[key]

    def set_by(self, other, keyfunc):
        """Set self[k] = other[keyfunc(k)]"""

        for k in self:
            self[k] = other[keyfunc(k)]

    def apply_iop(self, other, op):
        """
        Apply binary op in place.
        
        Will attempt to coerce values of other to self.dtype. Defaults will be 
        updated as appropriate. In particular, if either self or other has no 
        default, self will have no default. Otherwise, the new default is 
        op(self.default, other.default).
        """

        if isinstance(other, BaseNumDict):
            if self._dtype_matches(other.dtype):
                keys = set(self.keys()) | set(other.keys())
                for k in keys:
                    self[k] = op(self[k], other[k])
                if self.default is None:
                    default = None
                elif other.default is None:
                    default = None
                else:
                    default = op(self.default, other.default) 
                self.default = default
                return self
            else:
                return NotImplemented
        elif self._dtype_matches(type(other)):
            for k, v in self.items():
                self[k] = op(v, other)
            if self.default is not None:
                self.default = op(self.default, other)
            return self
        else:
            return NotImplemented


#################
### Functions ###
#################


D = TypeVar("D", bound=BaseNumDict)
D1 = TypeVar("D1", bound=BaseNumDict)
D2 = TypeVar("D2", bound=BaseNumDict)


def restrict(d: D, container: Container) -> D:
    """Return a numdict whose keys are restricted by container."""

    mapping = {k: d[k] for k in d if k in container}

    return type(d)(mapping, d.dtype, d.default)


def keep(d: D, func: Callable[..., bool]):
    """Return a copy of d keeping only the keys mapped to True by func."""

    mapping = {k: d[k] for k in d if func(k)}

    return type(d)(mapping, d.dtype, d.default)


def drop(d: D, func: Callable[..., bool]):
    """Return a copy of d dropping the keys mapped to True by func."""

    mapping = {k: d[k] for k in d if func(k)}

    return type(d)(mapping, d.dtype, d.default)


def transform_keys(
    d: D, func: Callable[..., Hashable], *args, **kwds
) -> D:
    """
    Return a copy of d where each key is mapped to func(key, *args, **kwds).

    Warning: If function is not one-to-one wrt keys, output will be 
    corrupted. This is not checked.
    """

    mapping = {func(k, *args, **kwds): d[k] for k in d}

    return type(d)(mapping, d.dtype, d.default)


def threshold(d: D, th: numbers.Number) -> D:
    """
    Return a copy of d containing only values above theshold.
    
    Values below threshold are effectively sent to the default, if this is 
    defined.
    """

    mapping = {k: d[k] for k in d if th < d[k]}

    return type(d)(mapping, d.dtype, d.default)


def clip(d: D, low: numbers.Number = None, high: numbers.Number = None) -> D:
    """
    Return a copy of d with values clipped.
    
    dtype must define +/- inf values.
    """

    low = low or d.dtype("-inf")
    high = high or d.dtype("inf")

    mapping = {k: max(low, min(high, d[k])) for k in d}

    return type(d)(mapping, d.dtype, d.default)


def isclose(d1: D1, d2: D2) -> bool:
    """Return True if self is close to other in values."""
    
    _d = d1.apply_binary_op(d2, math.isclose)

    return all(_d.values())


def valuewise(op, d: D) -> numbers.Number:
    """
    Apply op to values of d.
    
    Output inherits dtype of d.
    """

    v = d.dtype()
    for item in d.values():
        v = op(v, item)

    return v


def val_sum(d: D) -> numbers.Number:
    """Return the sum of the values of d."""

    return valuewise(operator.add, d)


def elementwise(op, *ds: D, dtype=None) -> D:
    """
    Apply op element wise to sequence of numdicts ds.
    
    Value of dtype is inherited from the first d if None is passed.
    """

    _dtype = dtype or ds[0].dtype

    keys: set = set()
    keys.update(*ds)

    grouped: dict = {}
    defaults: list = []
    for d in ds:
        defaults.append(d.default)
        for k in keys:
            grouped.setdefault(k, []).append(d[k])

    return type(d)({k: op(grouped[k]) for k in grouped}, _dtype, op(defaults))


def ew_sum(*ds: D, dtype=None) -> D:
    """
    Elementwise sum of values in ds.
    
    Wraps elementwise().
    """

    return elementwise(sum, *ds, dtype=None)


def ew_max(*ds: D, dtype=None)  -> D:
    """
    Elementwise maximum of values in ds.
    
    Wraps elementwise().
    """

    return elementwise(max, *ds, dtype=None)


def boltzmann(d: D, t: float) -> D:
    """Construct a boltzmann distribution from d with temperature t."""

    output = NumDict()
    if len(d) > 0:
        numerators = math.e ** (d / t)
        denominator = val_sum(numerators)
        output |= numerators / denominator

    return type(d)(output)


def draw(d: D, k: int=1, val=1.0) -> D:
    """
    Draw k keys from numdict without replacement.
    
    If k >= len(d), returns a selection of all elements in d. Sampled elements 
    are given a value of 1.0 by default. Output inherits type, dtype and 
    default values from d.
    """

    pr = NumDict(d)
    output = NumDict()
    if len(d) > k:
        while len(output) < k:
            cs, ws = tuple(zip(*pr.items()))
            choices = random.choices(cs, weights=ws)
            output.extend(choices, value=val)
            pr.keep(output.__contains__)
    else:
        output.extend(d, value=val)
    
    return type(d)(output, d.dtype, d.default)
