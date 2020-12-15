"""Definitions for numerical dictionaries with autodiff support."""


__all__ = [
    # types
    "Op", "GradientOp", "D", 
    # decorators for registering new ops
    "op", "grad", 
    # classes
    "GradientTape", "NumDict", "MutableNumDict", 
    # functions from here onward
    "epsilon", "log", "exp", "sigmoid",
    "keep", "drop", "transform_keys",
    "threshold", "clip", "boltzmann", "draw",
    "by", "sum_by", "max_by",
    "ew_sum", "ew_mean", "ew_max", "ew_min",
    "constant", "freeze", "unfreeze", "is_close", "val_sum"
]


from typing import (
    Mapping, Any, Hashable, TypeVar, Type, Optional, Callable, Iterable, Dict, 
    List, Union, Tuple, Set, Container, overload, cast
)
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import chain
from functools import wraps
import math
import random
import operator


# Types for defining differentiable ops and their gradients.

Op = Callable[..., "NumDict"]
GradientOp = Callable[..., Tuple["NumDict", ...]]

# For a given mathematical operation, the type Op is for computing the 
# forward pass and the type GradientOp is for computing the backward pass.

# The argument signature of Op should roughly be 
#   *inputs: NumDict, **kwds: Any
# More precisely, inputs should be of length 1 at least and ops may be of 
# fixed arity. Any required arguments that are not of type D should be set as 
# named arguments.

# Likewise, the argument signature of GradientOp should roughly be:
#   grads: NumDict, *inputs: NumDict, **kwds: Any
# Here, *inputs, **kwds are the same as for Op, corresponding exactly to the 
# signature of the associated forward op. Finally, grads is the incoming 
# gradient for the backward pass.

# Mappings for recording and retrieving ops and their gradients. 
OPS: Dict[str, Op] = {}
GRADS: Dict[str, GradientOp] = {}

# Context variable for storing active gradient tapes.
TAPE: ContextVar = ContextVar("TAPE")


class GradientTapeError(Exception):
    """Raised when an inappropriate GradientTape event occurs."""
    pass


# Needs to be registered with pprint, awful to read w/ large numdicts.
@dataclass
class TapeCell(object):
    """A gradient tape entry."""

    value: "NumDict"
    op: str = ""
    operands: Tuple[int, ...] = ()
    kwds: dict = field(default_factory=dict) 


class GradientTape(object):
    """
    A simple gradient tape.

    Tracks diffable ops and computes forward and backward passes. Does not 
    support nesting.
    """

    # May support nesting and higher order derivatives. Investigate.

    __slots__ = ("_tape", "_index", "_token", "_recording", "_persistent")

    _tape: List[TapeCell] 
    _index: Dict[int, int]
    _token: Any
    _recording: bool
    _persistent: bool

    def __init__(self, persistent: bool = False) -> None:

        self._tape = []
        self._index = {}
        self._recording = False
        self._persistent = persistent

    def __repr__(self):

        s = "<{} persistent={} recording={} len={}>"
        name = type(self).__name__ 
        persistent = self.persistent
        recording = self._recording 
        length = len(self.data)
        
        return s.format(name, persistent, recording, length)

    def __enter__(self):

        try:
            TAPE.get()
        except LookupError:
            self._token = TAPE.set(self)
            self._recording = True
        else:
            raise GradientTapeError("Cannot stack tapes.")
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self._recording = False
        TAPE.reset(self._token)
    
    @property
    def persistent(self) -> bool:

        return self._persistent

    @property
    def data(self) -> List[TapeCell]:

        return self._tape

    def reset(self) -> None:
        """Reset tape."""

        if self._recording:
            raise GradientTapeError("Cannot reset while recording.")
        else:
            self._tape = []
            self._index = {}

    def register(
        self, 
        value: "NumDict", 
        op: str = "", 
        inputs: Tuple["NumDict", ...] = (), 
        kwds: dict = None
    ) -> None:

        if not self._recording:
            msg = "Cannot register object when not recording."
            raise GradientTapeError(msg)

        if kwds is None:
            kwds = {}

        # register any new operands
        for numdict in inputs:
            if id(numdict) not in self._index:
                self.register(numdict)

        operands = tuple(self._index[id(numdict)] for numdict in inputs)
        cell = TapeCell(value, op, operands, kwds)
        self._index[id(value)] = len(self._tape)
        self._tape.append(cell)

    def index(self, numdict: "NumDict") -> int:
        """Return the tape index at which numdict is registered."""

        return self._index[id(numdict)]

    @overload
    def forward(self, __1: int) -> "NumDict":
        ...

    @overload
    def forward(
        self, __1: int, __2: int, *indices: int
    ) -> Tuple["NumDict", ...]:
        ...

    def forward(self, *indices: int) -> Union[Tuple["NumDict", ...], "NumDict"]:
        """Perform a forward pass over the current tape."""

        if self._recording:
            msg = "Cannot compute forward pass while recording."
            raise GradientTapeError(msg)
        elif not self._persistent:
            msg = "Forward pass is not enabled on non-persistent tape."
            raise GradientTapeError(msg)

        self._index.clear()
        for i, entry in enumerate(self._tape):
            if entry.op == "":
                self._index[id(entry.value)] = i
            else:
                op = OPS[entry.op]
                inputs = (self._tape[i].value for i in entry.operands)
                output = op(*inputs, **entry.kwds) 
                self._index[id(output)] = i
                entry.value = output

        values = tuple(self._tape[i].value for i in indices)

        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    def backward(
        self, seed: int, indices: Set[int], seed_val: float = 1.0
    ) -> Dict[int, "NumDict"]: 
        """
        Perform a backward pass over the current tape.

        :param seed: Tape index seeding the backward pass.
        :param variables: A set of tape indices to be treated as variables in 
            the backward pass. Gradients will be calculated only for these 
            variables.
        :param seed_val: The seed value.
        """

        # Need to ensure correctness: the seed may not be a single value...
        # Perhaps should assume, if not single value, that we are 
        # differentiating against the sum. Need to think about correct 
        # implementation.

        if self._recording:
            msg = "Cannot compute backward pass while recording."
            raise GradientTapeError(msg)

        delta = {seed: NumDict(default=seed_val)}
        for i, entry in reversed(list(enumerate(self._tape))):
            delta.setdefault(i, NumDict(default=0.0))
            if entry.op:
                grad_op = GRADS[entry.op]
                inputs = (self._tape[k].value for k in entry.operands)
                grads = grad_op(delta[i], *inputs, **entry.kwds) 
                for j, k in enumerate(entry.operands):
                    if k in indices or self._tape[k].op != "":
                        delta.setdefault(k, NumDict(default=0.0))
                        delta[k] += grads[j]

        if not self.persistent:
            self.reset()

        return delta


    @overload
    def evaluate(self, __1: "NumDict") -> "NumDict":
        ...

    @overload
    def evaluate(
        self, __1: "NumDict", __2: "NumDict", *variables: "NumDict"
    ) -> Tuple["NumDict", ...]:
        ...

    def evaluate(
        self, *variables: "NumDict"
    ) -> Union["NumDict", Tuple["NumDict", ...]]:
        """Evaluate variables against current state of the tape."""
        
        indices = [self.index(var) for var in variables]

        return self.forward(*indices)

    @overload
    def gradients(
        self, output: "NumDict", variables: "NumDict"
    ) -> Tuple["NumDict", "NumDict"]:
        ...

    @overload
    def gradients(
        self, output: "NumDict", variables: "NumDict", forward: bool
    ) -> Tuple["NumDict", "NumDict"]:
        ...

    @overload
    def gradients(
        self, output: "NumDict", variables: Tuple["NumDict", ...]
    ) -> Tuple["NumDict", Tuple["NumDict", ...]]:
        ...

    @overload
    def gradients(
        self, output: "NumDict", variables: Tuple["NumDict", ...], forward: bool
    ) -> Tuple["NumDict", Tuple["NumDict", ...]]:
        ...

    def gradients(
        self, 
        output: "NumDict", 
        variables: Union["NumDict", Tuple["NumDict", ...]], 
        forward: bool = True
    ) -> Tuple["NumDict", Union["NumDict", Tuple["NumDict", ...]]]:
        """
        Compute gradients of variables against output.

        Accepts as variables a sequence of numdicts.

        If variables contains only one element, will return a single value. 
        Otherwise will return a tuple, with each element matching the 
        corresponding variables entry.

        :param output: Value against which to take gradients.
        :param variables: A sequence of numdicts containing variable values for 
            the backward pass. Gradients will be calculated only for these 
            values.
        :param forward: Whether to perform a forward pass prior to computing 
            gradients if self is persistent. By default, True.
        """

        if self._recording:
            msg = "Cannot compute gradients while recording."
            raise GradientTapeError(msg)

        index = self._index
        seed = index[id(output)]
        if isinstance(variables, tuple):
            indices = set(index[id(var)] for var in variables)
        else:
            indices = {index[id(variables)]}
        
        if self._persistent and forward:
            output = cast("NumDict", self.forward(seed))
        grads = self.backward(seed, indices)

        result: Union["NumDict", Tuple["NumDict", ...]]
        if isinstance(variables, tuple):
            return output, tuple(grads[index[id(var)]] for var in variables)
        else:
            return output, grads[index[id(variables)]]


def _register(
    value: "NumDict", op: str, inputs: Tuple["NumDict", ...], kwds: dict
) -> None:

    try:
        tape = TAPE.get()
    except LookupError:
        pass # Maybe print a warning? - Can
    else:
        if value is not NotImplemented:
            # _args must contain NumDicts only
            _inputs = []
            for x in inputs: 
                if isinstance(x, NumDict):
                    _input = x
                else:
                    _input = type(value)(default=x)
                _inputs.append(_input)
            tape.register(value, op, _inputs, kwds)


def op(func: Op) -> Op:
    """Decorator for registering differentiable mathematical ops."""

    OPS[func.__qualname__] = func

    @wraps(func)
    def wrapper(*inputs: "NumDict", **kwds: Any) -> "NumDict":
        """Compute op and register result in current active tape."""

        output = func(*inputs, **kwds) 
        _register(output, func.__qualname__, inputs, kwds)

        return output
    
    return wrapper


def grad(op: Op) -> Callable[[GradientOp], GradientOp]:
    """Decorator for registering op gradient functions."""

    def wrapper(func: GradientOp) -> GradientOp:

        GRADS[op.__qualname__] = func

        return func

    return wrapper


class NumDict(Mapping[Hashable, float]):
    """
    A numerical dictionary (immutable).

    Maps keys to numerical values and supports various mathematical ops. 

    May have a default value. Defaults are updated appropriately when 
    mathematical ops are applied. Querying a missing key returns the default 
    value (if it is defined), but the missing key is not added to the queried 
    numdict. 

    If a key is explicitly passed to a numdict, it is considered an explicit 
    member of the numdict. If default value is defined, any key that is not an 
    explicit member is considered an implicit member. Containment checks will 
    only succeed for explicit members.
    """

    __slots__ = ("_dict", "_default")

    _dict: Dict[Hashable, float]
    _default: Optional[float]

    def __init__(
        self, 
        data: Mapping[Hashable, Union[float, int]] = None, 
        default: Union[float, int] = None
    ) -> None:
        """
        Initialize a new NumDict instance.

        :param data: Mapping from which to populate values of self.
        :param default: Default value for keys not in self.
        """

        if data is None:
            data = {}

        self._dict = {k: float(data[k]) for k in data}
        self._default = float(default) if default is not None else None

    @property
    def default(self) -> Optional[float]:
        """Default value of NumDict instance."""

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

        Warning: If self.default is not None, self[key] may return a value 
        when `key in self` returns False. Do not use self[key] to check for 
        containment.
        """

        return key in self._dict

    def __getitem__(self, key) -> float:
        """
        Return the value explicitly or implicitly associated with key.

        Note that self[key] will succeed for implicit members. It is dangerous 
        to use d[key] to check for (explicit) membership.
        """

        try:
            return self._dict[key]
        except KeyError:
            if self._default is None:
                raise
            else:
                return self._default

    def __neg__(self) -> "NumDict":

        return self.unary(operator.neg)

    __neg__ = op(__neg__)

    @staticmethod
    @grad(__neg__)
    def _grad_neg(grads: "NumDict", d: "NumDict") -> Tuple["NumDict"]:

        return (grads * constant(d, -1.0),)

    def __abs__(self) -> "NumDict":

        return self.unary(operator.abs)
    
    __abs__ = op(__abs__)

    @staticmethod
    @grad(__abs__)
    def _abs_grad(grads: "NumDict", d: "NumDict") -> Tuple["NumDict"]:

        return (grads * ((2 * (d > 0)) - 1),)

    def __eq__( # type: ignore[override]
        self, other: Union["NumDict", float, int]
    ) -> "NumDict":

        return self.binary(other, operator.eq)

    def __ne__( # type: ignore[override]
        self, other: Union["NumDict", float, int]
    ) -> "NumDict":

        return self.binary(other, operator.ne)

    def __lt__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self.binary(other, operator.lt)

    def __le__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self.binary(other, operator.le)

    def __gt__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self.binary(other, operator.gt)

    def __ge__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self.binary(other, operator.ge)

    def __add__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self.binary(other, operator.add)

    __add__ = op(__add__)

    @staticmethod
    @grad(__add__)
    def _grad_add(grads, d1, d2):
        
        return (grads * constant(d1, 1), grads * constant(d2, 1))    

    def __sub__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self.binary(other, operator.sub)

    __sub__ = op(__sub__)

    @staticmethod
    @grad(__sub__)
    def _grad_sub(grads, d1, d2):
        
        return (grads * constant(d1, 1), grads * constant(d2, -1))

    def __mul__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self.binary(other, operator.mul)

    __mul__ = op(__mul__)

    @staticmethod
    @grad(__mul__)
    def _grad_mul(grads, d1, d2):

        return (grads * d2, grads * d1)

    def __truediv__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self.binary(other, operator.truediv)

    __truediv__ = op(__truediv__)

    @staticmethod
    @grad(__truediv__)
    def _grad_truediv(grads, d1, d2):

        return (grads / d2, grads * (- d1) / (d2 ** 2))

    def __pow__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self.binary(other, operator.pow)

    __pow__ = op(__pow__)

    @staticmethod
    @grad(__pow__)
    def _grad_pow(grads, d1, d2):

        return (grads * d2 * (d1 ** (d2 - 1)), grads * log(d1) * (d1 ** d2))

    def __radd__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self + other

    def __rsub__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return (- self) + other

    def __rmul__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self * other

    def __rtruediv__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self.binary(other, self._rtruediv)

    __rtruediv__ = op(__rtruediv__)

    @staticmethod
    @grad(__rtruediv__)
    def _grad_rtruediv(grads, d1, d2): # TODO: check correctness! - Can

        return (grads * (- d2) / (d1 ** 2), grads / d1) 

    def __rpow__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self.binary(other, self._rpow)

    __rpow__ = op(__rpow__)

    @staticmethod
    @grad(__rpow__)
    def _grad_rpow(grads, d1, d2): # TODO: check correctness! - Can

        return (grads * log(d2) * (d2 ** d1), grads * d1 * (d2 ** (d1 - 1)))

    def unary(self, op: Callable[[float], float]) -> "NumDict":
        """
        Apply op to each element of self.

        Returns a new numdict.        
        """

        mapping = {k: op(self._dict[k]) for k in self}

        default: Optional[float]
        if self.default is not None:
            default = op(self.default)
        else:
            default = None

        return NumDict(mapping, default)

    def binary(
        self, 
        other: Union["NumDict", float, int], 
        op: Callable[[float, float], float]
    ) -> "NumDict":
        """
        Apply binary op to each element of self and other.

        Returns a new numdict.
        
        If other is a constant c, acts as if other[key] = c.

        If both self and other define defaults, the new default is equal to
        op(self_default, other_default). Otherwise no default is defined.
        """

        _other: "NumDict"
        if isinstance(other, (float, int)):
            _other = NumDict(default=other)
        elif isinstance(other, NumDict):
            _other = cast(NumDict, other)
        else:
            return NotImplemented

        keys = set(self.keys()) | set(_other.keys())
        mapping = {k: op(self[k], _other[k]) for k in keys}

        default: Optional[float]    
        if self.default is None:
            default = None
        elif _other.default is None:
            default = None
        else:
            default = op(self.default, _other.default)
        
        return NumDict(mapping, default)

    @staticmethod
    def _rtruediv(a, b):

        return b / a

    @staticmethod
    def _rpow(a, b):

        return b ** a


class MutableNumDict(NumDict):

    __slots__ = ()

    @property
    def default(self) -> Optional[float]:
        """Default value of NumDict instance."""

        return self._default

    @default.setter
    def default(self, val: Union[float, int]):

        self._default = float(val)

    def __setitem__(self, key: Hashable, val: Union[float, int]) -> None:

        self._dict[key] = float(val)

    def __delitem__(self, key: Hashable) -> None:

        del self._dict[key]

    def __iadd__(self, other: Union[NumDict, float, int]) -> "MutableNumDict":

        return self._inplace(other, operator.add)

    def __isub__(self, other: Union[NumDict, float, int]) -> "MutableNumDict":

        return self._inplace(other, operator.sub)

    def __imul__(self, other: Union[NumDict, float, int]) -> "MutableNumDict":

        return self._inplace(other, operator.mul)

    def __itruediv__(
        self, other: Union[NumDict, float, int]
    ) -> "MutableNumDict":

        return self._inplace(other, operator.truediv)

    def __ipow__(self, other: Union[NumDict, float, int]) -> "MutableNumDict":

        return self._inplace(other, operator.pow)

    def max(self, other: Union[NumDict, float, int]) -> "MutableNumDict":
        """Compute elementwise maximums in-place and return self."""

        return self._inplace(other, max)

    def min(self, other: Union[NumDict, float, int]) -> "MutableNumDict":
        """Compute elementwise minimums in-place and return self."""

        return self._inplace(other, min)

    def update(self, numdict: NumDict, update_default: bool = False) -> None:
        """
        Update self with keys and values of mapping.
        
        Optionally, also update default.
        """

        for k, v in numdict.items():
            self.__setitem__(k, v)
        if update_default:
            self.default = numdict.default

    def clear(self, clear_default: bool = False) -> None:
        """
        Remove all explicit members of self.
        
        Optionally, also clear default.
        """

        self._dict.clear()
        if clear_default:
            self.default = None

    def squeeze(self, default: float = None):
        """
        Drop explicit values that are close to self.default (inplace).
        
        :param default: Default value to assume, if self.default is None. If 
            provided when self.default is defined, will be ignored.
        """

        if self.default is not None:
            default = self.default
        elif default is not None:
            pass
        else:
            raise ValueError("Cannot squeeze numdict with no default.")

        keys = list(self.keys())
        for k in keys:
            if math.isclose(self[k], default):
                del self[k]

    def extend(self, *iterables, value=None):
        """
        Add each new item found in the union of iterables as a key to self.
        
        Inplace operation.

        For each item in the union of iterables, if the item does not already 
        have a set value in self, sets the item to have value `value`, which, 
        if None is passed, defaults to self.default, if defined, and 0.0 
        otherwise.
        """

        if value is not None:
            v = value
        elif self.default is not None:
            v = self.default
        else:
            v = 0.0

        for k in chain(*iterables):
            if k not in self:
                self[k] = v

    def keep(self, func=None, keys=None):
        """
        Keep only the desired keys as explicit members (inplace).

        Keys are kept iff func(key) is True or key in container is True.
        """

        if func is None and keys is None:
            raise ValueError("Must pass at least one of func or keys.")

        keys = list(self.keys())
        for key in keys:
            keep = func is not None and func(key)
            keep |= keys is not None and key in keys
            if not keep:
                del self[key]

    def drop(self, func=None, keys=None):
        """
        Drop unwanted explicit members (inplace).

        Keys are dropped iff func(key) is True or key in container is True.
        """

        if func is None and keys is None:
            raise ValueError("Must pass at least one of func or keys.")

        keys = list(self.keys())
        for key in keys:
            drop = func is not None and func(key)
            drop |= keys is not None and key in container
            if drop:
                del self[key]

    def set_by(self, other, keyfunc):
        """Set self[k] = other[keyfunc(k)]"""

        for k in self:
            self[k] = other[keyfunc(k)]

    def _inplace(
        self, other: Union[NumDict, float], op: Callable[[float, float], float]
    ) -> "MutableNumDict":
        """
        Apply binary op in place.
        
        Defaults will be updated as appropriate. In particular, if either self 
        or other has no default, self will have no default. Otherwise, the new 
        default is op(self.default, other.default).
        """

        _other: NumDict
        if isinstance(other, (float, int)):
            _other = NumDict(default=other)
        elif isinstance(other, NumDict):
            _other = cast(NumDict, other)
        else:
            return NotImplemented

        keys = set(self.keys()) | set(_other.keys())
        for k in keys:
            self[k] = op(self[k], _other[k])
        
        if self.default is None:
            default = None
        elif _other.default is None:
            default = None
        else:
            default = op(self.default, _other.default) 
        self._default = default
        
        return self


#################
### Functions ###
#################


D = Union[MutableNumDict, NumDict]


### Basic Ops ###


def epsilon():
    """A very small value (1e-07),"""

    return 1e-07


@op
def exp(d: D) -> NumDict:
    """Apply exponentiation elementwise to d."""

    return math.e ** d

@grad(exp)
def _grad_exp(grads, d):

    return (grads * exp(d),)


def _log(x):

    try:
        return math.log(x)
    except ValueError:
        return float("nan")

@op
def log(d: D) -> NumDict:
    """Apply the log function elementwise to d."""
    
    return d.unary(_log)

@grad(log)
def _grad_log(grads: NumDict, d: D) -> Tuple[NumDict]:

    return (grads / d,)


# no need for grad definition as defined using diffable ops.
def sigmoid(d: D) -> NumDict:
    """Apply the logistic function elementwise to d."""

    return 1 / (1 + exp(-d))


# needs grad definition
def keep(
    d: D, func: Callable[..., bool] = None, keys: Container = None
) -> NumDict:
    """
    Return a copy of d keeping only the desired keys. 
    
    Keys are kept iff func(key) is True or key in container is True.
    """

    if func is None and keys is None:
        raise ValueError("At least one of func or keys must not be None.")

    mapping = {
        k: d[k] for k in d 
        if (func is not None and func(k)) or (keys is not None and k in keys)
    }

    return NumDict(mapping, d.default)


# needs grad definition
def drop(
    d: D, func: Callable[..., bool] = None, keys: Container = None
) -> NumDict:
    """
    Return a copy of d dropping unwanted keys. 
    
    Keys are dropped iff func(key) is True or key in container is True.
    """

    if func is None and keys is None:
        raise ValueError("At least one of func or keys must not be None.")

    mapping = {
        k: d[k] for k in d 
        if (func is not None and not func(k)) or 
        (keys is not None and k not in keys)
    }

    return NumDict(mapping, d.default)


@op
def set_by(
    target: D, source: D, *, keyfunc: Callable[..., Hashable]
) -> NumDict:
    """
    Construct a numdict mapping target keys to matching values in source. 
    
    For each key in source, output[key] = source[keyfunc(key)]. Defaults are 
    discarded.
    """

    return NumDict({k: source[keyfunc(k)] for k in target}, None)

@grad(set_by)
def _grad_set_by(grads, target, source, *, keyfunc):

    return (grads * NumDict(default=0), sum_by(grads, keyfunc=keyfunc))


# needs grad definition
def transform_keys(d: D, *, func: Callable[..., Hashable], **kwds) -> NumDict:
    """
    Return a copy of d where each key is mapped to func(key, **kwds).

    Warning: If function is not one-to-one wrt keys, will raise ValueError.
    """

    mapping = {func(k, **kwds): d[k] for k in d}

    if len(d) != len(mapping):
        raise ValueError("Func must be one-to-one on keys of arg d.")

    return NumDict(mapping, d.default)


# needs grad definition? is this even differentiable?
def threshold(d: D, *, th: float) -> NumDict:
    """
    Return a copy of d containing only values above theshold.
    
    If the default is below threshold, it is set to None in the output.
    """

    mapping = {k: d[k] for k in d if th < d[k]}
    if d.default is not None:
        default = d.default if th < d.default else None 

    return NumDict(mapping, default)


# needs grad definition
def clip(d: D, low: float = None, high: float = None) -> NumDict:
    """
    Return a copy of d with values clipped.
    
    dtype must define +/- inf values.
    """

    low = low or float("-inf")
    high = high or float("inf")

    mapping = {k: max(low, min(high, d[k])) for k in d}

    return NumDict(mapping, d.default)


# needs grad definition, not to mention reworking
def boltzmann(d: D, t: float) -> NumDict:
    """Construct a boltzmann distribution from d with temperature t."""

    output = MutableNumDict(default=0.0)
    if len(d) > 0:
        numerators = exp(d / t)
        denominator = val_sum(numerators)
        output.max(numerators / denominator)

    return NumDict(output)


# needs grad definition
def draw(d: D, k: int=1, val=1.0) -> NumDict:
    """
    Draw k keys from numdict without replacement.
    
    If k >= len(d), returns a selection of all elements in d. Sampled elements 
    are given a value of 1.0 by default. Output inherits type, dtype and 
    default values from d.
    """

    pr = MutableNumDict(d)
    output = MutableNumDict()
    if len(d) > k:
        while len(output) < k:
            cs, ws = tuple(zip(*pr.items()))
            choices = random.choices(cs, weights=ws)
            output.extend(choices, value=val)
            pr.keep(output.__contains__)
    else:
        output.extend(d, value=val)
    
    return NumDict(output, d.default)


### By Ops ###


def by(
    d: D, 
    op: Callable[..., float],
    keyfunc: Callable[..., Hashable], 
    **kwds: Any
) -> NumDict:
    """
    Compute op over elements grouped by keyfunc.
    
    Key should be a function mapping each key in self to a grouping key. New 
    keys are determined based on the result of keyfunc(k, **kwds), where 
    k is a key from d. Defaults are discarded.
    """

    _d: Dict[Hashable, List[float]] = {}
    for k, v in d.items():
        _d.setdefault(keyfunc(k, **kwds), []).append(v)
    mapping = {k: op(v) for k, v in _d.items()}

    return NumDict(mapping)


@op
def sum_by(d: D, *, keyfunc: Callable[..., Hashable], **kwds: Any) -> NumDict:
    """
    Sum the values of d grouped by keyfunc.
    
    Maps each l in the range of keyfunc to the sum of all d[k] such that 
    keyfunc(k) is equal to l. See by() for further details.
    """

    return by(d, sum, keyfunc, **kwds)

@grad(sum_by)
def _grad_sum_by(grads, d, *, keyfunc):

    return (NumDict({k: grads[keyfunc(k)] for k in d}),)


#needs grad definition
@op
def max_by(d: D, *, keyfunc: Callable[..., Hashable], **kwds: Any) -> NumDict:
    """
    Find maximum values in d grouped by keyfunc.
    
    Maps each l in the range of keyfunc to the max of all d[k] such that 
    keyfunc(k) is equal to l. See by() for further details.
    """

    return by(d, max, keyfunc, **kwds)

@grad(max_by)
def _grad_max_by(grads, d, *, keyfunc):

    _isclose = math.isclose
    y = max_by(d, keyfunc=keyfunc) # Should block tape registration.
    arg_max = {k for k, v in d.items() if _isclose(v, y[keyfunc(k)])}
    grad_max = NumDict({k: grads[keyfunc(k)] if k in arg_max else 0 for k in d})

    return (grad_max,)


### Elementwise Variadic Ops ###


def elementwise(op: Callable[..., float], *ds: D) -> NumDict:
    """
    Apply op elementwise to a sequence of numdicts.
    
    If any numdict in ds has None default, then default is None, otherwise the 
    new default is calculated by running op on all defaults.
    """

    keys: set = set()
    keys.update(*ds)

    grouped: dict = {}
    defaults: list = []
    for d in ds:
        defaults.append(d.default)
        for k in keys:
            grouped.setdefault(k, []).append(d[k])
    
    if any([d is None for d in defaults]):
        default = None
    else:
        default = op(defaults)

    return NumDict({k: op(grouped[k]) for k in grouped}, default)


# needs grad definition
def ew_sum(*ds: D) -> NumDict:
    """
    Elementwise sum of values in ds.
    
    Wraps elementwise().
    """

    return elementwise(sum, *ds)


# needs grad definition
def ew_mean(*ds: D) -> NumDict:
    """
    Elementwise sum of values in ds.
    
    Wraps elementwise().
    """

    return elementwise(sum, *ds) / len(ds)


# needs grad definition
def ew_max(*ds: D)  -> NumDict:
    """
    Elementwise maximum of values in ds.
    
    Wraps elementwise().
    """

    return elementwise(max, *ds)


# needs grad definition
def ew_min(*ds: D) -> NumDict:
    """
    Elementwise maximum of values in ds.
    
    Wraps elementwise().
    """

    return elementwise(min, *ds)


### Non-differentiable functions ###


def constant(d: D, value: float) -> NumDict:
    """Return a copy of d where all values are set to a constant."""

    mapping = {k: value for k in d._dict}

    default: Optional[float]
    if d.default is not None:
        default = value
    else:
        default = None

    return type(d)(mapping, default)


def freeze(d: MutableNumDict) -> NumDict:
    """Return a frozen copy of d."""

    return NumDict(d, d.default)


def unfreeze(d: NumDict) -> MutableNumDict:
    """Return a mutable copy of d."""

    return MutableNumDict(d, d.default)


def isclose(d1: D, d2: D) -> bool:
    """Return True if self is close to other in values."""
    
    _d = d1.binary(d2, math.isclose)

    return all(_d.values())


def exponential_moving_avg(d: D, *ds: D, alpha: float) -> List[NumDict]:
    """Given a sequence of numdicts, return a smoothed sequence."""

    avg = [d]
    for _d in ds:
        avg.append(alpha * _d + (1 - alpha) * avg[-1])

    return avg


def tabulate(*ds: D) -> Dict[Hashable, List[float]]:
    """
    Tabulate data from a sequence of numdicts.

    Produces a dictionary inheriting its keys from ds, and mapping each key to 
    a list such that the ith value of the list is equal to ds[i][k].
    """

    tabulation: Dict[Hashable, List[float]] = {}
    for d in ds:
        for k, v in d.items():
            l = tabulation.setdefault(k, [])
            l.append(v)

    return tabulation


def valuewise(
    op: Callable[[float, float], float], d: D, initial: float
) -> float:
    """Recursively apply commutative binary op to values of d."""

    v = initial
    for item in d.values():
        v = op(v, item)

    return v


def val_sum(d: D) -> Any:
    """Return the sum of the values of d."""

    return valuewise(operator.add, d, 0.0)
