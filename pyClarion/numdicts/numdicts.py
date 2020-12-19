"""Basic definitions for numerical dictionaries with autodiff support."""


# TODO: GradientOps may not handle defaults correctly! Check and correct. - Can


__all__ = [
    "Op", "GradientOp", "D", "GradientTape", "NumDict", "MutableNumDict", 
    "record_call", "register_op", "register_grad"
]


from typing import (
    Mapping, Any, Hashable, TypeVar, Type, Optional, Callable, Iterable, Dict, 
    List, Union, Tuple, Set, Container, Iterator, overload, cast
)
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import chain
from functools import wraps
import math
import random
import operator


D = TypeVar("D", bound=Union["NumDict", "MutableNumDict"])

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


# Needs to have a nice repr, awful to read w/ large numdicts.
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
    # Seems like it can work if all supported ops have grad functions that are 
    # themselves diffable... Test it. - Can

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


def record_call(
    op: Callable, 
    value: "NumDict", 
    inputs: Tuple[Union["NumDict", float, int], ...], 
    kwds: dict
) -> None:
    """Record a call to an op with autodiff support in current active tape."""

    try:
        tape = TAPE.get()
    except LookupError:
        pass 
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
            tape.register(value, op.__qualname__, _inputs, kwds)


OpVar = TypeVar("OpVar", bound=Op)
def register_op(func: OpVar) -> OpVar:
    """Decorator for registering ops with autodiff support."""

    OPS[func.__qualname__] = func

    return func


GradientOpVar = TypeVar("GradientOpVar", bound=GradientOp)
def register_grad(op: Op) -> Callable[[GradientOp], GradientOp]:
    """Decorator for registering op gradient functions."""

    def wrapper(func: GradientOpVar) -> GradientOpVar:

        GRADS[op.__qualname__] = func

        return func

    return wrapper


# Ideally, NumDict would be covariant in its keys. But this is not compatible 
# with Mapping. - Can
class NumDict(Mapping[Any, float]):
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

    _dict: Dict[Any, float]
    _default: Optional[float]

    def __init__(
        self, 
        data: Mapping[Any, Union[float, int]] = None, 
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

    def __str__(self) -> str:

        fmtargs = type(self).__name__, str(self._dict), self._default

        return "{}({}, default={})".format(*fmtargs)

    def __repr__(self) -> str:

        fmtargs = type(self).__name__, repr(self._dict), self._default

        return "{}({}, default={})".format(*fmtargs)

    def __len__(self) -> int:

        return len(self._dict)

    def __iter__(self) -> Iterator[Hashable]:

        yield from iter(self._dict)

    def __contains__(self, key: Any) -> bool:
        """
        Return True iff key is explicitly set in self.

        Warning: If self.default is not None, self[key] may return a value 
        when `key in self` returns False. Do not use self[key] to check for 
        containment.
        """

        return key in self._dict

    def __getitem__(self, key: Any) -> float:
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

    @register_op
    def __neg__(self) -> "NumDict":

        value = self._unary(operator.neg)
        record_call(self.__neg__, value, (self,), {})

        return value

    @staticmethod
    @register_grad(__neg__)
    def _grad_neg(grads: "NumDict", d: "NumDict") -> Tuple["NumDict"]:

        return (grads * d.constant(val=-1.0),)

    @register_op
    def __abs__(self) -> "NumDict":

        value = self._unary(operator.abs)
        record_call(self.__abs__, value, (self,), {})

        return value
    
    @staticmethod
    @register_grad(__abs__)
    def _abs_grad(grads: "NumDict", d: "NumDict") -> Tuple["NumDict"]:

        return (grads * ((d ** 2) ** 0.5),)

    def __eq__( # type: ignore[override]
        self, other: Union["NumDict", float, int]
    ) -> "NumDict":

        return self._binary(other, operator.eq)

    def __ne__( # type: ignore[override]
        self, other: Union["NumDict", float, int]
    ) -> "NumDict":

        return self._binary(other, operator.ne)

    def __lt__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self._binary(other, operator.lt)

    def __le__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self._binary(other, operator.le)

    def __gt__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self._binary(other, operator.gt)

    def __ge__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self._binary(other, operator.ge)

    @register_op
    def __add__(self, other: Union["NumDict", float, int]) -> "NumDict":

        value = self._binary(other, operator.add)
        record_call(self.__add__, value, (self, other), {})
        
        return value

    @staticmethod
    @register_grad(__add__)
    def _grad_add(
        grads: "NumDict", d1: "NumDict", d2: "NumDict"
    ) -> Tuple["NumDict", ...]:
        
        return (grads * d1.constant(val=1), grads * d2.constant(val=1))    

    @register_op
    def __sub__(self, other: Union["NumDict", float, int]) -> "NumDict":

        value = self._binary(other, operator.sub)
        record_call(self.__sub__, value, (self, other), {})

        return value

    @staticmethod
    @register_grad(__sub__)
    def _grad_sub(
        grads: "NumDict", d1: "NumDict", d2: "NumDict"
    ) -> Tuple["NumDict", ...]:
        
        return (grads * d1.constant(val=1), grads * d2.constant(val=-1))

    @register_op
    def __mul__(self, other: Union["NumDict", float, int]) -> "NumDict":

        value = self._binary(other, operator.mul)
        record_call(self.__mul__, value, (self, other), {})

        return value

    @staticmethod
    @register_grad(__mul__)
    def _grad_mul(
        grads: "NumDict", d1: "NumDict", d2: "NumDict"
    ) -> Tuple["NumDict", ...]:

        return (grads * d2, grads * d1)

    @register_op
    def __truediv__(self, other: Union["NumDict", float, int]) -> "NumDict":

        value = self._binary(other, operator.truediv)
        record_call(self.__truediv__, value, (self, other), {})

        return value

    @staticmethod
    @register_grad(__truediv__)
    def _grad_truediv(
        grads: "NumDict", d1: "NumDict", d2: "NumDict"
    ) -> Tuple["NumDict", ...]:

        return (grads / d2, grads * (- d1) / (d2 ** 2))

    @register_op
    def __pow__(self, other: Union["NumDict", float, int]) -> "NumDict":

        value = self._binary(other, operator.pow)
        record_call(self.__pow__, value, (self, other), {})

        return value

    @staticmethod
    @register_grad(__pow__)
    def _grad_pow(
        grads: "NumDict", d1: "NumDict", d2: "NumDict"
    ) -> Tuple["NumDict", ...]:

        return (grads * d2 * (d1 ** (d2 - 1)), grads * d1.log() * (d1 ** d2))

    def __radd__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self + other

    def __rsub__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return (- self) + other

    def __rmul__(self, other: Union["NumDict", float, int]) -> "NumDict":

        return self * other

    @register_op
    def __rtruediv__(self, other: Union["NumDict", float, int]) -> "NumDict":

        value = self._binary(other, self._rtruediv)
        record_call(self.__rtruediv__, value, (self, other), {})

        return value

    @staticmethod
    @register_grad(__rtruediv__)
    def _grad_rtruediv(
        grads: "NumDict", d1: "NumDict", d2: "NumDict"
    ) -> Tuple["NumDict", ...]: # TODO: check correctness! - Can

        return (grads * (- d2) / (d1 ** 2), grads / d1) 

    @register_op
    def __rpow__(self, other: Union["NumDict", float, int]) -> "NumDict":

        value = self._binary(other, self._rpow)
        record_call(self.__rpow__, value, (self, other), {})
        
        return value

    @staticmethod
    @register_grad(__rpow__)
    def _grad_rpow(
        grads: "NumDict", d1: "NumDict", d2: "NumDict"
    ) -> Tuple["NumDict", ...]: # TODO: check correctness! - Can

        return (grads * d2.log() * (d2 ** d1), grads * d1 * (d2 ** (d1 - 1)))

    # TODO: Is it necessary to register constant as an op? - Can 
    @register_op
    def constant(self, *, val: float) -> "NumDict":
        """
        Return a copy of d where all values are set to a constant.
        
        The resulting numdict has a default (set to value) iff self has a 
        default.
        """

        mapping = {k: val for k in self._dict}

        default: Optional[float]
        if self.default is not None:
            default = val
        else:
            default = None

        return NumDict(mapping, default)

    @staticmethod
    @register_grad(constant)
    def _grad_constant(
        grads: "NumDict", d: "NumDict", *, value: float
    ) -> Tuple["NumDict", ...]:

        return (grads * d.constant(val=0),)


    @register_op
    def exp(self) -> "NumDict":
        """Apply e-base exponentiation elementwise to d."""

        value = self._unary(math.exp)
        record_call(self.exp, value, (self,), {})
        
        return value

    @staticmethod
    @register_grad(exp)
    def _grad_exp(grads: "NumDict", d: "NumDict") -> Tuple["NumDict", ...]:

        return (grads * d.exp(),)

    @register_op
    def log(self) -> "NumDict":
        """Apply the log function elementwise to d."""
        
        value = self._unary(self._log)
        record_call(self.log, value, (self,), {})

        return value

    @staticmethod
    @register_grad(log)
    def _grad_log(grads: "NumDict", d: "NumDict") -> Tuple["NumDict", ...]:

        return (grads / d,)

    def _unary(self, op: Callable[[float], float]) -> "NumDict":
        """
        Apply op to each element of self.

        Returns a new numdict.        
        """

        mapping = {k: op(v) for k, v in self.items()}

        default: Optional[float]
        if self.default is not None:
            default = op(self.default)
        else:
            default = None

        return NumDict(mapping, default)

    def _binary(
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
            _other = other
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

    @staticmethod
    def _log(x):

        try:
            return math.log(x)
        except ValueError:
            return float("nan")


class MutableNumDict(NumDict):
    """
    A mutable numerical dictionary.

    Supports various inplace methods, in addition to basic NumDict methods. 
    Inplace methods are assumed undifferentiable and will not be tracked by 
    GradientTape instances.

    Note that non-inplace methods (+, -, *, /, etc.) always return NumDicts and 
    (even when the operands are themselves MutableNumDicts).
    """

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

    def update(
        self, numdict: NumDict, update_default: bool = False
    ) -> "MutableNumDict":
        """
        Update self with keys and values of mapping.
        
        Optionally, also update default.
        """

        for k, v in numdict.items():
            self.__setitem__(k, v)
        if update_default:
            self.default = numdict.default

        return self

    def clear(self, clear_default: bool = False) -> "MutableNumDict":
        """
        Remove all explicit members of self.
        
        Optionally, also clear default.
        """

        self._dict.clear()
        if clear_default:
            self.default = None

        return self

    def clearupdate(
        self, numdict: NumDict, update_default: bool = False
    ) -> "MutableNumDict":
        """
        Update self with keys and values of mapping after first clearing self.
        
        Optionally, also update default.
        """

        self.clear()
        self.update(numdict, update_default=update_default)

        return self

    def squeeze(self, default: float = None) -> "MutableNumDict":
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

        return self

    def extend(
        self, *iterables: Iterable[Hashable], value: float = None
    ) -> "MutableNumDict":
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

        return self

    def keep(
        self, 
        func: Callable[..., bool] = None, 
        keys: Container[Hashable] = None,
        **kwds: Any
    ) -> "MutableNumDict":
        """
        Keep only the desired keys as explicit members (inplace).

        Keys are kept iff func(key, **kwds) or key in container is True.
        """

        if func is None and keys is None:
            raise ValueError("Must pass at least one of func or keys.")

        self_keys = list(self.keys())
        for key in self_keys:
            func_success = func is not None and func(key, **kwds)
            key_success = keys is not None and key in keys
            if not (func_success or key_success):
                del self[key]

        return self

    def drop(
        self, 
        func: Callable[..., bool] = None, 
        keys: Container[Hashable] = None,
        **kwds: Any
    ) -> "MutableNumDict":
        """
        Drop unwanted explicit members (inplace).

        Keys are dropped iff func(key, **kwds) or key in container is True.
        """

        if func is None and keys is None:
            raise ValueError("Must pass at least one of func or keys.")

        _keys = list(self.keys())
        for key in _keys:
            func_success = func is not None and func(key, **kwds)
            key_success = keys is not None and key in keys
            if func_success or key_success:
                del self[key]

        return self

    def set_by(
        self, other: NumDict, keyfunc: Callable[..., Hashable], **kwds: Any
    ) -> "MutableNumDict":
        """Set self[k] = other[keyfunc(k, **kwds)]"""

        for k in self:
            self[k] = other[keyfunc(k, **kwds)]

        return self

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
