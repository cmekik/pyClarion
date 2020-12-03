"""
Provides support for basic native automatic diffferentiation in pyClarion.

Defines the `diffable`, `GradientTape`, and `DiffableNumDict` classes, which 
support automatic reverse-mode differentiation.

The general implementation is based on CMU autodiff lecture notes:
http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf

The GradientTape design is based on the tensorflow 2 GradientTape:
https://www.tensorflow.org/api_docs/python/tf/GradientTape
"""


__all__ = ["diffable", "GradientTape", "DiffableNumDict"]


from .numdicts import NumDict

from itertools import chain
from math import log
from contextvars import ContextVar
from typing import (
    Tuple, List, Mapping, Any, Union, Callable, TypeVar, Hashable, Optional, 
    Dict, Iterable, Type, Set, Mapping, Sequence, cast
)
import operator


V = TypeVar(
    "V", 
    bound=Union[List["diffable"], Tuple["diffable", ...], "DiffableNumDict"]
)


# A context variable for storing gradient tapes.
tape_ctx: ContextVar = ContextVar("tape_ctx") # actually List["diffable"]


class diffable(object):
    """
    A floating point value amenable to reverse-mode gradient computations.

    Collaborates with GradientTape. Does not expose the full float inteface.
    
    By default, supports backpropagation with the following operations: abs, +, 
    -, *, /, **, &, |. The bitwise operations & and | are interpreted as fuzzy 
    logical operators. They compute min and max respectively.

    Other differentiable operators may be registered as needed.

    Comparison functions ==, <, <=, >, >= are also defined.
    """

    __slots__ = ("_val", "_op", "_operands", "_index")

    _basic_ops = {
        "", "neg", "abs" "add", "sub", "mul", "truediv", "pow", "and", "or"
    }

    ops: Dict[str, Callable[..., float]] = {
        "neg": operator.__neg__,
        "abs": operator.__abs__,
        "add": operator.__add__,
        "sub": operator.__sub__,
        "mul": operator.__mul__,
        "truediv": operator.__truediv__,
        "pow": operator.__pow__,
        "and": operator.__and__,
        "or": operator.__or__
    }

    grads: Dict[str, List[Callable[..., float]]] = {
        "neg": [(lambda a: -1.0)],
        "abs": [(lambda a: a / abs(a))],
        "add": [(lambda a, b: 1.0), (lambda a, b: 1.0)],
        "sub": [(lambda a, b: 1.0), (lambda a, b: -1.0)],
        "mul": [(lambda a, b: b), (lambda a, b: a)],
        "truediv": [(lambda a, b: 1 / b), (lambda a, b: - a / (b ** 2))],
        "pow": [
            (lambda a, b: b * (a ** (b - 1))), 
            (lambda a, b: log(a) * (a ** b))
        ],
        "and": [
            (lambda a, b: 1.0 if a <= b else 0.0), 
            (lambda a, b: 1.0 if b <= a else 0.0)
        ],
        "or": [
            (lambda a, b: 1.0 if a >= b else 0.0), 
            (lambda a, b: 1.0 if b >= a else 0.0)
        ]
    }

    def __init__(
        self, 
        val: Any = 0.0, 
        op: str = "", 
        operands: Tuple[int, ...] = () 
    ) -> None:
        """
        Initialize a new diffable instance.

        Will attempt to convert val to float.

        Avoid manually setting op and operands. These will be handled 
        automatically in the context of a GradientTape. 

        :param val: Real value associated with diffable.
        :param op: Name of operation by which val was computed.
        :param operands: Gradient tape indices for operands from which val was 
            computed.
        """

        self.val = val
        self._op = op
        self._operands = operands

        self._index = None
        self.register()

    def __repr__(self):

        val = repr(self.val)
        op = repr(self.op)
        operands = repr(self.operands)

        return "diffable({}, {}, {})".format(val, op, operands)

    def __float__(self):

        return float(self.val)

    def __neg__(self):

        return diffable(
            (- self.val),
            op="neg",
            operands=(self.index,)
        )
    
    def __pos__(self):

        return self

    def __abs__(self):

        return diffable(
            abs(self.val),
            op="abs",
            operands=(self.index,)
        )

    def __invert__(self):

        return diffable(1.0) - self

    def __eq__(self, other):

        return self.val == other.val

    def __lt__(self, other):

        return self.val < other.val

    def __le__(self, other):

        return self.val <= other.val

    def __gt__(self, other):

        return self.val > other.val

    def __ge__(self, other):

        return self.val >= other.val

    def __add__(self, other):

        if isinstance(other, diffable):
            return diffable(
                self.val + other.val, 
                op="add", 
                operands=(self.index, other.index) 
            )
        else:
            return NotImplemented

    def __sub__(self, other):

        if isinstance(other, diffable):
            return diffable(
                val=self.val - other.val,
                op="sub",
                operands=(self.index, other.index)
            )
        else:
            return NotImplemented

    def __mul__(self, other):

        if isinstance(other, diffable):
            return diffable(
                self.val * other.val,
                op="mul",
                operands=(self.index, other.index)
            )
        else:
            return NotImplemented

    def __truediv__(self, other):

        if isinstance(other, diffable):
            return diffable(
                self.val / other.val,
                op="truediv",
                operands=(self.index, other.index)
            )
        else:
            return NotImplemented

    def __pow__(self, other):

        if isinstance(other, diffable):
            return diffable(
                self.val ** other.val,
                op="pow",
                operands=(self.index, other.index)
            )
        else:
            return NotImplemented

    def __and__(self, other):

        if isinstance(other, diffable):
            return diffable(
                min(self.val, other.val),
                op="and",
                operands=(self.index, other.index)
            )
        else:
            return NotImplemented

    def __or__(self, other):

        if isinstance(other, diffable):
            return diffable(
                max(self.val, other.val),
                op="and",
                operands=(self.index, other.index)
            )
        else:
            return NotImplemented

    @property
    def val(self) -> float:
        """
        Numerical value associated with self. 
        
        If a new value is set, will be converted to float first.
        """

        return self._val
    
    @val.setter
    def val(self, v: Any) -> None:

        self._val = float(v)
    
    @property
    def index(self) -> Optional[int]:

        return self._index

    @property
    def op(self) -> str:

        return self._op

    @property
    def operands(self) -> Tuple[int, ...]:
        
        return self._operands

    def register(self):
        """
        If currently within context of a gradient tape, add self to tape.
        
        Does nothing if not currently within context of a gradient tape.
        """

        try:
            l = tape_ctx.get()
        except LookupError:
            pass # Maybe print a warning? - Can
        else:
            if self._index is None:
                self._index = len(l)
                l.append(self)
            else:
                msg = "Diffable already registered in tape. Reset first."
                raise ValueError(msg)

    def release(self):
        """
        Release self from affiliation to a gradient tape.

        Removes all tape and op data from self. Calling this method manually 
        will corrupt any active gradient tape associated with self.
        """

        self._index = None
        self._op = ""
        self._operands = ()

    @classmethod
    def add_op(
        cls, 
        name: str, 
        func: Callable[..., float], 
        grads: List[Callable[..., float]]
    ) -> None:
        """
        Register a new diffable op.

        :param name: Name of op in the op registry.
        :param func: Callable implementing new op.
        :param grads: List of callables, one for computing the gradient of op 
            with respect to each argument in order.  
        """

        if name in cls.ops:
            raise ValueError("Op already defined.")
        else:
            cls.ops[name] = func
            cls.grads[name] = grads
    
    @classmethod
    def remove_op(cls, name: str) -> None:
        """
        Remove a diffable op.

        Will throw an error if an attempt is made to remove a basic op.

        :param name: Name of op to be removed.
        """

        if name in cls._basic_ops:
            raise ValueError("Cannot remove basic op.")
        else:
            del cls.ops[name]
            del cls.grads[name]


class DiffableNumDict(NumDict):
    """
    A numdict for storing and manipulating diffable values.
    
    This type of numdict is fixed to operate only on diffable instances and 
    does not support default values. 
    """

    __slots__ = ()

    def __init__(self, d, dtype=diffable, default=None) -> None:

        # This is a hacky solution, but would otherwise have to rewrite many 
        # ops. - Can
        if dtype != diffable:
            msg = "Cannot instantiate DiffableNumDict with dtype {}."
            raise TypeError(msg.format(dtype.__name__))

        super().__init__(d, dtype, default)

    def __setitem__(self, key, obj):
        """
        Map key to val.

        If val is in self, will set self[key].val to obj. If val is not in self, 
        will set self[key] to obj.
        """

        if key in self:
            self._dict[key].val = obj
        else:
            self._dict[key] = obj

    def __delitem__(self, key):

        del self._dict[key]

    @property
    def default(self):

        return self._default

    @default.setter
    def default(self, val):

        if val is None:
            self._default = None 
        else: 
            if self._default is None:
                self._default = self._cast(val)
            else:
                self._default.val = val

    def register(self):
        """
        If currently within context of a gradient tape, add every value of 
        self to tape.
        
        Does nothing if not currently within context of a gradient tape.
        """

        for v in self.values():
            v.register()

        if self.default is not None:
            self.default.register()


class GradientTapeCtxError(Exception):
    """Raised when an GradientTape event occurs in the wrong context."""

    pass


class GradientTape(object):
    """
    A simple gradient tape.

    Tracks diffable ops and computes forward and backward passes. Does not 
    support nesting.
    """

    _tape: List[diffable] 
    _token: Any
    _recording: bool

    def __init__(self, persistent: bool = False) -> None:

        self._tape = []
        self._recording = False
        self._persistent = persistent

    def __del__(self) -> None:

        for v in self._tape:
            v.release()

    def __len__(self):

        return len(self._tape)

    def __iter__(self):

        yield from iter(self._tape)

    def __getitem__(self, key: int) -> diffable:

        return self._tape[key]

    def __enter__(self):

        try:
            tape_ctx.get()
        except LookupError:
            self._token = tape_ctx.set([])
            self._recording = True
        else:
            raise GradientTapeCtxError("Cannot stack tapes.")
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self._recording = False
        self._tape = tape_ctx.get()
        tape_ctx.reset(self._token)
    
    @property
    def persistent(self) -> bool:

        return self._persistent

    def reset(self) -> None:
        """Reset tape."""

        if self._recording:
            raise GradientTapeCtxError("Cannot reset while recording.")
        else:
            for v in self._tape:
                v.release()
            self._tape = []
            self._recording = False

    def gradients(
        self, output: diffable, *variables: V, forward: bool = True
    ) -> Union[V, Tuple[V, ...]]:
        """
        Compute gradients of variables against output.

        Accepts as variables a sequence of lists or tuples of diffables or 
        diffable numdicts.

        If variables contains only one element, will return a single value 
        matching the type of element. Otherwise will return a tuple, with each 
        element matching the corresponding variables entry.

        Issues a sequence of calls to self.forward(), self.extract_indices(), 
        self.backward(), and self.extract_grads(). The call to self.forward() 
        is only issued if self.persistent is True and can be blocked by setting 
        the forward kwd to False. If self.persistent is False, will release 
        diffables associated with self prior to returning.

        :param output: Diffable against which to take gradients.
        :param variables: A sequence of mappings or sequences of diffable 
            numdicts containing variable values for the backward pass. 
            Gradients will be calculated only for these values.
        :param forward: Whether to perform a forward pass prior to computing 
            gradients if self is persistent. By default, True.
        """

        if self._recording:
            msg = "Cannot compute gradients while recording."
            raise GradientTapeCtxError(msg)

        if output.index is not None:
            if self._persistent and forward:
                self._forward()
            seed = output.index
            indices = self._extract_indices(*variables)
            grads = self._backward(seed, indices)
            grad_sequence = self._extract_grads(grads, *variables)
        else:
            msg = "Output has no recorded tape index."
            raise ValueError(msg)

        if not self._persistent:
            self.reset()

        return grad_sequence

    def _forward(self) -> None:
        """Perform a forward pass over the current tape."""
        
        if self._recording:
            msg = "Cannot compute forward pass while recording."
            raise GradientTapeCtxError(msg)

        for entry in self._tape:
            if entry.op == "":
                pass
            else:
                args = (self._tape[i] for i in entry.operands)
                entry.val = diffable.ops[entry.op](*args)

    def _backward(
        self, seed: int, indices: Set[int], seed_val: float = 1.0
    ) -> Dict[int, float]:
        """
        Perform a backward pass over the current tape.

        :param seed: Tape index seeding the backward pass.
        :param variables: A set of tape indices to be treated as variables in 
            the backward pass. Gradients will be calculated only for these 
            variables.
        :param seed_val: The seed value.
        """

        if self._recording:
            msg = "Cannot compute backward pass while recording."
            raise GradientTapeCtxError(msg)

        delta = {seed: seed_val}
        for i, entry in reversed(list(enumerate(self))):
            delta.setdefault(i, 0.0)
            if entry.op != "":
                for j, k in enumerate(entry.operands):
                    if self[k].op != "" or self[k].index in indices:
                        op_j = diffable.grads[entry.op][j]
                        delta.setdefault(k, 0.0)
                        args_j = (self[k].val for k in entry.operands)
                        delta[k] += delta[i] * op_j(*args_j)
        
        return delta

    @staticmethod
    def _extract_indices(*variables: V) -> Set[int]:
        """Extract tape indices for diffables in variables."""
        
        s: Set[int] = set()
        for item in variables:
            if isinstance(item, DiffableNumDict):
                s.update([v.index for v in item.values()])
            elif isinstance(item, (list, tuple)):
                s.update(type(item)([v.index for v in item]))
            else:
                msg = "Unexpected type passed as variable: {}."
                raise TypeError(msg.format(type(item).__name__))
        
        return s

    @staticmethod
    def _extract_grads(
        grads: Dict[int, float], *variables: V
    ) -> Union[V, Tuple[V, ...]]:
        """Extract grads for diffables in variables from raw grad dict."""
        
        ret: List[V] = []
        for item in variables:
            if isinstance(item, DiffableNumDict):
                d = {k: grads[v.index] for k, v in item.items()}
                if item.default is None:
                    default = None
                else:
                    default = grads[item.default.index]
                dd = DiffableNumDict(d, default=default)
                ret.append(cast(V, dd))
            if isinstance(item, list):
                l = [grads[v.index] for v in item]
                ret.append(cast(V, l))
        
        if len(ret) > 1:
            return tuple(ret)
        elif len(ret) == 1:
            return ret.pop()
        else:
            raise ValueError("Must provide at least one variable.")
