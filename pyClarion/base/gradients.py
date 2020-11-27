"""
Provides support for basic native automatic diffferentiation in pyClarion.

Defines the `diffable`, `OpTracer`, and `DiffableNumDict` classes, which support 
automatic reverse-mode differentiation with gradient tapes/Wengert lists.

Based on CMU autodiff lecture notes:
http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf
"""


__all__ = ["diffable", "OpTracer", "DiffableNumDict"]


from .numdicts import BaseNumDict

from itertools import chain
from math import log
from contextvars import ContextVar
from typing import (
    Tuple, List, Mapping, Any, Union, Callable, TypeVar, Hashable, Optional, 
    Dict, Iterable, Type
)
import operator


# A context variable for storing gradient tapes.
tape_ctx: ContextVar = ContextVar("tape_ctx") # actually List["diffable"]


class diffable(object):
    """
    A floating point value amenable to reverse-mode gradient computations.

    Collaborates with OpTracer. Does not expose the full float inteface.
    
    By default, supports backpropagation with the following operations: abs, +, 
    -, *, /, **, &, |. The bitwise operations & and | are interpreted as fuzzy 
    logical operators. They compute min and max respectively.

    Other differentiable operators may be registered as needed.

    Comparison functions ==, <, <=, >, >= are also defined.
    """

    __slots__ = ("_val", "_op", "_operands", "_index")

    _basic_ops = {
        "neg", "abs" "add", "sub", "mul", "truediv", "pow", "and", "or"
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
        "truediv": [(lambda a, b: 1 / b), (lambda a, b: - a * (b ** -2))],
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
        val: float = 0.0, 
        op: str = "var", 
        operands: Tuple[int, ...] = () 
    ) -> None:
        """
        Initialize a new diffable instance.

        Avoid manually setting op and operands. These will be handled 
        automatically in the context of a OpTracer. 

        :param val: Real value associated with diffable.
        :param op: Name of operation by which val was computed.
        :param operands: Gradient tape indices for operands from which val was 
            computed.
        """

        self._val = val
        self._op = op
        self._operands = operands

        self._index = 0
        self.add_to_current_tape()

    def __repr__(self):

        val = repr(self.val)
        op = repr(self.op)
        operands = repr(self.operands)

        return "diffable({}, {}, {})".format(val, op, operands)

    def __float__(self):

        return self.val

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

        return diffable(
            self.val + other.val, 
            op="add", 
            operands=(self.index, other.index) 
        )

    def __sub__(self, other):

        return diffable(
            val=self.val - other.val,
            op="sub",
            operands=(self.index, other.index)
        )

    def __mul__(self, other):

        return diffable(
            self.val * other.val,
            op="mul",
            operands=(self.index, other.index)
        )

    def __truediv__(self, other):

        return diffable(
            self.val / other.val,
            op="truediv",
            operands=(self.index, other.index)
        )

    def __pow__(self, other):

        return diffable(
            self.val ** other.val,
            op="pow",
            operands=(self.index, other.index)
        )

    def __and__(self, other):

        return diffable(
            min(self.val, other.val),
            op="and",
            operands=(self.index, other.index)
        )

    def __or__(self, other):

        return diffable(
            max(self.val, other.val),
            op="and",
            operands=(self.index, other.index)
        )

    @property
    def val(self) -> float:

        return self._val
    
    @val.setter
    def val(self, v: float) -> None:

        self._val = float(v)
    
    @property
    def index(self) -> int:

        return self._index

    @property
    def op(self) -> str:

        return self._op

    @property
    def operands(self) -> Tuple[int, ...]:
        
        return self._operands

    def add_to_current_tape(self):
        """
        If currently within context of a gradient tape, add self to tape.
        
        Does nothing if not currently within context of a gradient tape.
        """

        try:
            l = tape_ctx.get()
        except LookupError:
            pass # Maybe print a warning? - Can
        else:
            self._index = len(l)
            l.append(self)

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


class OpTracerCtxError(Exception):
    """Raised when a OpTracer event occurs in the wrong context."""

    pass


class OpTracer(object):
    """
    A simple gradient tape.

    Tracks diffable ops and computes forward and backward passes. Does not 
    support nesting.
    """

    _tape: List[diffable] 
    _token: Any
    _recording: bool

    def __init__(self) -> None:

        self._tape = []
        self._recording = False

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
            raise OpTracerCtxError("Cannot stack tapes.")
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self._recording = False
        self._tape = tape_ctx.get()
        tape_ctx.reset(self._token)

    def reset(self) -> None:
        """Reset tape."""

        if self._recording:
            raise OpTracerCtxError("Cannot reset while recording.")
        else:
            self._tape = []
            self._recording = False

    def watch(self, diffables: Iterable[diffable]) -> None:
        """Add diffable variables to gradient tape."""

        if self._recording:
            for v in diffables:
                v.add_to_current_tape()
        else:
            raise OpTracerCtxError("Tape not currently recording.")

    def forward(self) -> None:
        """Perform a forward pass over the current tape."""
        
        for entry in self._tape:
            if entry.op == "var":
                pass
            else:
                args = (self._tape[i] for i in entry.operands)
                entry.val = diffable.ops[entry.op](*args)

    def backward(self, f: Union[diffable, int], val=1.0) -> Dict[int, float]:
        """
        Perform a backward pass over the current tape.

        :param f: Diffable value or index with which to seed the backward pass.
        :param val: The seed value.
        """

        if isinstance(f, diffable):
            idx = f.index
        elif isinstance(f, int):
            idx = f
        else:
            msg = "Unexpected seed type for backward pass: {}"
            raise TypeError(msg.format(type(f).__name__))

        delta = {idx: val}
        for i, entry in reversed(list(enumerate(self))):
            delta.setdefault(i, 0.0)
            if entry.op != "var":
                for j, k in enumerate(entry.operands):
                    op_j = diffable.grads[entry.op][j]
                    delta.setdefault(k, 0.0)
                    args_j = (self[k].val for k in entry.operands)
                    delta[k] += delta[i] * op_j(*args_j)
        
        return delta


class DiffableNumDict(BaseNumDict, Mapping[Any, diffable]):
    """
    A numdict for storing and manipulating diffable values.
    
    This type of numdict is fixed to operate only on diffable instances and 
    does not support default values. 
    """

    __slots__ = ()

    def __init__(
        self, 
        d: Mapping, 
        dtype: Type = diffable, 
        default: None = None
    ) -> None:

        # This is a hacky solution, but would otherwise have to rewrite many 
        # ops. - Can
        if dtype != diffable:
            msg = "Cannot instantiate DiffNumDict with dtype {}."
            raise TypeError(msg.format(dtype.__name__))
        if default is not None:
            msg = "Default values not allowed."
            raise TypeError(msg)

        super().__init__(d, dtype, default)

    def __setitem__(self, key, obj):
        """
        Map key to val.

        If val is in self and obj has type diffable, will set self[key].val to 
        obj.val. If obj is not a diffable instance, will set self[key].val to 
        float(obj). If val is not in self and obj is a diffable instance, will 
        set self[key] to obj. If obj is not a diffable instance will set 
        self[key] to diffable(obj).
        """

        if key in self:
            if type(obj) == diffable:
                self._dict[key].val = obj.val
            else:
                self._dict[key].val = float(obj)
        else:
            if type(obj) == diffable:
                self._dict[key] = obj
            else:
                self._dict[key] = diffable(obj)

    def __delitem__(self, key):

        del self._dict[key]

    @property
    def dtype(self):

        return self._dtype
