from __future__ import annotations

from . import numdict as nd

from typing import (Tuple, List, Dict, Any, Set, Union, TypeVar, 
    Hashable, Callable, overload, ClassVar)
from typing_extensions import ParamSpec
from functools import wraps
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


__all__ = ["GradientTape"]


T = TypeVar("T", bound=Hashable)
P = ParamSpec("P")
C = TypeVar("C", bound=Callable)


class TapeError(RuntimeError):
    """Raised when an inappropriate GradientTape event occurs."""
    pass


# Needs to have a nice repr, awful to read w/ large numdicts.
@dataclass
class TapeCell:
    """A gradient tape entry."""
    value: nd.NumDict
    op: str = ""
    operands: Tuple[int, ...] = ()
    kwds: dict = field(default_factory=dict)


class GradientTape:
    """
    A gradient tape.

    Tracks diffable ops and computes forward and backward passes. Does not 
    support nesting.

    Tracked NumDicts are protected until the first call to self.gradients(). 
    Afterwards, protection settings are restored to their original values.
    """

    __slots__ = ("_cells", "_index", "_token", "_rec", "_prot", "_block")

    TAPE: ClassVar[ContextVar] = ContextVar("TAPE")

    OPS: ClassVar = {}
    GRADS: ClassVar = {}

    _cells: List[TapeCell]
    _protected: List[bool]
    _index: Dict[int, int]
    _block: Set[int]
    _token: Any
    _rec: bool

    def __init__(self) -> None:
        self._cells = []
        self._prot = []
        self._index = {}
        self._block = set()
        self._rec = False
        self._token = None

    def __repr__(self):
        name = type(self).__name__
        rec = self._rec
        length = len(self._cells)
        return f"<{name} length: {length} rec: {rec}>"

    def __enter__(self):
        try:
            self.TAPE.get()
        except LookupError:
            self._rec = True
            self._token = self.TAPE.set(self)
        else:
            raise TapeError("Cannot stack gradient tapes.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.TAPE.reset(self._token)
        self._token = None
        self._rec = False

    def _register(
        self,
        value: nd.NumDict,
        op: str = "",
        inputs: Tuple[nd.NumDict, ...] = (),
        kwds: dict = None
    ) -> None:
        # Add a new tape cell containing the given information.

        if not self._rec:
            raise TapeError("Register NumDict only when recording.")

        for d in inputs: # register any new operands
            if id(d) not in self._index:
                self._register(d)
        operands = tuple(self._index[id(d)] for d in inputs)
        kwds = {} if kwds is None else kwds
        new_cell = TapeCell(value, op, operands, kwds)

        self._index[id(value)] = len(self._cells)
        self._cells.append(new_cell)

        # Temporarily protect value
        self._prot.append(value.prot) 
        value.prot = True

    def _get_index(self, d: nd.NumDict) -> int:
        # Return the tape index at which d is registered.
        return self._index[id(d)]

    def _backward(
        self, seed: int, indices: Set[int], seed_c: float
    ) -> Dict[int, nd.NumDict]:
        """
        Perform a backward pass over the current tape.

        :param seed: Tape index seeding the backward pass.
        :param indices: A set of tape indices to be treated as variables in 
            the backward pass. Gradients will be calculated only for these 
            variables.
        :param seed_val: The seed value.
        """

        if self._rec: raise TapeError("Stop recording before backward pass.")
        delta = {seed: nd.NumDict(c=seed_c)}
        for i, cell in reversed(list(enumerate(self._cells))):
            delta.setdefault(i, nd.NumDict(c=0.0))
            if cell.op:
                grad_op = self.GRADS[cell.op]
                if grad_op is None or id(cell.value) in self._block:
                    pass
                else:
                    inputs = (self._cells[k].value for k in cell.operands)
                    grads = grad_op(delta[i], cell.value, *inputs, **cell.kwds)
                    for j, k in enumerate(cell.operands):
                        if k in indices or self._cells[k].op != "":
                            delta.setdefault(k, nd.NumDict(c=0.0))
                            delta[k] += grads[j]
        return delta

    def reset(self) -> None:
        """Reset tape."""
        if self._rec:
            raise TapeError("Cannot reset while recording.")
        else:
            # Restore original protection settings
            for cell, prot in zip(self._cells, self._prot):
                cell.value.prot = prot
            self._prot.clear()
            self._cells.clear() 
            self._index.clear()

    def block(self, d: nd.NumDict) -> nd.NumDict:
        """
        Block gradient accumulation through d.
        
        Returns d as is for in-line use.

        >>> with GradientTape() as t:
        ...     d1 = nd.NumDict(c=2)
        ...     d2 = nd.NumDict({1: 1, 2: 2})
        ...     d3 = d2.reduce_sum()
        ...     result = d1 * t.block(d3)
        >>> t.gradients(result, (d1, d2, d3))
        nd.NumDict(c=2), (nd.NumDict(c=3), nd.NumDict(c=0), nd.NumDict(c=2))
        """

        self._block.add(id(d))
        return d

    @classmethod
    @contextmanager
    def pause(cls):
        """
        Suspend recording of autodiff ops.

        >>> with GradientTape() as t:
        ...     d1 = nd.NumDict(c=3)
        ...     with GradientTape.pause():
        ...         d2 = nd.NumDict(c=4)
        ...         d3 = d2 / 5
        ...     d4 = d1 * d3
        >>> t.gradients(d4, d2) 
        Traceback (most recent call last):
            ...
        KeyError: 
        """
        
        try:
            tape = cls.TAPE.get()
        except LookupError:
            yield
        else:
            rec = tape._rec
            tape._rec = False
            yield
            tape._rec = rec

    @overload
    def gradients(
        self, output: nd.NumDict, variables: nd.NumDict
    ) -> Tuple[nd.NumDict, nd.NumDict]:
        ...

    @overload
    def gradients(
        self, output: nd.NumDict, variables: Tuple[nd.NumDict, ...]
    ) -> Tuple[nd.NumDict, Tuple[nd.NumDict, ...]]:
        ...

    def gradients(
        self,
        output: nd.NumDict,
        variables: Union[nd.NumDict, Tuple[nd.NumDict, ...]]
    ) -> Tuple[nd.NumDict, Union[nd.NumDict, Tuple[nd.NumDict, ...]]]:
        """
        Compute gradients of variables against output.

        Accepts as variables a sequence of NumDicts.

        If variables contains only one element, will return a single value. 
        Otherwise will return a tuple, with each element matching the 
        corresponding variables entry.

        :param output: Value against which to take gradients.
        :param variables: A sequence of NumDicts containing variable values for 
            the backward pass. Gradients will be calculated only for these 
            values.
        """

        if self._rec: raise TapeError("Stop recording to compute gradients.")
        seed = self._get_index(output)
        if isinstance(variables, tuple):
            indices = set(self._get_index(var) for var in variables)
            _grads = self._backward(seed, indices, 1.0)
            grads = tuple(_grads[self._get_index(var)] for var in variables)
            self.reset()
            return output, grads
        else:
            indices = {self._get_index(variables)}
            _grads = self._backward(seed, indices, 1.0)
            grads = _grads[self._get_index(variables)]
            self.reset()
            return output, grads    

    @classmethod
    def op(cls, no_grad=False) -> Callable[[C], C]:
        """Register a new op."""

        def wrapper(f: Callable[P, nd.NumDict]) -> Callable[P, nd.NumDict]:

            name = f.__qualname__

            @wraps(f)
            def op_wrapper(*args: P.args, **kwargs: P.kwargs) -> nd.NumDict:
                d = f(*args, **kwargs)
                try: 
                    tape = cls.TAPE.get()
                except LookupError: 
                    pass
                else: 
                    tape._register(d, name, args, kwargs)
                return d

            if name in cls.OPS:
                raise ValueError(f"Op name '{name}' already in registry.")
            cls.OPS[name] = op_wrapper
            if no_grad:
                cls.GRADS[name] = None

            return op_wrapper
        
        return wrapper # type: ignore

    @classmethod
    def grad(cls, op: Callable) -> Callable[[C], C]:
        """Register gradient function for op."""

        name = op.__qualname__
        if name not in cls.OPS:
            raise ValueError(f"Unregistered op '{name}' passed to grad.")

        def wrapper(func: C) -> C:
            cls.GRADS[name] = func
            return func

        return wrapper
