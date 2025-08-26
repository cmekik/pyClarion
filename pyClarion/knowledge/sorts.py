from typing import Self, Sequence, overload

from .base import Sort, Var
from .terms import Bus, Atom, Compound, Chunk, Rule


class Fundaments[C: Bus | Atom](Sort[C]):
    """A data sort for atomic terms.""" 
    _vars_: dict[str, Var]

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._vars_ = {}

    @overload
    def __call__(self) -> Sequence[C]:
        ...

    @overload
    def __call__(self, name: str) -> Var[Self]:
        ...

    def __call__(self, name: str | None = None) -> Var[Self] | Sequence[C]:
        if name is not None:
            return self._vars_.setdefault(name, Var(name, self))
        return [self[val] for val in self]


class Atoms(Fundaments[Atom]):
    """
    A data sort for data atoms.

    Represents a collection of atomic data terms that are alike in content 
    (e.g., color terms, shape terms, etc.).
    """
    _mtype_ = Atom


class Buses(Fundaments[Bus]):
    """A data sort for data lines."""

    _mtype_ = Bus


class Compounds[C: Compound](Sort[C]):
    """A data sort for compound terms."""
    pass


class Chunks(Compounds[Chunk]):
    """
    A data sort for chunk terms.

    Represents a collection of chunk terms. This sort includes a `nil` term as 
    a necessary member.
    """
    _mtype_ = Chunk
    nil: Chunk

    def __init__(self, name: str = ""):
        super().__init__(name)


class Rules(Compounds[Rule]):
    """
    A data sort for rule terms.

    Represents a collection of rule terms. This sort includes a `nil` term as a 
    necessary member.
    """
    _mtype_ = Rule
    nil: Rule 

    def __init__(self, name: str = ""):
        super().__init__(name)