__all__ = ["Fundaments", "Atoms", "Params", "Buses", "Compounds", "Chunks", 
    "Rules"]


from typing import Self, Sequence, ClassVar, overload

from .base import Sort
from .terms import Var, Bus, Atom, Param, Compound, Chunk, Rule
from ..numdicts.keys import Key
from ..numdicts.keyspaces import Dyad


class Fundaments[C: Bus | Param | Atom](Sort[C]):
    """A data sort for atomic terms.""" 
    _vars_: dict[str, Var]

    def __init__(self, name: str = "", prefix: str = "") -> None:
        super().__init__(name, prefix)
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

    def __init__(self, name: str = "", prefix: str = "atom"):
        super().__init__(name, prefix)

    @classmethod
    def _get_member_type_(cls) -> type[Atom]:
        return Atom

class Params(Fundaments[Param]):
    """
    A data sort for parameter atoms.

    Represents a collection of parameter terms.
    """
    _init_val_defaults_: ClassVar[dict[str, float]]

    def __init__(self, name: str = "", prefix: str = "param"):
        super().__init__(name, prefix)
        for label, value in self._init_val_defaults_.items():
            self[label].init_val = value

    def _init_values_(self) -> dict[Key, float]:
        return {~self[label]: self[label].init_val for label in self}

    @classmethod
    def _get_member_type_(cls) -> type[Param]:
        return Param


class Buses(Fundaments[Bus]):
    """A data sort for data lines."""
    def __init__(self, name: str = "", prefix: str = "bus"):
        super().__init__(name, prefix)

    @classmethod
    def _get_member_type_(cls) -> type[Bus]:
        return Bus


class Compounds[C: Compound](Sort[C]):
    """A data sort for compound terms."""
    _init_required_ = False


class Chunks[D: Dyad](Compounds[Chunk]):
    """
    A data sort for chunk terms.

    Represents a collection of chunk terms. This sort includes a `nil` term as 
    a necessary member.
    """
    _dyads_: D
    nil: Chunk

    def __init__(self, d: D, name: str = "", prefix: str = "chunk") -> None:
        super().__init__(name, prefix)
        self._dyads_ = d

    @classmethod
    def _get_member_type_(cls) -> type[Chunk]:
        return Chunk


class Rules[L: Chunks, R: Chunks](Compounds[Rule]):
    """
    A data sort for rule terms.

    Represents a collection of rule terms. This sort includes a `nil` term as a 
    necessary member.
    """
    _lhs_: L
    _rhs_: R
    nil: Rule 

    def __init__(self, name: str, lhs: L, rhs: R, prefix: str = "rule") -> None:
        super().__init__(name, prefix)
        self._lhs_ = lhs
        self._rhs_ = rhs

    @classmethod
    def _get_member_type_(cls) -> type[Rule]:
        return Rule
