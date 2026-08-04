__all__ = ["Atom", "Param", "Atoms", "Params", "DataFamily", "AtomFamily", 
    "ParamFamily", "Nodes"]


from typing import Self, ClassVar, Sequence, Any, overload

from .base import Family, Sort, Term
from .compounds import (Var, Compounds, Bus, AnyDVPairs, Chunk)
from ..numdicts.keys import Key


class Primitive(Term):
    def __rpow__(self, other: Bus) -> "Chunk":
        if isinstance(other, Bus):
            return Chunk({(other, self): 1.0})


class Atom(Primitive):
    """
    An atomic data term.

    Represents some basic data element (e.g., a feature, a parameter, etc.).
    """
    pass


class Level(Primitive):
    """
    An atomic scale level.

    Represents some basic ordered data element.
    """

    def __lt__(self, other: Any) -> bool:
        if other is self:
            return False
        else:
            return self.__le__(other)

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, Level):
            pass
        elif self._parent_ == other._parent_:
            for ks in self._parent_._members_.values():
                if ks is self:
                    return True
                elif ks is other:
                    return False
            assert False # This should never happen in practice.
        return NotImplemented


class Param(Primitive):
    """
    An atomic parameter term.

    Represents some basic model parameter.
    """

    init_val: float

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.init_val = float("nan")


class Primitives[C: Primitive](Sort[C]):
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


class Atoms(Primitives[Atom]):
    """
    A data sort for data atoms.

    Represents a collection of atomic data terms that are alike in content 
    (e.g., color terms, shape terms, etc.).
    """
    nil: Atom

    def __init__(self, name: str = "", prefix: str = "atom"):
        super().__init__(name, prefix)

    @classmethod
    def _get_member_type_(cls) -> type[Atom]:
        return Atom


class Levels(Primitives[Level]):
    """
    A data sort for scalar terms.

    Represents an ordered collection of primitive values.
    """
    nil: Level

    def __init__(self, name: str = "", prefix: str = "scalar"):
        super().__init__(name, prefix)

    @classmethod
    def _get_member_type_(cls) -> type[Level]:
        return Level


class Params(Primitives[Param]):
    """
    A data sort for parameter atoms.

    Represents a collection of parameter terms.
    """
    _init_val_defaults_: ClassVar[dict[str, float]]

    def __init__(self, name: str = "", prefix: str = "param"):
        super().__init__(name, prefix)
        defaults = {}
        for cls in reversed(type(self).mro()):
            if issubclass(cls, Params):
                defaults.update(cls._init_val_defaults_)
        for label, value in defaults.items():
            self[label].init_val = value

    def _init_values_(self) -> dict[Key, float]:
        return {~self[label]: self[label].init_val for label in self}

    @classmethod
    def _get_member_type_(cls) -> type[Param]:
        return Param

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for sup in cls.mro():
            if issubclass(sup, Params):
                cls._init_val_defaults_.update(sup._init_val_defaults_)


class DataFamily(Family[Sort]):
    def __init__(self, name: str = ""):
        super().__init__(name, skip=(Compounds,))

    @classmethod
    def _get_member_type_(cls) -> type[Sort]:
        return Sort


class AtomFamily(Family[Atoms]):
    @classmethod
    def _get_member_type_(cls) -> type[Atoms]:
        return Atoms


class LevelFamily(Family[Levels]):
    @classmethod
    def _get_member_type_(cls) -> type[Levels]:
        return Levels


class ParamFamily(Family[Params]):
    @classmethod
    def _get_member_type_(cls) -> type[Params]:
        return Params


type Nodes = AnyDVPairs | Sort[Term]
