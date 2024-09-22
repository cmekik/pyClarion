from typing import Any, Self, Iterable, Sequence, Type, NoReturn, overload

from ..numdicts import GenericKeySpace, KeySpace, Index, KeyForm, root, path


class Support(GenericKeySpace["Family"]):
    @property
    def _child_type_(self) -> Type["Family"]:
        return Family


class Family(GenericKeySpace["Sort"]):
    @property
    def _child_type_(self) -> Type["Sort"]:
        return Sort


class Sort(GenericKeySpace["Atom"]):
    @property
    def _child_type_(self) -> Type["Atom"]:
        return Atom

    @overload
    def __rmatmul__(self, other: "Atom") -> "Dimension":
        ...

    @overload
    def __rmatmul__(self, other: "Sort") -> "Dyads":
        ...

    def __rmatmul__(self, other):
        if isinstance(other, Atom):
            return Dimension(other, self)
        if isinstance(other, Sort):
            return Dyads(other, self)
        return NotImplemented


class Atom(KeySpace):
    def __setattr__(self, name: str, value: Any) -> None:
        if name != "_parent_" and isinstance(value, KeySpace):
            raise TypeError(f"{type(self).__name__} object cannot have "
                "child keyspace")
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> NoReturn:
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def __pos__(self: Self) -> Self:
        return self

    def __matmul__(self, other: "Atom | Var | Iterable[Atom]") -> "Chunk":
        if isinstance(other, (Atom, Var)):
            return Chunk({(self, other): 1.0})
        else:
            return Chunk({(self, atom): 1.0 for atom in other})


class Chunk(Atom):
    _dyads_: dict[tuple[Atom, "Atom | Var"], float]
    
    def __init__(self, dyads: dict[tuple[Atom, "Atom | Var"], float]) -> None:
        super().__init__()
        self._dyads_ = dyads

    def __pos__(self: Self) -> Self:
        return self
    
    def __neg__(self: Self) -> Self:
        return type(self)({d: -w for d, w in self._dyads_.items()})
    
    def __rmul__(self: Self, other: float) -> Self:
        if isinstance(other, float):
            return type(self)({d: other * w for d, w in self._dyads_.items()})
        return NotImplemented

    def __add__(self, other: "Chunk") -> "Chunk":
        if isinstance(other, Chunk):
            dyads = self._dyads_.copy()
            dyads.update(other._dyads_)
            return Chunk(dyads)
        return NotImplemented

    def __rshift__(self, other: "Chunk") -> "Rule":
        if isinstance(other, Chunk):
            return Rule([other, self])
        return NotImplemented


class Rule(Atom):
    _chunks_ : list[Chunk]

    def __init__(self, chunks: Sequence[Chunk]):
        super().__init__()
        self._chunks_ = list(chunks)


class Var:
    def __init__(self, sort: Sort) -> None:
        self.sort = sort


class Dimension(Index):
    def __init__(self, atom: Atom, sort: Sort) -> None:
        ksp = root(atom)
        if ksp != root(sort):
            raise ValueError("Mismatched keyspaces")
        ref = path(atom).link(path(sort), 0)
        super().__init__(root(atom), KeyForm(ref, (0, 1)))


class Dyads(Index):
    def __init__(self, sort1: Sort, sort2: Sort) -> None:
        ksp = root(sort1)
        if ksp != root(sort2):
            raise ValueError("Mismatched keyspaces")
        ref = path(sort1).link(path(sort2), 0)
        super().__init__(ksp, KeyForm(ref, (1, 1)))


class Triads(Index):
    def __init__(self, sort1: Sort, sort2: Sort, sort3: Sort) -> None:
        ksp = root(sort1)
        if ksp != root(sort2) or ksp != root(sort3):
            raise ValueError("Mismatched keyspaces")
        ref = path(sort1).link(path(sort2), 0).link(path(sort3), 0)
        super().__init__(ksp, KeyForm(ref, (1, 1, 1)))
