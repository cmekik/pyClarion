from typing import Any, Self, Iterable, Sequence, NoReturn, overload

from ..numdicts import KeySpace, Index, KeyForm, Key, root, path


class Sort(KeySpace):
    def __setitem__(self, name: str, value: Any) -> None:
        if not isinstance(value, KeySpace):
            raise TypeError(f"{type(self).__name__}.__setitem__ expects value " 
                f"of type {Atom.__name__}, but got a value of type "
                f"{type(value).__name__} instead")
        setattr(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "_parent_" and isinstance(value, KeySpace):
            if not isinstance(value, Atom):
                raise TypeError(f"{type(self).__name__} object cannot have "
                    f"child keyspace of type {type(value).__name__}")
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> "Atom":
        if name.startswith("_") and name.endswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")
        new = Atom()
        self.__setattr__(name, new)
        return new

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

    def __pow__(self, other: "Atom | Var | Iterable[Atom]") -> "Chunk":
        if isinstance(other, (Atom, Var)):
            return Chunk({(self, other): 1.0})
        else:
            return Chunk({(self, atom): 1.0 for atom in other})

    def __rpow__(self, other: "Atom | Var") -> "Chunk":
        if isinstance(other, (Atom, Var)):
            return Chunk({(other, self): 1.0})
        return NotImplemented


class Chunk(Atom):
    _dyads_: "dict[tuple[Atom | Var, Atom | Var], float]"
    
    def __init__(
        self, dyads: "dict[tuple[Atom | Var, Atom | Var], float]"
    ) -> None:
        super().__init__()
        self._dyads_ = dyads
    #     self._init_vars_()

    # def _init_vars_(self) -> None:
    #     var_dict = {}
    #     for dyad in self._dyads_:
    #         for elt in dyad:
    #             if not isinstance(elt, Var):
    #                 continue
    #             sort = var_dict.setdefault(elt.name, elt.sort)
    #             if elt.sort is not sort:
    #                 raise ValueError("Inconsistent var def")
    #     for var, sort in var_dict.items():
    #         self[var] = sort

    # def __call__(self):
    #     ...

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

    def __sub__(self, other: "Chunk") -> "Chunk":
        if isinstance(other, Chunk):
            dyads = self._dyads_.copy()
            dyads.update({d: -w for d, w in other._dyads_.items()})
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
    def __init__(self, name: str, sort: Sort) -> None:
        self.name = name
        self.sort = sort


class Dimension(Index):

    def __init__(self, atom: Atom, sort: Sort) -> None:
        ksp = root(atom)
        if ksp != root(sort):
            raise ValueError("Mismatched keyspaces")
        key = path(atom).link(path(sort).link(Key("?"), 1), 0)
        super().__init__(ksp, KeyForm.from_key(key))


class Dyads(Index):

    def __init__(self, sort1: Sort, sort2: Sort) -> None:
        ksp = root(sort1)
        if ksp != root(sort2):
            raise ValueError("Mismatched keyspaces")
        ref = path(sort1).link(path(sort2), 0)
        super().__init__(ksp, ref, (1, 1))


class Triads(Index):

    def __init__(self, sort1: Sort, sort2: Sort, sort3: Sort) -> None:
        ksp = root(sort1)
        if ksp != root(sort2) or ksp != root(sort3):
            raise ValueError("Mismatched keyspaces")
        ref = path(sort1).link(path(sort2), 0).link(path(sort3), 0)
        super().__init__(ksp, ref, (1, 1, 1))        
