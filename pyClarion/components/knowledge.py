from typing import Self, Iterable, Type, Iterator, overload
from itertools import product

from ..numdicts import KeySpaceBase, Index, KeyForm, Key, root, path


class Sort[C: "Term"](KeySpaceBase[KeySpaceBase, C]):

    def __init__(self, mtype: Type[C]) -> None:
        super().__init__(KeySpaceBase, mtype)

    @overload
    def __rpow__(self, other: "Term") -> "Dimension":
        ...

    @overload
    def __rpow__(self, other: "Sort") -> "Dyads":
        ...

    def __rpow__(self, other: "Term | Sort") -> "Dimension | Dyads":
        if isinstance(other, Term):
            return Dimension(other, self)
        if isinstance(other, Sort):
            return Dyads(other, self)
        return NotImplemented


class Term(KeySpaceBase[Sort, Sort]):

    def __init__(self) -> None:
        super().__init__(Sort, Sort)
        self._vars_ = {}

    def __pow__(self, other: "Term | Var | Iterable[Term]") -> "Chunk":
        if isinstance(other, (Term, Var)):
            return Chunk({(self, other): 1.0})
        else:
            return Chunk({(self, atom): 1.0 for atom in other})

    def __rpow__(self, other: "Term | Var") -> "Chunk":
        if isinstance(other, (Term, Var)):
            return Chunk({(other, self): 1.0})
        return NotImplemented


class Atom(Term):
    pass


class Chunk(Term):
    _dyads_: "dict[tuple[Term | Var, Term | Var], float]"
    
    def __init__(
        self, dyads: "dict[tuple[Term | Var, Term | Var], float]"
    ) -> None:
        super().__init__()
        self._dyads_ = dyads
        self._vars_ = self._collect_vars_(dyad for dyad in dyads)

    @staticmethod
    def _collect_vars_(dyads: "Iterable[tuple[Term | Var, Term | Var]]") \
        -> dict[str, Sort]:
        vars_ = {}
        for dyad in dyads:
            for elt in dyad:
                if not isinstance(elt, Var):
                    continue
                sort = vars_.setdefault(elt.name, elt.sort)
                if elt.sort is not sort:
                    raise ValueError("Inconsistent var def")
        return vars_

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
            return Rule({other: 1.0, self: 1.0})
        return NotImplemented


class Rule(Term):
    _chunks_ : dict[Chunk, float]

    def __init__(self, chunks: dict[Chunk, float]):
        super().__init__()
        self._chunks_ = chunks
        self._vars_ = Chunk._collect_vars_(d for c in chunks for d in c._dyads_)


class Var:
    def __init__(self, name: str, sort: Sort) -> None:
        self.name = name
        self.sort = sort

    def __call__(self, valuation: dict[str, Term]) -> Term:
        if (value := valuation[self.name]) in self.sort:
            return value
        raise ValueError()


def instantiations[T: Chunk | Rule](term: T) -> Iterator[T]:
    for v in valuations(term):
        yield instantiate(term, v)


def valuations(term: Chunk | Rule) -> Iterator[dict[str, Term]]:
    for vals in product(*term._vars_.values()):
        yield {lb: sp[v] for v, (lb, sp) in zip(vals, term._vars_.items())}


def instantiate[T: Chunk | Rule](term: T, vals: dict[str, Term]) -> T:
    if isinstance(term, Rule):
        chunks = {instantiate(c, vals): w for c, w in term._chunks_.items()}
        return type(term)(chunks)
    if isinstance(term, Chunk):
        dyads: dict[tuple[Term | Var, Term | Var], float] = {
            (t1(vals) if isinstance(t1, Var) else t1, 
             t2(vals) if isinstance(t2, Var) else t2): w 
            for (t1, t2), w in term._dyads_.items()}
        return type(term)(dyads)


class Dimension(Index):
    term: Term
    sort: Sort

    def __init__(self, term: Term, sort: Sort) -> None:
        ksp = root(term)
        if ksp != root(sort):
            raise ValueError("Mismatched keyspaces")
        key = path(term).link(path(sort).link(Key("?"), 1), 0)
        super().__init__(ksp, KeyForm.from_key(key))
        self.term = term
        self.sort = sort


class Dyads(Index):
    sort1: Sort
    sort2: Sort

    def __init__(self, sort1: Sort, sort2: Sort) -> None:
        ksp = root(sort1)
        if ksp != root(sort2):
            raise ValueError("Mismatched keyspaces")
        ref = path(sort1).link(path(sort2), 0)
        super().__init__(ksp, ref, (1, 1))
        self.sort1 = sort1 
        self.sort2 = sort2
