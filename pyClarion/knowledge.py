from typing import Self, Iterable, Type, Iterator, TypedDict, NotRequired, get_type_hints
from itertools import product

from .numdicts import KeySpaceBase, Index, KeyForm, Key, root, path, ValidationError


class Branch[P: KeySpaceBase, C: "Branch"](KeySpaceBase[P, C]):
    pass


class Family(Branch[KeySpaceBase, "Sort"]):

    def __init__(self) -> None:
        super().__init__(KeySpaceBase, Sort)


class Sort[C: "Term"](Branch[KeySpaceBase, C]):
    _required_: frozenset[Key]

    def __init__(self, mtype: Type[C]) -> None:
        super().__init__(KeySpaceBase, mtype)
        cls = type(self)
        for name, typ in get_type_hints(cls).items():
            if isinstance(typ, type) and issubclass(typ, Term):
                setattr(self, name, typ())
        self._required_ = frozenset(self._members_)

    def __delattr__(self, name: str) -> None:
        if Key(name) in self._required_:
            raise ValidationError(f"Cannot remove required key '{name}'")


class Atoms(Sort["Atom"]):
    def __init__(self):
        super().__init__(Atom)


class Chunks(Sort["Chunk"]):
    def __init__(self):
        super().__init__(Chunk)


class Rules(Sort["Rule"]):
    def __init__(self):
        super().__init__(Rule)


class Term(Branch[Sort, Sort]):

    def __init__(self) -> None:
        super().__init__(Sort, Sort)

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


class Compound(Term):
    _descr_: str
    _vars_: dict
    _instances_: list[Self]
    _is_instance_: bool

    def __init__(self, inst) -> None:
        super().__init__()
        self._descr_ = ""
        self._vars_ = {}
        self._instances_ = []
        self._inst_ = inst

    def __rxor__(self: Self, other: str) -> Self:
        if not other.isidentifier():
            ValueError("Compound term identifier must be a valid "
                "python identifier")
        self._descr_ = other
        return self


class Chunk(Compound):
    _dyads_: "dict[tuple[Term | Var, Term | Var], float]"
    
    def __init__(
        self, 
        dyads: "dict[tuple[Term | Var, Term | Var], float]",
        inst: bool = False
    ) -> None:
        super().__init__(inst)
        self._dyads_ = dyads
        self._vars_.update(self._collect_vars_(dyad for dyad in dyads))

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


class Rule(Compound):
    _chunks_ : dict[Chunk, float]

    def __init__(self, chunks: dict[Chunk, float], inst: bool = False):
        super().__init__(inst)
        self._chunks_ = chunks
        self._vars_.update(
            Chunk._collect_vars_(d for c in chunks for d in c._dyads_))


class Var:
    def __init__(self, name: str, sort: Sort) -> None:
        self.name = name
        self.sort = sort

    def __call__(self, valuation: dict[str, Term]) -> Term:
        if (value := valuation[self.name]) in self.sort:
            return value
        raise ValueError()


def instantiations[T: Chunk | Rule](term: T) -> Iterator[T]:
    if not term._vars_:
        return
    for v in valuations(term):
        yield instantiate(term, v)


def valuations(term: Chunk | Rule) -> Iterator[dict[str, Term]]:
    for vals in product(*term._vars_.values()):
        yield {lb: sp[v] for v, (lb, sp) in zip(vals, term._vars_.items())}


def instantiate[T: Chunk | Rule](term: T, vals: dict[str, Term]) -> T:
    if isinstance(term, Rule):
        chunks = {instantiate(c, vals): w for c, w in term._chunks_.items()}
        return type(term)(chunks, inst=True)
    if isinstance(term, Chunk):
        dyads: dict[tuple[Term | Var, Term | Var], float] = {
            (t1(vals) if isinstance(t1, Var) else t1, 
             t2(vals) if isinstance(t2, Var) else t2): w 
            for (t1, t2), w in term._dyads_.items()}
        return type(term)(dyads, inst=True)


def standard_form(level: Branch) -> Key:
    if isinstance(level, Term):
        return Key(f"{path(level)}")
    if isinstance(level, Sort):
        return Key(f"{path(level)}:?")
    elif isinstance(level, Family):
        return Key(f"{path(level)}:?:?")
    else:
        raise TypeError()


class ByKwds(TypedDict):
    by: KeyForm
    b: NotRequired[int]


class BranchedIndex(Index):
    branches: tuple[Branch, ...]

    def __init__(self, *branches: Branch) -> None:
        root_ = root(branches[0])
        if any(root(b) != root_ for b in branches):
            raise ValueError("Mismatched keyspaces: Non identical roots")
        key = Key()
        for b in branches:
            key = key.link(standard_form(b), 0)
        super().__init__(root_, KeyForm.from_key(key))
        self.branches = branches

    def trunc(self, *directives: tuple[int, int]) -> KeyForm:
        keys: dict[int, Key] = {}
        for i, n in directives:
            k = standard_form(self.branches[i]) 
            k, _ = k.cut(k.size - n)
            keys[i] = k
        key = Key()
        for i, branch in enumerate(self.branches):
            key = key.link(keys.get(i, standard_form(branch)), 0)
        return KeyForm.from_key(key)

    def aggr(self, *bs: int) -> ByKwds:
        branches = [branch for i, branch in enumerate(self.branches) if i in bs]
        k = Key()
        for b in branches:
            k = k.link(standard_form(b), 0)
        ret: ByKwds = {"by": KeyForm.from_key(k)}
        matches = ret["by"].k.find_in(self.keyform.k)
        bbs = [m[1:1 + len(bs)] for m in matches]
        ret["b"] = bbs.index(tuple(b + 1 for b in bs))
        return ret


class Monads(BranchedIndex):
    def __init__(self, b: Branch) -> None:
        super().__init__(b)


class Dyads(BranchedIndex):
    def __init__(self, b1: Branch, b2: Branch) -> None:
        super().__init__(b1, b2)


class Triads(BranchedIndex):
    def __init__(self, b1: Branch, b2: Branch, b3: Branch) -> None:
        super().__init__(b1, b2, b3)
