from typing import Self, Iterable, Type, Iterator, TypedDict, NotRequired, get_type_hints
from weakref import WeakValueDictionary
from itertools import product, count

from .numdicts import KeySpaceBase, KeyForm, Key, path, ValidationError


class Branch[P: KeySpaceBase, C: "Branch"](KeySpaceBase[P, C]):
    pass


class Sort[C: "Term"](Branch[KeySpaceBase, C]):
    _required_: frozenset[Key]
    _counter_: count
    _vars_: dict[str, "Var"]

    def __init__(self, mtype: Type[C]) -> None:
        super().__init__(KeySpaceBase, mtype)
        cls = type(self)
        for name, typ in get_type_hints(cls).items():
            if isinstance(typ, type) and issubclass(typ, Term):
                setattr(self, name, typ())
        self._required_ = frozenset(self._members_)
        self._counter_ = count()
        self._vars_ = {}

    def __call__(self, name: str) -> "Var":
        return self._vars_.setdefault(name, Var(name, self))

    def __delattr__(self, name: str) -> None:
        if Key(name) in self._required_:
            raise ValidationError(f"Cannot remove required key '{name}'")


class Family[S: "Sort"](Branch[KeySpaceBase, S]):

    def __init__(self, sort: Type[S] = Sort) -> None:
        super().__init__(KeySpaceBase, sort)


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
    _instances_: WeakValueDictionary[str, Self]
    _template_: Self | None

    def __init__(self, template: Self | None = None) -> None:
        super().__init__()
        self._descr_ = ""
        self._vars_ = {}
        self._instances_ = WeakValueDictionary()
        self._template_ = template

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
        dyads: "dict[tuple[Term | Var, Term | Var], float] | None" = None,
        template: Self | None = None
    ) -> None:
        dyads = dyads or {}
        super().__init__(template)
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
            return Rule({self: 1.0, other: 1.0})
        return NotImplemented


class Rule(Compound):
    _chunks_ : dict[Chunk, float]

    def __init__(
        self, 
        chunks: dict[Chunk, float] | None = None, 
        template: Self | None = None
    ) -> None:
        chunks = chunks or {}
        super().__init__(template)
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


class Atoms(Sort[Atom]):
    def __init__(self):
        super().__init__(Atom)


class Chunks(Sort[Chunk]):
    nil: Chunk

    def __init__(self):
        super().__init__(Chunk)


class Rules(Sort[Rule]):
    nil: Rule 

    def __init__(self):
        super().__init__(Rule)


class Actions(Atoms):
    nil: Atom


class ActionFamily[S: Actions](Family[S]):
    def __init__(self, sort: Type[S] = Actions) -> None:
        super().__init__(sort)


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
        return type(term)(chunks, term)
    if isinstance(term, Chunk):
        dyads: dict[tuple[Term | Var, Term | Var], float] = {
            (t1(vals) if isinstance(t1, Var) else t1, 
             t2(vals) if isinstance(t2, Var) else t2): w 
            for (t1, t2), w in term._dyads_.items()}
        return type(term)(dyads, term)
    raise TypeError()


class ChunkData(TypedDict):
    ciw: dict[Key, float]
    tdw: dict[Key, float]


class RuleData(TypedDict):
    riw: dict[Key, float]
    lhw: dict[Key, float]
    rhw: dict[Key, float]
    lhs: ChunkData
    rhs: ChunkData


def compile_chunk(chunk: Chunk, sort: Chunks) -> ChunkData:
    if not chunk._name_:
        raise ValueError("Cannot compile unnamed chunk")
    ciw = {}; tdw = {}; k = path(sort)
    kc = k.link(Key(chunk._name_), k.size)
    if chunk._template_ is None:
        ciw[kc.link(kc, 0)] = 1.0
    else:
        if not chunk._template_._name_:
            raise ValueError("Cannot compile instance chunk with unnamed "
                "template")
        chunk._template_._instances_[chunk._name_] = chunk
        kt = k.link(Key(chunk._template_._name_), k.size)
        ciw[kt.link(kc, 0)] = 1.0
    if not chunk._vars_:
        for (s1, s2), w in chunk._dyads_.items():
            assert isinstance(s1, Term) and isinstance(s2, Term)
            kw = kc.link(path(s1), 0).link(path(s2), 0)
            tdw[kw] = w 
    return ChunkData(ciw=ciw, tdw=tdw)


def compile_chunks(*chunks: Chunk, sort: Chunks) \
    -> tuple[list[Chunk], ChunkData]:
    new_chunks = []; chunk_data = ChunkData(ciw={}, tdw={})
    for chunk in chunks:
        chunk._name_ = chunk._name_ or f"c{next(sort._counter_)}"
        data = compile_chunk(chunk, sort)
        new_chunks.append(chunk)
        chunk_data["ciw"].update(data["ciw"]) 
        chunk_data["tdw"].update(data["tdw"])
        for inst in instantiations(chunk):
            inst._name_ = inst._name_ or f"c{next(sort._counter_)}" 
            _data = compile_chunk(inst, sort)
            new_chunks.append(inst)
            chunk_data["ciw"].update(_data["ciw"])
            chunk_data["tdw"].update(_data["tdw"])
    return new_chunks, chunk_data


def compile_rule(rule: Rule, sort: Rules, lhs: Chunks, rhs: Chunks) -> RuleData:
    riw = {}; lhw = {}; rhw = {}; k = path(sort)
    kr = k.link(Key(rule._name_), k.size)
    if rule._template_ is None:
        riw[kr.link(kr, 0)] = 1.0
    else:
        if not rule._template_._name_:
            raise ValueError("Cannot compile instance chunk with unnamed "
                "template")
        rule._template_._instances_[rule._name_] = rule
        kt = k.link(Key(rule._template_._name_), k.size)
        riw[kt.link(kr, 0)] = 1.0
    ret = RuleData(
        riw=riw, lhw=lhw, rhw=rhw, 
        lhs=ChunkData(ciw={}, tdw={}),
        rhs=ChunkData(ciw={}, tdw={}))
    chunks, weights = zip(*rule._chunks_.items())
    if rule._vars_:
        for chunk in chunks[:-1]:
            chunk._name_ = chunk._name_ or f"c{next(lhs._counter_)}"
            data = compile_chunk(chunk, lhs)
            ret["lhs"]["ciw"].update(data["ciw"])
            ret["lhs"]["tdw"].update(data["tdw"])
        chunk = chunks[-1]
        chunk._name_ = chunk._name_ or f"c{next(rhs._counter_)}"
        data = compile_chunk(chunk, rhs)
        ret["rhs"]["ciw"].update(data["ciw"])
        ret["rhs"]["tdw"].update(data["tdw"])
    else:
        _, lhs_data = compile_chunks(*chunks[:-1], sort=lhs)
        _, rhs_data = compile_chunks(chunks[-1], sort=rhs)
        k_lhs = path(lhs); k_rhs = path(rhs)
        for c, w in zip(chunks[:-1], weights[:-1]):
            kc = k_lhs.link(Key(c._name_), k_lhs.size)
            lhw[kr.link(kc, 0)] = w
        c, w = chunks[-1], weights[-1]
        kc = k_rhs.link(Key(c._name_), k_rhs.size)
        rhw[kr.link(kc, 0)] = w
        ret["lhs"]["ciw"].update(lhs_data["ciw"])
        ret["lhs"]["tdw"].update(lhs_data["tdw"])
        ret["rhs"]["ciw"].update(rhs_data["ciw"])
        ret["rhs"]["tdw"].update(rhs_data["tdw"])
    return ret


def compile_rules(*rules: Rule, sort: Rules, lhs: Chunks, rhs: Chunks) \
    -> tuple[list[Rule], RuleData]:
    new_rules = []
    rule_data = RuleData(
        riw={}, lhw={}, rhw={}, 
        lhs=ChunkData(ciw={}, tdw={}), 
        rhs=ChunkData(ciw={}, tdw={}))
    for rule in rules:
        rule._name_ = rule._name_ or f"r{next(sort._counter_)}"
        data = compile_rule(rule, sort, lhs, rhs)
        new_rules.append(rule)
        rule_data["riw"].update(data["riw"])
        rule_data["lhw"].update(data["lhw"])
        rule_data["rhw"].update(data["rhw"])
        rule_data["lhs"]["ciw"].update(data["lhs"]["ciw"])
        rule_data["lhs"]["tdw"].update(data["lhs"]["tdw"])
        rule_data["rhs"]["ciw"].update(data["rhs"]["ciw"])
        rule_data["rhs"]["tdw"].update(data["rhs"]["tdw"])
        for inst in instantiations(rule):
            inst._name_ = inst._name_ or f"r{next(sort._counter_)}" 
            _data = compile_rule(inst, sort, lhs, rhs)
            new_rules.append(inst)
            rule_data["riw"].update(_data["riw"])
            rule_data["lhw"].update(_data["lhw"])
            rule_data["rhw"].update(_data["rhw"])
            rule_data["lhs"]["ciw"].update(_data["lhs"]["ciw"])
            rule_data["lhs"]["tdw"].update(_data["lhs"]["tdw"])
            rule_data["rhs"]["ciw"].update(_data["rhs"]["ciw"])
            rule_data["rhs"]["tdw"].update(_data["rhs"]["tdw"])
    return new_rules, rule_data


class ByKwds(TypedDict):
    by: KeyForm
    b: NotRequired[int]


def keyform(branch: Branch, *, trunc: int = 0) -> KeyForm:
    match branch:
        case Atom():
            k, h = path(branch), 0 - trunc
        case Sort():
            k, h = path(branch), 1 - trunc
        case Family():
            k, h = path(branch), 2 - trunc
        case _:
            raise TypeError()
    if h < 0:
        raise ValueError("Truncation too deep")
    return KeyForm(k, (h,))
