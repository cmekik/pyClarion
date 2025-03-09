from typing import (Self, Iterable, Type, Iterator, TypedDict, get_type_hints, 
    overload, Sequence)
from weakref import WeakValueDictionary
from itertools import product, count

from .numdicts import KeyForm, Key, path, bind, ValidationError
from .numdicts.keyspaces import KSBase, KSRoot, KSNode, KSChild


class Root(KSRoot["Symbol"]):
    pass


class Symbol(KSChild):
    """
    Base class for data symbols.
    
    Do not directly instantiate or subclass this class.
    """
    
    def __mul__(self: Self, other: "Symbol") -> Key:
        if isinstance(other, Symbol):
            return bind(~self, ~other)
        return NotImplemented


class Term(Symbol):
    """
    Base class for data terms.

    Data terms represent indvidual data elements of a model (e.g., individual 
    features, parameters etc.).
    
    Do not directly instantiate or subclass this class. Use `Atom`, `Chunk`, or 
    `Rule` instead.
    """

    def __pow__(self, other: "Term | Var | Iterable[Term]") -> "Chunk":
        if isinstance(other, (Term, Var)):
            return Chunk({(self, other): 1.0})
        else:
            return Chunk({(self, atom): 1.0 for atom in other})

    def __rpow__(self, other: "Term | Var") -> "Chunk":
        if isinstance(other, (Term, Var)):
            return Chunk({(other, self): 1.0})
        return NotImplemented


class Sort[C: Term](KSNode[C], Symbol):
    """
    A data sort.

    Represents a collection of data terms that are alike in content (e.g., 
    color terms, shape terms, etc.). 

    Direct instantiation or subclassing of this class is not recommended. Use 
    `Atoms`, `Chunks`, or `Rules` instead.
    """

    _mtype_: Type[C]
    _required_: frozenset[Key]
    _counter_: count
    _vars_: dict[str, "Var"]

    def __init__(self, name: str = "", mtype: Type[C] = Term) -> None:
        super().__init__(name)
        self._mtype_ = mtype
        cls = type(self)
        for name, typ in get_type_hints(cls).items():
            if isinstance(typ, type) and issubclass(typ, mtype):
                self[name] = typ()
                setattr(self, name, self[name])
        self._required_ = frozenset(self._members_)
        self._counter_ = count()
        self._vars_ = {}

    def __call__(self, name: str) -> "Var":
        return self._vars_.setdefault(name, Var(name, self))

    def __delattr__(self, name: str) -> None:
        if Key(name) in self._required_:
            raise ValidationError(f"Cannot remove required key '{name}'")

    def __rmul__(self, other: Key) -> Key:
        if isinstance(other, Key):
            return bind(other, path(self))
        return NotImplemented


class Family[S: Sort](KSNode[S], Symbol):
    """
    A family of data sorts.

    Represents a collection of data terms that are alike in content (e.g., 
    color terms, shape terms, etc.). 
    """

    _mtype_: Type[S]
    _required_: frozenset[Key]

    def __init__(self, name: str = "", mtype: Type[S] = Sort) -> None:
        super().__init__(name)
        cls = type(self)
        self._mtype_ = mtype
        for name, typ in get_type_hints(cls).items():
            if isinstance(typ, type) and issubclass(typ, mtype):
                self[name] = typ()
                setattr(self, name, self[name])
        self._required_ = frozenset(self._members_)


class Atom(Term):
    """
    An atomic data term.

    Represents some basic data element (e.g., a feature, a parameter, etc.).
    """
    
    def __rmul__(self, other: Key) -> Key:
        if isinstance(other, Key):
            return bind(other, path(self))
        return NotImplemented


class Compound(Term):
    """
    Base class for compound terms.

    A compound term represents a data element with some constituency structure 
    (i.e., chunks and rules).

    Do not directly instantiate this class.
    """
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
        self._name_ = other
        return self


class Chunk(Compound):
    """
    A chunk term.
    
    Symbolically represents a Clarion chunk, together with its dimension-value 
    pairs.
    """
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
    
    @overload
    def __rmul__(self: Self, other: float) -> Self:
        ...

    @overload
    def __rmul__(self: Self, other: Key) -> Key:
        ...

    def __rmul__(self, other):
        if isinstance(other, float):
            return type(self)({d: other * w for d, w in self._dyads_.items()})
        elif isinstance(other, Key):
            return bind(other, path(self))
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

    def __rrshift__(self, others: Sequence["Chunk"]) -> "Rule":
        d = {}
        for other in others:
            if isinstance(other, Chunk):
                d[other] = 1.0
            else:
                return NotImplemented
        return Rule(d)


class Rule(Compound):
    """
    A rule term.
    
    Symbolically represents a Clarion rule, together with its constituent 
    chunks.
    """
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

    def __rmul__(self, other: Key) -> Key:
        if isinstance(other, Key):
            return bind(other, path(self))
        return NotImplemented

class Var:
    def __init__(self, name: str, sort: Sort) -> None:
        self.name = name
        self.sort = sort

    def __call__(self, valuation: dict[str, Term]) -> Term:
        if (value := valuation[self.name]) in self.sort:
            return value
        raise ValueError()


class Atoms(Sort[Atom]):
    """
    A data sort for atomic terms.

    Represents a collection of atomic data terms that are alike in content 
    (e.g., color terms, shape terms, etc.).
    """ 
    def __init__(self, name: str = ""):
        super().__init__(name, Atom)


class Chunks(Sort[Chunk]):
    """
    A data sort for chunk terms.

    Represents a collection of chunk terms. This sort includes a `nil` term as 
    a necessary member.
    """
    nil: Chunk

    def __init__(self, name: str = ""):
        super().__init__(name, Chunk)


class Rules(Sort[Rule]):
    """
    A data sort for rule terms.

    Represents a collection of rule terms. This sort includes a `nil` term as a 
    necessary member.
    """
    nil: Rule 

    def __init__(self, name: str = ""):
        super().__init__(name, Rule)


@overload
def instantiations(
    term: Chunk, caches: Sequence[dict[frozenset[str], Chunk]]
) -> Iterator[Chunk]:
    ...

@overload
def instantiations(
    term: Rule, caches: Sequence[dict[frozenset[str], Chunk]]
) -> Iterator[Rule]:
    ...


def instantiations(
    term: Chunk | Rule, caches: Sequence[dict[frozenset[str], Chunk]]
) -> Iterator[Chunk | Rule]:
    if not term._vars_:
        return
    for v in valuations(term):
        yield instantiate(term, v, caches)


def valuations(term: Chunk | Rule) -> Iterator[dict[str, Term]]:
    for vals in product(*term._vars_.values()):
        yield {lb: sp[v] for v, (lb, sp) in zip(vals, term._vars_.items())}


@overload
def instantiate(
    term: Chunk,
    vals: dict[str, Term], 
    caches: Sequence[dict[frozenset[str], Chunk]]    
) -> Chunk:
    ...

@overload
def instantiate(
    term: Rule, 
    vals: dict[str, Term], 
    caches: Sequence[dict[frozenset[str], Chunk]]
) -> Rule:
    ...

def instantiate(
    term: Chunk | Rule, 
    vals: dict[str, Term], 
    caches: Sequence[dict[frozenset[str], Chunk]]
) -> Chunk | Rule:
    if isinstance(term, Rule):
        items = term._chunks_.items()
        chunks = {instantiate(c, vals, [cache]) if c._vars_ else c: w 
            for (c, w), cache in zip(items, caches, strict=True)}
        return type(term)(chunks, term)
    if isinstance(term, Chunk):
        if len(caches) != 1:
            raise ValueError(f"Expected only one cache, got {len(caches)}")
        cache, = caches
        try:
            return cache[frozenset(term._vars_)]
        except KeyError:
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
        for inst in instantiations(chunk, [{}]):
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
        caches = [{} for _ in rule._chunks_]
        for inst in instantiations(rule, caches):
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
            for cache, chunk in zip(caches, inst._chunks_):
                cache[frozenset(chunk._vars_)] = chunk
    return new_rules, rule_data


def keyform(Symbol: Symbol, *, trunc: int = 0) -> KeyForm:
    match Symbol:
        case Atom():
            k, h = path(Symbol), 0 - trunc
        case Sort():
            k, h = path(Symbol), 1 - trunc
        case Family():
            k, h = path(Symbol), 2 - trunc
        case _:
            raise TypeError()
    if h < 0:
        raise ValueError("Truncation too deep")
    return KeyForm(k, (h,))


def describe_dyad(d: Term | Var, v: Term | Var, w: float) -> str:
    if isinstance(d, Term):
        key = path(d)
        s_d = ".".join([label for label, _ in key[-2:]]) 
    else:
        s_d = f"{d.sort._name_}('{d.name}')"
    if isinstance(v, Term):
        key = path(v)
        s_v = ".".join([label for label, _ in key[-2:]]) 
    else:
        s_v = f"{v.sort._name_}('{v.name}')"
    sign = "+" if w >= 0 else "-"
    if abs(w) == 1.0:
        return f"{sign} {s_d} ** {s_v}"
    else:           
        return f"{sign}{abs(w)} * {s_d} ** {s_v}"


def describe_chunk(chunk: Chunk):
    key = path(chunk)
    descr = chunk._descr_
    instances = chunk._instances_
    template = chunk._template_
    dyads = chunk._dyads_
    data = [f"chunk {key}"]
    if descr:
        data.append(f"'''{descr}'''")
    if instances:
        data.append(f"Abstract chunk with {len(instances)} instances")
    if template:
        data.append(f"Instance of chunk {path(template)}")
    for (d, v), w in dyads.items():
        data.append(describe_dyad(d, v, w))
    if not dyads:
        data.append("Empty chunk")
    return "\n    ".join(data)


def describe_rule(rule: Rule):
    key = path(rule)
    descr = rule._descr_
    instances = rule._instances_
    template = rule._template_
    chunks = rule._chunks_
    data = [f"rule {key}"]
    if descr:
        data.append(f"'''{descr}'''")
    if instances:
        data.append(f"Abstract rule with {len(instances)} instances")
    if template:
        data.append(f"Instance of rule {path(template)}")
    for c in chunks:
        data.append(describe_chunk(c).replace("\n", "\n    "))
    if chunks:
        data.insert(-1, ">>")
    else:
        data.append("Empty rule")
    return "\n    ".join(data)


def describe(knowledge: Chunk | Rule) -> str:
    match knowledge:
        case Chunk():
            return describe_chunk(knowledge)
        case Rule():
            return describe_rule(knowledge)
        case _:
            raise TypeError()
