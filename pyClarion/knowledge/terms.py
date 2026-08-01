__all__ = ["Atom", "Param", "Indexical", "this", "Var", "MatchVar", "Compound", 
    "Chunk", "ChunkData", "Rule", "RuleData", "Bus"]


from typing import Self, Iterator, Iterable, Sequence, TypedDict, overload, cast
from weakref import WeakSet
from itertools import product, count
from contextlib import contextmanager
import enum

from .base import Sort, Term
from ..numdicts.keys import Key, KeyForm
from ..numdicts.keyspaces import Unit, Child, Dyad, Triad, find_parent


class Atom(Term):
    """
    An atomic data term.

    Represents some basic data element (e.g., a feature, a parameter, etc.).
    """
    pass


class Param(Term):
    """
    An atomic parameter term.

    Represents some basic model parameter.
    """

    init_val: float

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.init_val = float("nan")


class Indexical(enum.Enum):
    chunk = enum.auto()
    rule = enum.auto()
    chunk_instance = enum.auto()
    rule_instance = enum.auto()


this = Indexical


class Var[S: Sort](Unit):
    """A variable data term."""
    name: str
    sort: S
    _subset: frozenset[Term]

    def __init__(self, name: str, sort: S) -> None:
        self.name = name
        self.sort = sort
        self._subset = frozenset()

    def __call__(self, valuation: dict[str, Term]) -> Term:
        term = valuation[self.name]
        if self.validate(term):
            return term
        raise ValueError(f"Value {term} assigned to Var {self} does not "
            "belong to the correct sort.")

    def __contains__(self, key: str | Key) -> bool:
        try:
            term = self.sort._members_[Key(key)]
        except KeyError:
            return False
        return self.validate(term)

    def __iter__(self) -> Iterator[str]:
        for key in self._iter_(1):
            yield key[1][0]

    def __invert__(self) -> Key:
        return ~self.sort

    def __mul__[T: Child](self: Self, other: T) -> "Dyad[Self, T]":
        """Return the keyspace obtained by crossing self with other"""
        if isinstance(other, Child):
            return Dyad((self, other))
        return NotImplemented

    def _keyform_(self, *hs: int) -> KeyForm:
        return self.sort._keyform_(*hs)
    
    def _iter_(self, *hs: int) -> Iterator[Key]:
        h, = hs
        if h <= 0:
            return
        if h == 1:
            for key in self.sort._members_:
                if key in self:
                    yield key 
    
    def validate(self, term: Term) -> bool:
        if find_parent(term) != self.sort:
            return False
        if self._subset and term not in self._subset:
            return False
        return True

    @contextmanager
    def subset(self, keys: Iterable[Term]):
        if keys:
            _subset = self._subset
            self._subset = self._subset & frozenset(keys)
            yield self
            self._subset = _subset
        else:
            yield self


class MatchVar[C: "Compound"]:
    term: C
    variables: tuple[Var, ...]
 
    def __init__(self, term: C, *variables: Var) -> None:
        self.term = term
        self.variables = variables


type Constant = Term | Indexical
type Variable = Var | MatchVar
type Symbol = Constant | Variable


class Bus(Term):
    """
    A data line.
    
    Represents some address for activations.

    """

    def __pow__(self, other: Symbol | Iterable[Symbol]) -> "Chunk":
        if isinstance(other, (Term, Indexical, Var, MatchVar)):
            return Chunk({(self, other): 1.0})
        else:
            return Chunk({(self, atom): 1.0 for atom in other})


class Compound(Term):
    """
    Base class for compound terms.

    A compound term represents a data element with some constituency structure 
    (i.e., chunks and rules).

    Do not directly instantiate this class.
    """
    _vars_: set[Var]
    _valuation_: frozenset[tuple[Var, Term]]
    _instances_: WeakSet[Self]
    _template_: Self | None
    _counter_: count


    def __init__(self, template: Self | None = None) -> None:
        super().__init__()
        self._vars_ = set()
        self._valuation_ = frozenset()
        self._instances_ = WeakSet()
        self._template_ = template
        self._counter_ = count()

    def __rxor__(self: Self, other: str) -> Self:
        if not other.isidentifier():
            ValueError("Compound term identifier must be a valid "
                "python identifier")
        self._name_ = other
        return self

    def __call__(self, *variables: Var) -> MatchVar[Self]:
        return MatchVar(self, *variables)
    
    def _match_(self, vals: dict[Var, Term]) -> set[Self]:
        candidates = set(self._instances_)
        for var, val in vals.items():
            elim = {_t for _t in candidates if not (var, val) in _t._valuation_}
            candidates = {_t for _t in candidates if _t not in elim}
        return candidates

    @staticmethod
    def _collect_vars_(dyads: Iterable[tuple[Symbol, Symbol]]) -> set[Var]:
        _vars_ = set()
        for dyad in dyads:
            for elt in dyad:
                if not isinstance(elt, Var):
                    continue
                _vars_.add(elt)
        return _vars_

    def _instantiations_(self) -> Iterator[Self]:
        for vals in self._valuations_():
            yield self._instantiate_(vals)

    def _valuations_(self) -> Iterator[dict[Var, Term]]:
        _vars_ = list(self._vars_)
        for vals in product(*_vars_):
            if vals:
                yield {var: var.sort[val] for var, val in zip(_vars_, vals)}

    def _instantiate_(self: Self, vals: dict[Var, Term]) -> Self:
        raise NotImplementedError()


class ChunkData(TypedDict):
    ciw: "dict[Dyad[Chunk, Chunk], float]"
    tdw: "dict[Triad[Chunk, Bus, Term], float]"


class Chunk(Compound):
    """
    A chunk term.
    
    Symbolically represents a Clarion chunk, together with its dimension-value 
    pairs.
    """
    _dyads_: dict[tuple[Bus, Symbol], float]
    _rule_: "Rule | None"

    @overload
    def __init__(self, dyads: dict[tuple[Bus, Symbol], float]) -> None:
        ...
    
    @overload
    def __init__(
        self: Self, 
        dyads: dict[tuple[Bus, Constant], float], 
        template: Self
    ) -> None:
        ...

    def __init__(self, dyads=None, template=None) -> None:
        dyads = dyads or {}
        super().__init__(template)
        self._dyads_ = dyads
        self._vars_.update(self._collect_vars_(dyads))

    def __str__(self) -> str:
        try:
            key = ~self
        except AttributeError:
            key = hex(id(self))
        instances = self._instances_
        template = self._template_
        dyads = self._dyads_
        data = [f"chunk {key}"]
        if instances:
            data.append(f"Abstract chunk with {len(instances)} instances")
        if template:
            try:
                kt = ~template
            except AttributeError:
                kt = hex(id(template))
            data.append(f"Instance of chunk {kt}")
        for (d, v), w in dyads.items():
            data.append(self._str_dyad_(d, v, w))
        if not dyads:
            data.append("Empty chunk")
        return "\n    ".join(data)

    @classmethod
    def _str_constituent_(cls, x: Symbol) -> str:
        match x:
            case Term():
                key = ~x
                return ".".join([label for label, _ in key[-2:]])
            case Indexical():
                return f"this.{x.name}"
            case Var():
                return f"{x.sort._name_}('{x.name}')"
            case MatchVar():
                vs = ', '.join([cls._str_constituent_(v) for v in x.variables])
                return f"{x.term._name_}({vs})"

    @classmethod
    def _str_dyad_(cls, b: Bus, s: Symbol, w: float) -> str:
        s_b = cls._str_constituent_(b)
        s_s = cls._str_constituent_(s)
        sign = "+" if w >= 0 else "-"
        if abs(w) == 1.0:
            return f"{sign} {s_b} ** {s_s}"
        else:           
            return f"{sign}{abs(w)} * {s_b} ** {s_s}"

    def __pos__(self: Self) -> Self:
        return self
    
    def __neg__(self: Self) -> Self:
        return type(self)({d: -w for d, w in self._dyads_.items()})
    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
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

    def __rrshift__(self, others: Sequence["Chunk"]) -> "Rule":
        d = {}
        for other in others:
            if isinstance(other, Chunk):
                d[other] = 1.0
            else:
                return NotImplemented
        d[self] = 1.0
        return Rule(d)

    def _evaluate_var_(self, x: Symbol, vals: dict[Var, Term]) \
        -> Iterable[Constant]:
        if isinstance(x, (Term, Indexical)):
            return (x,)
        elif isinstance(x, Var):
            return (vals[x],)
        elif isinstance(x, MatchVar):
            return x.term._match_(vals)
        else:
            raise ValueError(f"Unexpected consitutent '{x}' in symbolic chunk " 
                "annotation.")

    def _evaluate_indexicals_(self, x: Constant) -> Term:
        if isinstance(x, Term):
            return x
        elif x is this.chunk:
            return self if self._template_ is None else self._template_
        elif x is this.rule:
            rule = (self._rule_ if self._template_ is None 
                else self._template_._rule_)
            if rule is None:
                raise ValueError()
            return rule
        elif x is this.chunk_instance:
            return self
        elif x is this.rule_instance:
            rule = self._rule_
            if rule is None:
                raise ValueError()
            return rule
        raise TypeError()

    def _instantiate_(self: Self, vals: dict[Var, Term]) -> Self:
        num = next(self._counter_)
        try:
            name = f"{self._name_}_{num}"
        except AttributeError as e:
            raise AttributeError("Unnamed abstract chunk.") from e
        dyads: dict[tuple[Bus, Constant], float] = {} 
        for (bus, symbol), w in self._dyads_.items():
            constants = self._evaluate_var_(symbol, vals)
            for constant in constants:
                dyads[(bus, constant)] = w
        inst = type(self)(dyads, self)
        inst._name_ = name
        inst._valuation_ = frozenset(vals.items())
        self._instances_.add(inst)
        return inst

    def _compile_(self) -> "ChunkData":
        ciw: dict[Dyad[Chunk, Chunk], float] = {}
        tdw: dict[Triad[Chunk, Bus, Term], float] = {}
        kt, kc = (cast(Chunk, self),) * 2
        if self._template_ is not None:
            kt = cast(Chunk, self._template_)
        ciw[kc * kt] = 1.0
        ciw[kc * kc] = 1.0
        if not self._vars_:
            for (bus, symbol), w in self._dyads_.items():
                assert not isinstance(symbol, (Var, MatchVar))
                term = self._evaluate_indexicals_(symbol)
                kw = kc * bus * term
                tdw[kw] = w 
        return ChunkData(ciw=ciw, tdw=tdw)


class RuleData(TypedDict):
    riw: dict[Key, float]
    lhw: dict[Key, float]
    rhw: dict[Key, float]


class Rule(Compound):
    """
    A rule term.
    
    Symbolically represents a Clarion rule, together with its constituent 
    chunks.
    """
    _name__ : str
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
        for chunk in chunks:
            chunk._rule_ = self

    def __str__(self) -> str:
        try:
            key = ~self
        except AttributeError:
            key = hex(id(self))
        instances = self._instances_
        template = self._template_
        chunks = self._chunks_
        data = [f"rule {key}"]
        if instances:
            data.append(f"Abstract rule with {len(instances)} instances")
        if template:
            try:
                kt = ~template
            except AttributeError:
                kt = hex(id(template))
            data.append(f"Instance of rule {kt}")
        for c in chunks:
            data.append(str(c).replace("\n", "\n    "))
        if chunks:
            data.insert(-1, ">>")
        else:
            data.append("Empty rule")
        return "\n    ".join(data)

    @property
    def _name_(self) -> str:
        return self._name__

    @_name_.setter
    def _name_(self, name: str) -> None:
        try:
            self._name_
        except AttributeError:
            pass
        else:
            try:
                self._parent_
            except AttributeError:
                pass
            else:
                raise ValueError("Name already set.")
        self._name__ = name
        for i, chunk in enumerate(self._chunks_):
            try:
                chunk._name_
            except AttributeError:
                chunk._name_ = f"{self._name_}_{i}"

    def _instantiate_(self: Self, vals: dict[Var, Term]) -> Self:
        num = next(self._counter_)
        try: 
            name = f"{self._name_}_{num}"
        except AttributeError as e:
            raise AttributeError("Unnamed abstract rule.") from e
        chunks = {}
        for c, w in self._chunks_.items():
            if not c._vars_:
                chunks[c] = w
                continue
            matches = {i for i in c._instances_ 
                if i._valuation_.issubset(vals.items())}
            if len(matches) < 1:
                raise ValueError("No match")
            if 1 < len(matches):
                raise ValueError("Vals does not select unique instance")
            i, = matches 
            chunks[i] = w
        inst = type(self)(chunks, self)
        inst._name_ = name
        inst._valuation_ = frozenset(vals.items())
        self._instances_.add(inst)
        return inst

    def _compile_(self: "Rule") -> RuleData:
        riw = {}; lhw = {}; rhw = {}
        kt, kr = (~self,) * 2
        if self._template_ is not None:
            kt = ~self._template_
        riw[kr * kt] = 1.0
        riw[kr * kr] = 1.0
        if not self._vars_:
            chunks, weights = zip(*self._chunks_.items())
            for c, w in zip(chunks[:-1], weights[:-1]):
                lhw[~c * kr] = w
            c, w = chunks[-1], weights[-1]
            rhw[kr * ~c] = w
        return RuleData(riw=riw, lhw=lhw, rhw=rhw)
