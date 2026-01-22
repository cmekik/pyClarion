from typing import Self, Iterator, Iterable, Sequence, TypedDict, overload
from weakref import WeakSet
from itertools import product
import enum

from .base import Term, Var
from ..numdicts import Key


class Indexical(enum.Enum):
    chunk = enum.auto()
    rule = enum.auto()


this = Indexical


class Atom(Term):
    """
    An atomic data term.

    Represents some basic data element (e.g., a feature, a parameter, etc.).
    """
    pass


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


    def __init__(self, template: Self | None = None) -> None:
        super().__init__()
        self._vars_ = set()
        self._valuation_ = frozenset()
        self._instances_ = WeakSet()
        self._template_ = template

    def __rxor__(self: Self, other: str) -> Self:
        if not other.isidentifier():
            ValueError("Compound term identifier must be a valid "
                "python identifier")
        self._name_ = other
        return self

    @staticmethod
    def _collect_vars_(
        dyads: Iterable[tuple[Term | Indexical | Var, Term | Indexical | Var]]
    ) -> set[Var]:
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
    ciw: dict[Key, float]
    tdw: dict[Key, float]


class Chunk(Compound):
    """
    A chunk term.
    
    Symbolically represents a Clarion chunk, together with its dimension-value 
    pairs.
    """
    _dyads_: dict[tuple[Term | Indexical | Var, Term | Indexical | Var], float]
    _rule_: "Rule | None"

    @overload
    def __init__(
        self, 
        dyads: dict[tuple[Term | Indexical | Var, Term | Indexical | Var], float]
    ) -> None:
        ...
    
    @overload
    def __init__(
        self: Self, 
        dyads: dict[tuple[Term, Term], float], 
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

    @staticmethod
    def _str_constituent_(x: Term | Indexical | Var) -> str:
        match x:
            case Term():
                key = ~x
                return ".".join([label for label, _ in key[-2:]])
            case Indexical():
                return f"this.{x.name}"
            case Var():
                return f"{x.sort._name_}('{x.name}')"

    @classmethod
    def _str_dyad_(cls, 
        d: Term | Indexical |  Var, v: Term | Indexical | Var, w: float
    ) -> str:
        s_d = cls._str_constituent_(d)
        s_v = cls._str_constituent_(v)
        sign = "+" if w >= 0 else "-"
        if abs(w) == 1.0:
            return f"{sign} {s_d} ** {s_v}"
        else:           
            return f"{sign}{abs(w)} * {s_d} ** {s_v}"

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

    def _interpret_constitutent_(self, 
        x: Term | Indexical | Var, vals: dict[Var, Term]
    ) -> Term:
        if isinstance(x, Term):
            return x
        elif isinstance(x, Var):
            return vals[x]
        elif x is this.chunk:
            return self if self._template_ is None else self._template_
        elif x is this.rule:
            if self._rule_ is None:
                raise ValueError("Chunk has no parent rule.")
            return self._rule_
        else:
            raise ValueError(f"Unexpected consitutent '{x}' in symbolic chunk " 
                "annotation.")

    def _instantiate_(self: Self, vals: dict[Var, Term]) -> Self:
        dyads: dict[tuple[Term, Term], float] = {} 
        for (t1, t2), w in self._dyads_.items():
            lhs = self._interpret_constitutent_(t1, vals)
            rhs = self._interpret_constitutent_(t2, vals)
            dyads[(lhs, rhs)] = w
        inst = type(self)(dyads, self)
        inst._valuation_ = frozenset(vals.items())
        self._instances_.add(inst)
        return inst

    def _compile_(self) -> "ChunkData":
        ciw, tdw = {}, {}
        kt, kc = (~self,) * 2
        if self._template_ is not None:
            kt = ~self._template_
        ciw[kt * kc] = 1.0
        if not self._vars_:
            for (s1, s2), w in self._dyads_.items():
                t1 = self._interpret_constitutent_(s1, {})
                t2 = self._interpret_constitutent_(s2, {})
                kw = kc * ~t1 * ~t2
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

    def _instantiate_(self: Self, vals: dict[Var, Term]) -> Self:
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
        return type(self)(chunks, self)

    def _compile_(self: "Rule") -> RuleData:
        riw = {}; lhw = {}; rhw = {}
        kt, kr = (~self,) * 2
        if self._template_ is not None:
            kt = ~self._template_
        riw[kr * kt] = 1.0
        if not self._vars_:
            chunks, weights = zip(*self._chunks_.items())
            for c, w in zip(chunks[:-1], weights[:-1]):
                lhw[~c * kr] = w
            c, w = chunks[-1], weights[-1]
            rhw[kr * ~c] = w
        return RuleData(riw=riw, lhw=lhw, rhw=rhw)


type DataVar = Var #[Sort[Atom]] | Var[Sort[Chunk]] | Var[Sort[Rule]]
type Datamer = Atom | Chunk | Rule | Indexical | DataVar


class Bus(Term):
    """
    A data line.
    
    Represents some address for activations.
    """

    def __pow__(self, other: Datamer | Iterable[Datamer]) -> Chunk:
        if isinstance(other, (Atom, Chunk, Rule, Indexical, Var)):
            return Chunk({(self, other): 1.0})
        else:
            return Chunk({(self, atom): 1.0 for atom in other})
