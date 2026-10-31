__all__ = ["Compound", "BusStructure", "DataStructure", "Dyads", "Nodes", 
    "Chunk", "Rule", "Compounds", "Chunks", "Rules", "ChunkFamily", 
    "RuleFamily"]


from typing import Final
from weakref import ref
from textwrap import indent
from collections import deque
from contextvars import ContextVar, Token

from .expressions import Bus, Buses, BusFamily, ChunkExpr, RuleExpr
from .language import (Var, Constraint, Indexical, Symbol, Scope, Primitives, 
    Compound, Compounds, this, VarNode, Instance, GroundInstance, TypedInstance, 
    GenericInstance, TypedGenericForm, AbstractForm, TypedForm, GenericForm)
from .base import (Family, Sort, Term)
from ..numdicts.keyspaces import Dyad, find_parent
from ..numdicts.keys import Key


type BusStructure = BusFamily | Buses | Bus
type DataStructure = Family[Sort[Term]] | Sort[Term]
type Dyads = Dyad[BusStructure, DataStructure]
type Nodes = Dyads | Rules | Chunks | Primitives


class Chunk[N: VarNode](Compound[ChunkExpr[N]]):
    """
    A chunk term.
    
    Symbolically represents a Clarion chunk, together with its dimension-value 
    pairs.
    """
    _dyads_: dict[tuple[Bus, Term], float]

    @property
    def _rule_(self) -> "Rule | None":
        if self._rule_ref_ is None:
            return None
        rule = self._rule_ref_()
        if rule is None:
            raise RuntimeError("Orphaned rule-chunk")
        return rule

    def __str__(self) -> str:
        try:
            key = ~self
        except AttributeError:
            key = hex(id(self))
        data = [f"chunk {key}"]
        expr = self._expr_
        if expr is not None:
            if self._is_abstract_:
                num = len(self._instances_)
                data.append(indent(f"Has {num} instances", "    "))
            lines = self.lines()
            if lines:
                data.append(indent(lines, "    "))
            else:
                data.append(indent("(empty)", "    "))
        return "\n".join(data)

    def lines(self) -> str:
        if self._expr_ is None:
            return ""
        lines = []
        for (d, v), w in self._dyads_.items():
            lines.append(self._expr_._str_dyad_(d, v, w))
        return '\n'.join(lines)

    def _init_(self) -> None:
        try:
            rule = RULES.get()
        except LookupError as err:
            rule = None
        self._rule_ref_ = ref(rule) if rule is not None else None
        self._dyads_ = self._init_dyads_()

    def _init_dyads_(self) -> dict[tuple[Bus, Term], float]:
        if self._expr_ is None:
            return {}
        dyads = {}
        for (t, s), w in self._expr_.items():
            for dyad in self._interpret_dyad_(t, s):
                dyads[dyad] = w
        return dyads

    def _interpret_dyad_(self, t: Bus, s: Symbol) -> list[tuple[Bus, Term]]:
        match s:
            case Term():
                return [(t, s)]
            case Indexical():
                match s:
                    case this.chunk | this.chunk_group | this.chunk_instance:
                        value = self._interpret_indexical_(s)
                        if value is None:
                            return []
                        return [(t, value)]
                    case this.rule | this.rule_group | this.rule_instance:
                        if self._rule_ is None:
                            raise ValueError("Cannot resolve rule indexical")
                        value = self._rule_._interpret_indexical_(s)
                        if value is None:
                            return []
                        return [(t, value)]
                    case _:
                        raise ValueError(f"Unexpected object {s}")
            case Var():
                return self._interpret_var_(t, s)
            case Constraint():
                value = self._interpret_constraint_(s)
                if value is None:
                    return[]
                return [(t, value)]
            case _:
                raise ValueError(f"Unexpected symbol {s}")

    def _interpret_indexical_(self, i: Indexical) -> Term | None:
        match i:
            case this.chunk:
                chunk = self
                while chunk._template_ is not None:
                    chunk = chunk._template_
                return chunk
            case this.chunk_group:
                match self._form_:
                    case GroundInstance():
                        raise ValueError("Cannot resolve group indexical on "
                            "ground chunk")
                    case TypedInstance() | GenericInstance():
                        assert self._template_ is not None
                        return self._template_
                    case TypedForm() | TypedGenericForm():
                        return self
                    case GenericForm():
                        return None
                    case _:
                        raise TypeError("Unexpected chunk form")
            case this.chunk_instance:
                match self._form_:
                    case Instance():
                        return self
                    case AbstractForm():
                        return None
                    case _:
                        raise TypeError("Unexpected chunk form")
            case _:
                raise ValueError(f"Invalid chunk indexical {i}")

    def _interpret_var_(self, t: Bus, v: Var) -> list[tuple[Bus, Term]]:
        form = self._form_
        if form.node is not v.node:
            raise ValueError(f"Incompatible var {v}")
        match v.scope:
            case Scope.VAL:
                match form:
                    case TypedInstance() | GenericInstance():
                        return [(t, form.val)]
                    case AbstractForm():
                        return []
                    case _:
                        raise ValueError(f"Unexpected var {v}")
            case Scope.DIM:
                match form:
                    case TypedInstance() | TypedForm():
                        sort = form.node
                    case TypedGenericForm():
                        sort = form.val
                    case GenericInstance():
                        sort = form.val._parent_
                    case GenericForm():
                        return []
                    case _:
                        raise ValueError(f"Unexpected var {v}")
                return [(t, sort[x]) for x in sort] 
            case _:
                raise ValueError(f"Invalid variable scope {v.scope}")

    def _interpret_constraint_(self, s: Constraint) -> Term | None:
        self_form = self._form_
        symbol_form = s.term._form_
        if self_form.node is not symbol_form.node:
            raise ValueError(f"Incompatible constraint {s}")
        elif isinstance(self_form, GroundInstance):
            raise ValueError("Invalid constraint")
        elif isinstance(symbol_form, Instance):
            raise ValueError("Invalid constraint")
        match s.scope:
            case Scope.VAL:
                match (self_form, symbol_form):
                    case (AbstractForm(), _):
                        return None
                    case (TypedInstance(), TypedForm()):
                        return s.term._instances_[self_form.val]
                    case (GenericInstance(), TypedGenericForm()):
                        return s.term._instances_[self_form.val]
                    case (GenericInstance(), GenericForm()):
                        return (s.term
                            ._instances_[self_form.val._parent_]
                            ._instances_[self_form.val])
                    case _:
                        raise ValueError("Invalid constraint")
            case Scope.DIM:
                match (self_form, symbol_form):
                    case (GenericForm(), _):
                        return None
                    case (TypedInstance(), TypedForm()) \
                        | (TypedForm(), TypedForm()) \
                        | (GenericInstance(), TypedGenericForm()) \
                        | (TypedGenericForm(), TypedGenericForm()):
                        return s.term
                    case (GenericInstance(), GenericForm()):
                        return s.term._instances_[self_form.val._parent_]
                    case (TypedGenericForm(), GenericForm()):
                        return s.term._instances_[self_form.val]
                    case _:
                        raise ValueError("Invalid constraint")

    def instantiate(self, val: Sort[Term] | Term) -> "Chunk[N]":
        form = self._form_
        match (form, val):
            case (GenericForm(), Sort()):
                new_form = form.instantiate(val)
            case (TypedForm() | TypedGenericForm(), Term()):
                new_form = form.instantiate(val)
            case (Instance(), _):
                raise TypeError(f"Cannot instantiate concrete chunk {self}")
            case _:
                raise TypeError(f"Unexpected valuation {val} for chunk {self}")
        try:
            return self._instances_[val]
        except KeyError:
            assert self._expr_ is not None
            instance = type(self)(self._expr_, new_form, self)
            self._instances_[val] = instance
            return instance

    def compile(self) -> tuple[dict[Key, float], dict[Key, float]]:
        ciw: dict[Key, float] = {}
        tdw: dict[Key, float] = {}
        kt, kc = (self,) * 2
        if self._template_ is not None:
            kt = self._template_
        ciw[~kc * ~kt] = 1.0
        ciw[~kc * ~kc] = 1.0
        for (bus, term), w in self._dyads_.items():
            kw = ~kc * ~bus * ~term
            tdw[kw] = w
        return tdw, ciw

    @classmethod
    def chunk_factory[T: VarNode](
        cls: "type[Chunk[T]]", 
        expr: ChunkExpr[T]
    ) -> "Chunk[T]":
        sort, value = cls._factory(expr)
        new = deque([value])
        while new:
            obj = new.popleft()
            sort[obj._name_] = obj
            match (form := obj._form_):
                case Instance():
                    continue
                case GenericForm():
                    for val_name in form.node:
                        val = form.node[val_name]
                        new.append(obj.instantiate(val))
                case TypedForm():
                    for val_name in form.node:
                        val = form.node[val_name]
                        new.append(obj.instantiate(val))
                case TypedGenericForm():
                    for val_name in form.val:
                        val = form.val[val_name]
                        new.append(obj.instantiate(val))
                case _:
                    raise TypeError("Unexpected chunk form")
        return value


class Chunks(Compounds[Chunk]):
    """
    A data sort for chunk terms.

    Represents a collection of chunk terms. This sort includes a `nil` term as 
    a necessary member.
    """

    _dyads_: Final[Dyads]
    _: Chunk # This is the nil chunk

    def __init__(self, d: Dyads, name: str = "", prefix: str = "chunk") -> None:
        super().__init__(name, prefix)
        self._dyads_ = d
        try:
            self._
        except KeyError:
            self._ = Chunk()

    @classmethod
    def _get_member_type_(cls) -> type[Chunk]:
        return Chunk


class ChunkFamily(Family[Chunks]):
    _init_required_ = False

    @classmethod
    def _get_member_type_(cls) -> type[Chunks]:
        return Chunks


def chunk(expr: ChunkExpr) -> Chunk:
    return Chunk.chunk_factory(expr)


RULES: ContextVar["Rule"] = ContextVar("RULES")


class Rule[N: VarNode](Compound[RuleExpr[N]]):
    """
    A rule term.
    
    Symbolically represents a Clarion rule, together with its constituent 
    chunks.
    """
    _lhs_ : dict[Chunk, float]
    _rhs_: dict[Chunk, float]
    __tokens_: list[Token]

    def __str__(self) -> str:
        try:
            key = ~self
        except AttributeError:
            key = hex(id(self))
        data = [f"rule {key}"]
        if self._expr_ is not None:
            lhs = self._lhs_
            rhs = self._rhs_
            num = len(self._instances_)
            if self._is_abstract_:
                data.append(indent(f"Has {num} instances", "    "))
            if not lhs and not rhs:
                data.append(indent("(empty)", "   "))
            else:
                for c in lhs:
                    data.append(indent(c.lines(), "    "))
                if not lhs:
                    data.append(indent("(empty LHS)", "   "))
                data.append(indent(">>", "    "))
                for c in rhs:
                    data.append(indent(c.lines(), "    "))
                if not rhs:
                    data.append(indent("(empty RHS)", "    "))
        return "\n".join(data)

    def __enter__(self):
        self.__tokens_.append(RULES.set(self))
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        RULES.reset(self.__tokens_.pop())

    def _init_(self) -> None:
        self.__tokens_ = []
        if self._expr_ is None:
            self._lhs_ = {}
            self._rhs_ = {}
        else:
            template = self._template_
            self._lhs_ = self._init_chunks_(
                spec=self._expr_._lhs,
                template=None if template is None else template._lhs_)
            self._rhs_ = self._init_chunks_(
                spec=self._expr_._rhs,
                template=None if template is None else template._rhs_)

    def _init_chunks_(self, 
        spec: dict[ChunkExpr, float],
        template: dict[Chunk, float] | None = None
    ) -> dict[Chunk, float]:
        match self._form_:
            case GroundInstance() | TypedForm() | GenericForm():
                with self:
                    return {Chunk(expr): w for expr, w in spec.items()}
            case TypedGenericForm():
                level = Scope.DIM
            case TypedInstance() | GenericInstance():
                level = Scope.VAL
            case _:
                raise ValueError(f"Unexpected rule form {self._form_}")
        assert template is not None
        chunks = {}
        for chunk, w in template.items():
            new = chunk
            expr = chunk._expr_
            assert expr is not None
            with self:
                match chunk._form_:
                    case Instance():
                        if expr.scope & level:
                            new = Chunk(expr)
                    case AbstractForm():
                        if Scope.VAL in expr.scope \
                            or (expr.scope and level is Scope.DIM):
                            new = chunk.instantiate(self._form_.val)
                    case _:
                        raise TypeError(f"Unexpected chunk form {chunk._form_}")
            chunks[new] = w
        return chunks

    def _interpret_indexical_(self, i: Indexical) -> Term | None:
        match i:
            case this.rule:
                rule = self
                while rule._template_ is not None:
                    rule = rule._template_
                return rule
            case this.rule_group:
                match self._form_:
                    case GroundInstance():
                        raise ValueError("Cannot resolve group indexical on "
                            "ground rule")
                    case TypedInstance() | GenericInstance():
                        assert self._template_ is not None
                        return self._template_
                    case TypedForm() | TypedGenericForm():
                        return self
                    case GenericForm():
                        return None
            case this.rule_instance:
                match self._form_:
                    case Instance():
                        return self
                    case AbstractForm():
                        return None
            case _:
                raise ValueError(f"Invalid rule indexical {i}")

    def instantiate(self, val: Sort[Term] | Term) -> "Rule[N]":
        form = self._form_
        match (form, val):
            case (GenericForm(), Sort()):
                new_form = form.instantiate(val)
            case (TypedForm() | TypedGenericForm(), Term()):
                new_form = form.instantiate(val)
            case (Instance(), _):
                raise TypeError(f"Cannot instantiate concrete rule {self}")
            case _:
                raise TypeError(f"Unexpected valuation {val} for rule {self}")
        try:
            return self._instances_[val]
        except KeyError:
            assert self._expr_ is not None
            instance = type(self)(self._expr_, new_form, template=self)
            self._instances_[val] = instance
            return instance

    # def compile(self) \
    #     -> tuple[dict[Key, float], dict[Key, float], dict[Key, float]]:
    #     riw = {}; lhw = {}; rhw = {}
    #     kt, kr = (~self,) * 2
    #     if self._template_ is not None:
    #         kt = ~self._template_
    #     riw[kr * kt] = 1.0
    #     riw[kr * kr] = 1.0
    #     if not self._vars_:
    #         chunks, weights = zip(*self._chunks_.items())
    #         for c, w in zip(chunks[:-1], weights[:-1]):
    #             lhw[~c * kr] = w
    #         c, w = chunks[-1], weights[-1]
    #         rhw[kr * ~c] = w
    #     return RuleData(riw=riw, lhw=lhw, rhw=rhw)

    @classmethod
    def rule_factory(cls, expr: RuleExpr[N]) -> "Rule[N]":
        sort, value = Rule._factory(expr)
        assert isinstance(sort, Rules) and isinstance(value, Rule)
        new = deque([value])
        while new:
            obj = new.popleft()
            sort[obj._name_] = obj
            for chunk in obj._lhs_:
                try:
                    find_parent(chunk)
                except AttributeError:
                    sort._lhs_[chunk._name_] = chunk
            for chunk in obj._rhs_:
                try:
                    find_parent(chunk)
                except AttributeError:
                    sort._rhs_[chunk._name_] = chunk
            match form := obj._form_:
                case Instance():
                    continue
                case GenericForm():
                    assert obj._expr_ is not None
                    assert obj._expr_.scope is not Scope.NIL
                    for name in form.node:
                        val = form.node[name]
                        if not isinstance(val, Compounds):
                            new.append(obj.instantiate(val))
                case TypedForm():
                    assert obj._expr_ is not None
                    if Scope.VAL not in obj._expr_.scope:
                        continue
                    for name in form.node:
                        val = form.node[name]
                        new.append(obj.instantiate(val))
                case TypedGenericForm():
                    assert obj._expr_ is not None
                    if Scope.VAL not in obj._expr_.scope:
                        continue
                    for name in form.val:
                        val = form.val[name]
                        new.append(obj.instantiate(val))
                case _:
                    raise TypeError("Unexpected rule form")
        return value


class Rules(Compounds[Rule]):
    """
    A data sort for rule terms.

    Represents a collection of rule terms. This sort includes a `nil` term as a 
    necessary member.
    """

    _lhs_: Final[Chunks]
    _rhs_: Final[Chunks]

    _: Rule # This is the nil rule

    def __init__(self, 
        lhs: Chunks, 
        rhs: Chunks, 
        name: str = "", 
        prefix: str = "rule"
    ) -> None:
        super().__init__(name, prefix)
        self._lhs_ = lhs
        self._rhs_ = rhs
        try:
            self._
        except KeyError:
            self._ = Rule()

    @classmethod
    def _get_member_type_(cls) -> type[Rule]:
        return Rule


class RuleFamily(Family[Rules]):
    _init_required_ = False

    @classmethod
    def _get_member_type_(cls) -> type[Rules]:
        return Rules


def rule[N: VarNode](expr: RuleExpr[N]) -> Rule[N]:
    return Rule.rule_factory(expr)





    
# class RuleData(TypedDict):
#     riw: dict[Key, float]
#     lhw: dict[Key, float]
#     rhw: dict[Key, float]

#     # def _compile_(self: "Rule") -> RuleData:
#     #     riw = {}; lhw = {}; rhw = {}
#     #     kt, kr = (~self,) * 2
#     #     if self._template_ is not None:
#     #         kt = ~self._template_
#     #     riw[kr * kt] = 1.0
#     #     riw[kr * kr] = 1.0
#     #     if not self._vars_:
#     #         chunks, weights = zip(*self._chunks_.items())
#     #         for c, w in zip(chunks[:-1], weights[:-1]):
#     #             lhw[~c * kr] = w
#     #         c, w = chunks[-1], weights[-1]
#     #         rhw[kr * ~c] = w
#     #     return RuleData(riw=riw, lhw=lhw, rhw=rhw)