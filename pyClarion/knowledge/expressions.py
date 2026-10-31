from typing import Self, Iterator, overload, final
from textwrap import indent

from .base import Family, Sort, Term
from .language import (Expr, Symbol, Scope, Var, Constraint, Indexical, VarNode, 
    Primitive, this)


class Bus(Primitive):
    """
    A data line.
    
    Represents some address for activations and constructs chunk expressions 
    via the exponential operator. 
    """

    def __pow__(self, other: Symbol | tuple[Symbol, ...]) -> "ChunkExpr":
        expr = ChunkExpr()
        match other:
            case Term() | Indexical():
                expr.update({(self, other): 1.0})
            case Var(node=n) | Constraint(node=n):
                expr = ChunkExpr(node=n)
                expr.update({(self, other): 1.0})
            case tuple():
                if not other:
                    raise ValueError("Cannot parse empty tuple")
                for obj in other:
                    if not isinstance(obj, (Term, Indexical, Var, Constraint)):
                        raise TypeError(f"Unexpected symbol type {type(obj)}")
                    expr = expr + self ** obj
            case _:
                return NotImplemented
        return expr            


class Buses(Sort[Bus]):
    """A data sort for data lines."""
    def __init__(self, name: str = "", prefix: str = "bus"):
        super().__init__(name, prefix)

    @classmethod
    def _get_member_type_(cls) -> type[Bus]:
        return Bus


class BusFamily(Family[Buses]):
    @classmethod
    def _get_member_type_(cls) -> type[Buses]:
        return Buses


@final
class ChunkExpr[N: VarNode](Expr[N]):

    def update(self, dyads: dict[tuple[Bus, Symbol], float]) -> None:
        for (bus, symbol), w in dyads.items():
            self._dyads[(bus, symbol)] = w
            match symbol:
                case Var(scope=Scope.DIM) \
                    | Constraint(scope=Scope.DIM) \
                    | this.chunk_group \
                    | this.rule_group:
                    self._scope |= Scope.DIM
                case Var(scope=Scope.VAL) \
                    | Constraint(scope=Scope.VAL) \
                    | this.chunk_instance \
                    | this.rule_instance:
                    self._scope |= Scope.VAL
                case _:
                    pass

    def __init__(self, 
        name: str = "", 
        node: N = None, 
        scope: Scope = Scope.NIL
    ) -> None:
        super().__init__(name, node, scope)
        self._dyads = {}

    def __str__(self) -> str:
        clsname = self.__class__.__name__
        lines = self.lines()
        return f"{clsname}(\n{indent(lines, '    ')})"

    def __copy__(self) -> Self:
        new = super().__copy__()
        new.update(self._dyads)
        return new

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        new = self.__copy__()
        new.update({k: -v for k, v in new._dyads.items()})
        return new

    def __rmul__(self, other: float) -> Self:
        new = self.__copy__()
        new.update({k: other * v for k, v in new._dyads.items()})
        return new

    @overload
    def __add__(self, other: "ChunkExpr[None]") -> "ChunkExpr[N]":
        ...

    @overload
    def __add__[T: VarNode](self, other: "ChunkExpr[T]") -> "ChunkExpr[T]":
        ...

    def __add__(self, other) -> "ChunkExpr":
        if not isinstance(other, ChunkExpr):
            return NotImplemented
        elif self.node is not None and other.node is not None \
            and self.node is not other.node:
            raise ValueError("Multiple variables not allowed in chunk expr")
        new = ChunkExpr(other.name, self._propagate(other))
        new.update(self._dyads)
        new.update(other._dyads)
        return new


    @overload
    def __sub__(self, other: "ChunkExpr[None]") -> "ChunkExpr[N]":
        ...

    @overload
    def __sub__[T: VarNode](self, other: "ChunkExpr[T]") -> "ChunkExpr[T]":
        ...

    def __sub__(self, other) -> "ChunkExpr":
        if not isinstance(other, ChunkExpr):
            return NotImplemented
        elif self.node is not None and other.node is not None \
            and self.node is not other.node:
            raise ValueError("Multiple variables not allowed in chunk expr")
        new = ChunkExpr(other.name, self._propagate(other))
        new.update(self._dyads)
        new.update({k: -v for k, v in other._dyads.items()})
        return new

    @overload
    def __rshift__(self, other: "ChunkExpr[None]") -> "RuleExpr[N]":
        ...

    @overload
    def __rshift__[T: VarNode](self, other: "ChunkExpr[T]") -> "RuleExpr[T]":
        ...

    def __rshift__(self, other) -> "RuleExpr":
        if not isinstance(other, ChunkExpr):
            return NotImplemented
        elif self.node is not None and other.node is not None \
            and self.node is not other.node:
            raise ValueError("Multiple variables not allowed in rule expr")
        scope = self.scope | other.scope
        expr = RuleExpr(other.name, self._propagate(other), scope)
        expr._lhs[self] = 1.0
        expr._rhs[other] = 1.0
        return expr

    @overload
    def __rrshift__(self, other: tuple["ChunkExpr[None]", ...]) \
        -> "RuleExpr[N]":
        ...

    @overload
    def __rrshift__[T: VarNode](self, other: tuple["ChunkExpr[T]", ...]) \
        -> "RuleExpr[T]":
        ...

    def __rrshift__(self, other: tuple["ChunkExpr", ...]) -> "RuleExpr":
        if not isinstance(other, tuple):
            return NotImplemented
        rule_expr = RuleExpr(self.name, self.node, self.scope)
        rule_expr._rhs[self] = 1.0
        for chunk_expr in other:
            rule_expr = chunk_expr + rule_expr
        return rule_expr

    def items(self) -> Iterator[tuple[tuple[Bus, Symbol], float]]:
        yield from self._dyads.items()

    def lines(self) -> str:
        lines = []
        for (d, v), w in self._dyads.items():
            lines.append(self._str_dyad_(d, v, w))
        return '\n'.join(lines)        

    @classmethod
    def _str_constituent_(cls, x: Symbol) -> str:
        match x:
            case Term() | Sort():
                key = ~x
                return ".".join([label for label, _ in key[-2:]])
            case Indexical():
                return f"this.{x.name}"
            case Var(node=n, scope=Scope.VAL):
                return f"var({n._name_})"
            case Var(node=n, scope=Scope.DIM):
                return f"var({n._name_}).dim"
            case Constraint(scope=Scope.VAL):
                return f"{x.term._name_}(var({x.node._name_}))"
            case Constraint(scope=Scope.DIM):
                return f"{x.term._name_}(var({x.node._name_}).dim)"
            case _:
                raise ValueError(f"Unexpected constitutent {x}")

    @classmethod
    def _str_dyad_(cls, b: Term, s: Symbol, w: float) -> str:
        s_b = cls._str_constituent_(b)
        s_s = cls._str_constituent_(s)
        sign = "+" if w >= 0 else "-"
        if abs(w) == 1.0:
            return f"{sign} {s_b} ** {s_s}"
        else:           
            return f"{sign}{abs(w)} * {s_b} ** {s_s}"


@final
class RuleExpr[N: VarNode](Expr[N]):
    _lhs: dict[ChunkExpr, float]
    _rhs: dict[ChunkExpr, float]

    def __init__(self, 
        name: str = "", 
        node: N = None, 
        scope: Scope = Scope.NIL
    ) -> None:
        super().__init__(name, node, scope)
        self._lhs = {}
        self._rhs = {}

    def __copy__(self) -> Self:
        new = super().__copy__()
        new._lhs.update(self._lhs)
        new._rhs.update(self._rhs)
        return new

    def __rxor__(self, other: str) -> Self:
        if not isinstance(other, str):
            return NotImplemented
        elif not other.isidentifier():
            raise ValueError(f"'{other}' is not a valid name")
        new = type(self)(other, self.node, self.scope)
        new._lhs.update({f"{other}_L{i}" ^ expr: w 
            for i, (expr, w) in enumerate(self._lhs.items())})
        new._rhs.update({f"{other}_R{i}" ^ expr: w 
            for i, (expr, w) in enumerate(self._rhs.items())})
        return new

    @overload
    def __add__(self, other: "ChunkExpr[None] | RuleExpr[None]") \
        -> "RuleExpr[N]":
        ...

    @overload
    def __add__[T: VarNode](self, other: "ChunkExpr[T] | RuleExpr[T]") \
        -> "RuleExpr[T]":
        ...

    def __add__(self, other) -> "RuleExpr":
        if not isinstance(other, (ChunkExpr, RuleExpr)):
            return NotImplemented
        elif self.node is not None and other.node is not None \
            and self.node is not other.node:
            raise ValueError("Multiple variables not allowed in rule expr")
        scope = self.scope | other.scope
        new = RuleExpr(other.name, self._propagate(other), scope)
        new._rhs.update(self._rhs)
        new._lhs.update(self._lhs)
        match other:
            case ChunkExpr():
                new._rhs[other] = 1.0
            case RuleExpr():
                new._rhs.update(other._rhs)
                new._lhs.update(other._lhs)

        return new

    @overload
    def __radd__(self, other: "ChunkExpr[None] | RuleExpr[None]") \
        -> "RuleExpr[N]":
        ...

    @overload
    def __radd__[T: VarNode](self, other: "ChunkExpr[T] | RuleExpr[T]") \
        -> "RuleExpr[T]":
        ...

    def __radd__(self, other) -> "RuleExpr":
        if not isinstance(other, (ChunkExpr, RuleExpr)):
            return NotImplemented
        elif self.node is not None and other.node is not None \
            and self.node is not other.node:
            raise ValueError("Multiple variables not allowed in rule expr")
        scope = self.scope | other.scope
        new = RuleExpr(self.name, other._propagate(self), scope)
        match other:
            case ChunkExpr():
                new._lhs[other] = 1.0
            case RuleExpr():
                new._rhs.update(other._rhs)
                new._lhs.update(other._lhs)
        new._rhs.update(self._rhs)
        new._lhs.update(self._lhs)
        return new
