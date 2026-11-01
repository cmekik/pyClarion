from typing import Self, Iterator, overload, final, Final
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
                expr = ChunkExpr(dyads={(self, other): 1.0})
            case Var(node=n) | Constraint(node=n):
                expr = ChunkExpr(node=n, dyads={(self, other): 1.0})
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

    dyads: Final[dict[tuple[Bus, Symbol], float]]
        
    def __init__(self, 
        name: str = "", 
        node: N = None, 
        dyads: dict[tuple[Bus, Symbol], float] | None = None
    ) -> None:
        scope = Scope.NIL
        if dyads is None:
            dyads = {}
        else:
            scope = Scope.NIL
            for (_, symbol) in dyads:
                match symbol:
                    case Var(scope=Scope.DIM) \
                        | Constraint(scope=Scope.DIM) \
                        | this.chunk_group \
                        | this.rule_group:
                        scope |= Scope.DIM
                    case Var(scope=Scope.VAL) \
                        | Constraint(scope=Scope.VAL) \
                        | this.chunk_instance \
                        | this.rule_instance:
                        scope |= Scope.VAL
                    case _:
                        pass
        super().__init__(name, node, scope)
        self.dyads = dyads

    def __str__(self) -> str:
        clsname = self.__class__.__name__
        lines = self.lines()
        return f"{clsname}(\n{indent(lines, '    ')})"

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        dyads = {k: -v for k, v in self.dyads.items()}
        return type(self)(self.name, self.node, dyads)

    def __rmul__(self, other: float) -> Self:
        dyads = {k: other * v for k, v in self.dyads.items()}
        return type(self)(self.name, self.node, dyads)

    def __rxor__(self, other: str) -> Self:
        if not isinstance(other, str):
            return NotImplemented
        elif not other.isidentifier():
            raise ValueError(f"'{other}' is not a valid name")
        return type(self)(other, self.node, self.dyads)

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
        dyads = {**self.dyads, **other.dyads}
        return type(self)(other.name, self._propagate(other), dyads)

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
        dyads = {**self.dyads, **{k: -v for k, v in other.dyads.items()}}
        return type(self)(other.name, self._propagate(other), dyads)

    @overload
    def __rshift__(self, other: None) -> "RuleExpr[N]":
        ...

    @overload
    def __rshift__(self, other: "ChunkExpr[None]") -> "RuleExpr[N]":
        ...

    @overload
    def __rshift__[T: VarNode](self, other: "ChunkExpr[T]") -> "RuleExpr[T]":
        ...

    def __rshift__(self, other: "ChunkExpr | None") -> "RuleExpr":
        if other is None:
            return self + RuleExpr()
        elif not isinstance(other, ChunkExpr):
            return NotImplemented
        elif self.node is not None and other.node is not None \
            and self.node is not other.node:
            raise ValueError("Multiple variables not allowed in rule expr")
        return self + RuleExpr() + other


    @overload
    def __rrshift__(self, other: None) -> "RuleExpr[N]":
        ...

    @overload
    def __rrshift__(self, other: tuple["ChunkExpr[None]", ...]) \
        -> "RuleExpr[N]":
        ...

    @overload
    def __rrshift__[T: VarNode](self, other: tuple["ChunkExpr[T]", ...]) \
        -> "RuleExpr[T]":
        ...

    def __rrshift__(self, other: tuple["ChunkExpr", ...] | None) -> "RuleExpr":
        if other is None:
            return RuleExpr() + self
        elif not isinstance(other, tuple):
            return NotImplemented
        rule_expr = RuleExpr() + self
        for chunk_expr in other:
            rule_expr = chunk_expr + rule_expr
        return rule_expr

    def items(self) -> Iterator[tuple[tuple[Bus, Symbol], float]]:
        yield from self.dyads.items()

    def lines(self) -> str:
        lines = []
        for (d, v), w in self.dyads.items():
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
    lhs: Final[dict[ChunkExpr[N], float]]
    rhs: Final[dict[ChunkExpr[N], float]]

    def __init__(self, 
        name: str = "", 
        node: N = None, 
        lhs: dict[ChunkExpr[N], float] | None = None,
        rhs: dict[ChunkExpr[N], float] | None = None
    ) -> None:
        scope = Scope.NIL
        lhs = {} if lhs is None else lhs
        rhs = {} if rhs is None else rhs
        for chunk_expr in lhs:
            scope |= chunk_expr.scope
        for chunk_expr in rhs:
            scope |= chunk_expr.scope
        super().__init__(name, node, scope)
        self.lhs = lhs
        self.rhs = rhs

    def __rxor__(self, other: str) -> Self:
        if not isinstance(other, str):
            return NotImplemented
        elif not other.isidentifier():
            raise ValueError(f"'{other}' is not a valid name")
        len_lhs = len(self.lhs)
        len_rhs = len(self.rhs)
        if 1 < len_lhs:
            lhs = {f"{other}_{i}" ^ c: w 
                for i, (c, w) in enumerate(self.lhs.items())}
        else:
            lhs = {other ^ c: w for c, w in self.lhs.items()}
        if 1 < len_rhs:
            rhs = {f"{other}_{i}" ^ c: w 
                for i, (c, w) in enumerate(self.rhs.items())}
        else:
            rhs = {other ^ c: w for c, w in self.rhs.items()}
        return type(self)(other, self.node, lhs, rhs)

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
        node = self._propagate(other)
        match other:
            case ChunkExpr():
                lhs = self.lhs
                rhs = {**self.rhs, other: 1.0}
            case RuleExpr():
                lhs = {**self.lhs, **other.lhs}
                rhs = {**self.rhs, **other.rhs}
        return type(self)(self.name, node, lhs, rhs)

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
        node = other._propagate(self)
        match other:
            case ChunkExpr():
                lhs = {**self.lhs, other: 1.0}
                rhs = self.rhs
            case RuleExpr():
                lhs = {**self.lhs, **other.lhs}
                rhs = {**self.rhs, **other.rhs}
        return type(self)(self.name, node, lhs, rhs)
