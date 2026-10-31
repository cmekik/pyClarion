import enum

from abc import ABC, abstractmethod
from typing import Self, Literal, Final, overload
from weakref import WeakValueDictionary
from contextvars import ContextVar, Token
from collections import deque

from .base import Family, Sort, Term
from ..numdicts.keyspaces import find_parent


class Scope(enum.IntFlag):
    NIL = 0
    VAL = enum.auto()
    DIM = enum.auto()

type VarScope = Literal[Scope.VAL, Scope.DIM]


class Var[T: Family[Sort[Term]] | Sort[Term]]:
    __match__ = ("node", "scope")

    scope: Final[VarScope]
    node: Final[T]

    @property
    def dim(self) -> "Var":
        return Var(self.node, Scope.DIM)

    def __init__(self, node: T, scope: VarScope = Scope.VAL) -> None:
        self.node = node
        self.scope = scope 

    def validate(self, node: Sort[Term] | Term) -> bool:
        if find_parent(node) is not self.node:
            return False
        return True    

def var[T: Family[Sort[Term]] | Sort[Term]](x: T) -> Var[T]:
    return Var(x)


class Constraint[C: Compound, N: Family[Sort[Term]] | Sort[Term]]:
    __match__ = ("term", "node", "scope")

    term: Final[C]
    node: Final[N]
    scope: Final[VarScope]

    def __init__(self, term: C, node: N, scope: VarScope) -> None:
        self.term = term
        self.node = node
        self.scope: VarScope = scope


class Indexical(enum.Enum):
    """
    A relative reference for a compound term.

    Indexicals are constants that refer to current chunk or rule terms,
    their group-level forms, or their concrete instances from within a
    chunk or rule declaration.

    Indexicals are constants that point to current chunk or rule instances or 
    related abstract chunks from within a chunk or rule declaration.

    For instance, the following notation spawns a chunk containing a single 
    self-referential feature on some bus: `bus ** this.chunk`.
    """
    chunk = enum.auto()
    rule = enum.auto()
    chunk_group = enum.auto()
    rule_group = enum.auto()
    chunk_instance = enum.auto()
    rule_instance = enum.auto()


this = Indexical


type Constant = Sort[Term] | Term | Indexical
type Variable = Var | Constraint
type Symbol = Constant | Variable


type VarNode = Family[Sort[Term]] | Sort[Term] | None
type Valuation = Sort[Term] | Term | None


class Expr[N: VarNode]:

    name: Final[str]
    node: Final[N]
    scope: Final[Scope]

    def __init__(self, 
        name: str = "", 
        node: N = None, 
        scope: Scope = Scope.NIL
    ) -> None:
        self.name = name
        self.node = node
        self.scope = scope

    def __copy__(self) -> Self:
        return self.__class__(self.name, self.node, self.scope)

    def __rxor__(self, other: str) -> Self:
        if not isinstance(other, str):
            return NotImplemented
        elif not other.isidentifier():
            raise ValueError(f"'{other}' is not a valid name")
        return type(self)(other, self.node, self.scope)

    @overload
    def _propagate(self, other: "Expr[None]") -> N:
        ...

    @overload
    def _propagate[T: VarNode](self, other: "Expr[T]") -> T:
        ...

    def _propagate(self, other):
        node = other.node
        if other.node is None:
            node = self.node
        return node


class Form[N: VarNode, V: Valuation]:
    __slots__ = ("node", "val")

    node: Final[N]
    val: Final[V]

    def __init__(self, node: N, val: V) -> None:
        self.node = node
        self.val = val


class Instance[N: VarNode, V: Valuation](Form[N, V]):
    __slots__ = ()


class GenericInstance[N: Family[Sort[Term]], V: Term](Instance[N, V]):
    __slots__ = ()


class TypedInstance[N: Sort[Term], V: Term](Instance[N, V]):
    __slots__ = ()


class GroundInstance(Instance[None, None]):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(None, None)


class AbstractForm[N: VarNode, V: Valuation](Form[N, V], ABC):
    __slots__ = ()

    @abstractmethod
    def instantiate(self, val):
        raise NotImplementedError() 


class TypedForm[N: Sort[Term]](AbstractForm[N, None]):
    __slots__ = ()

    def __init__(self, node: N) -> None:
        super().__init__(node, None)

    def instantiate[T: Term](self, val: T) -> TypedInstance[N, T]:
        if val._parent_ is not self.node:
            raise ValueError(f"Incompatible valuation {val}")
        return TypedInstance(self.node, val)


class TypedGenericForm[N: Family[Sort[Term]], V: Sort[Term]](AbstractForm[N, V]):
    __slots__ = ()

    def instantiate[T: Term](self, val: T) -> GenericInstance[N, T]:
        if val._parent_ is not self.val:
            raise ValueError(f"Incompatible valuation {val}")
        return GenericInstance(self.node, val)


class GenericForm[N: Family[Sort[Term]]](AbstractForm[N, None]):
    __slots__ = ()

    def __init__(self, node: N) -> None:
        super().__init__(node, None)

    def instantiate[S: Sort[Term]](self, val: S) -> TypedGenericForm[N, S]:
        if val._parent_ is not self.node:
            raise ValueError(f"Incompatible valuation {val}")
        return TypedGenericForm(self.node, val)


type ValidForm = GroundInstance \
    | TypedInstance[Sort[Term], Term] \
    | GenericInstance[Family[Sort[Term]], Term] \
    | TypedGenericForm[Family[Sort[Term]], Sort[Term]] \
    | TypedForm[Sort[Term]] \
    | GenericForm[Family[Sort[Term]]]


type DerivedForm = TypedGenericForm[Family[Sort[Term]], Sort[Term]] \
    | TypedInstance[Sort[Term], Term] \
    | GenericInstance[Family[Sort[Term]], Term]


class Primitive(Term):
    """Base class for primitive terms."""
    pass


class Primitives[C: Primitive](Sort[C]):
    """A data sort for atomic terms.""" 
    _vars_: dict[str, Var]

    def __init__(self, name: str = "", prefix: str = "") -> None:
        super().__init__(name, prefix)
        self._vars_ = {}

    @classmethod
    def _get_member_type_(cls) -> type[Primitive]:
        return Primitive


class PrimitiveFamily(Family[Primitives[Primitive]]):

    @classmethod
    def _get_member_type_(cls) -> type[Primitives]:
        return Primitives


class Compound[E: Expr[VarNode]](Term):
    """
    Base class for compound terms.

    A compound term represents a data element with some constituency structure 
    (i.e., chunks and rules).

    Do not directly instantiate this class.
    """
    _template_: Self | None
    _instances_: WeakValueDictionary[Sort[Term] | Term, Self]

    @property
    def _expr_(self) -> E | None:
        return self._private_expr_

    @property
    def _form_(self) -> ValidForm:
        return self._private_form_

    @property
    def _is_abstract_(self) -> bool:
        expr = self._expr_
        if expr is None:
            return False
        match self._form_:
            case GenericForm():
                return bool(expr.scope)
            case TypedForm() | TypedGenericForm():
                return Scope.VAL in expr.scope
            case Instance():
                return False
            case _:
                raise TypeError(f"Unexpected form type {type(self._form_)}")

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, expr: E) -> None:
        ...

    @overload
    def __init__(self, expr: E, form: DerivedForm, template: Self) -> None:
        ...

    def __init__(self, 
        expr: E | None = None, 
        form: ValidForm | None = None, 
        template: Self | None = None
    ) -> None:
        if expr is None:
            if form is not None or template is not None:
                raise ValueError("Invalid compound term spec")
            name = ""
            form = GroundInstance()
        else:
            name = expr.name
            if form is None:
                match expr.node:
                    case None:
                        form = GroundInstance()
                    case Sort():
                        form = TypedForm(expr.node)
                    case Family():
                        form = GenericForm(expr.node)
                    case _:
                        raise TypeError("Unexpected variable node type "
                            f"{type(expr.node)}")
            elif expr.node is not form.node:
                raise ValueError("Inconsistent compound term spec")
            if form.val is not None:
                if template is None:
                    raise ValueError("Valued compound requires a template.")
                name = f"{template._name_}_{form.val._name_}"
            elif template is not None and form.val is None:
                raise ValueError("Unvalued compound cannot have a template.")
        super().__init__(name)
        self._private_expr_ = expr
        self._private_form_ = form
        self._template_ = template
        self._instances_ = WeakValueDictionary()
        self._init_()

    def _init_(self):
        pass

    def __call__(self, var: Var) -> Constraint:
        if self._expr_ is None:
            raise ValueError("Cannot constrain free compound term")
        elif not isinstance(var, Var):
            raise TypeError(f"Unexpected argument type {type(var)}")
        elif var.node is not self._expr_.node:
            raise ValueError("Incompatible var")
        elif isinstance(self._form_, Instance):
            raise ValueError("Cannot constrain concrete term")
        return Constraint(self, var.node, var.scope)

    @abstractmethod
    def instantiate(self, val: Sort[Term] | Term) -> Self:
        raise NotImplementedError()

    @classmethod
    def _factory(cls, expr: E) -> "tuple[Compounds[Self], Self]":
        try:
            sort = COMPOUNDS.get()
        except LookupError as err:
            raise RuntimeError("No active compound sort") from err
        if not issubclass(cls, sort._get_member_type_()):
            clsname = cls.__name__
            raise TypeError(f"No sort accepting {clsname} in current context")
        value = cls(expr)
        return sort, value


COMPOUNDS: ContextVar["Compounds"] = ContextVar("COMPOUNDS")


class Compounds[C: Compound](Sort[C]):
    """A data sort for compound terms."""
    _init_required_ = False
    __tokens_: list[Token]

    def __init__(self, name: str = "", prefix: str = ""):
        super().__init__(name, prefix)
        self.__tokens_ = []

    def __enter__(self):
        self.__tokens_.append(COMPOUNDS.set(self))
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        COMPOUNDS.reset(self.__tokens_.pop())


class CompoundFamily(Family[Compounds[Compound]]):
    _init_required_ = False

    @classmethod
    def _get_member_type_(cls) -> type[Compounds]:
        return Compounds


class DataFamily(Family[Sort]):
    def __init__(self, name: str = ""):
        super().__init__(name, skip=(Compounds,))

    @classmethod
    def _get_member_type_(cls) -> type[Sort]:
        return Sort