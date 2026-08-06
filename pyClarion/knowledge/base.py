__all__ = ["Term", "Sort", "Family", "Root"]


import enum

from typing import Generator, Iterator, Never
from itertools import count

from ..numdicts.keyspaces import Root as KSRoot, Node, Child, Unit, find_parent
from ..numdicts.keys import Key, KeyForm


class Term(Child):
    """
    Base class for terms.

    Data terms represent indvidual data elements of a model (e.g., individual 
    features, parameters etc.).
    
    Do not directly instantiate or subclass this class. Use `Atom`, `Chunk`, or 
    `Rule` instead.
    """


class Sort[C: Term](Node[C]):
    """
    A data sort.

    Represents a collection of data terms that are alike in content (e.g., 
    color terms, shape terms, etc.). 

    Direct instantiation or subclassing of this class is not recommended. Use 
    `Atoms`, `Chunks`, or `Rules` instead.
    """
    _h_offset_ = 1
    _prefix_: str
    _counter_: count

    def __init__(self, name: str = "", prefix: str = "") -> None:
        super().__init__(name)
        self._prefix_ = prefix
        self._counter_ = count()
        self._namer_ = self._name_generator_()
    
    def _name_generator_(self) -> Generator[str, None, Never]:
        while True:
            yield f"{self._prefix_}_{next(self._counter_)}"


class Family[S: Sort](Node[S]):
    """
    A family of data sorts.

    Represents a collection of data terms that are alike in content (e.g., 
    color terms, shape terms, etc.). 
    """
    _h_offset_ = 2


class Root(KSRoot[Family]):
    """The root of a hierarchy of data symbols."""
    _h_offset_ = 3

    @classmethod
    def _get_member_type_(cls) -> type[Family]:
        return Family


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

    # @contextmanager
    # def subset(self, keys: Iterable[Term]):
    #     if keys:
    #         _subset = self._subset
    #         self._subset = self._subset & frozenset(keys)
    #         yield self
    #         self._subset = _subset
    #     else:
    #         yield self


class MatchVar[C: Term]:
    term: C
    variables: tuple[Var, ...]
 
    def __init__(self, term: C, *variables: Var) -> None:
        self.term = term
        self.variables = variables


class Indexical(enum.Enum):
    """
    A relative reference for a compound term.

    Indexicals are constants that point to current chunk or rule instances or 
    their templates from within a chunk or rule declaration.

    For instance, the following notation spawns a chunk containing a single 
    self-referential feature on some bus: `bus ** this.chunk`.

    See `Bus` and `Chunk` for further details on symbolic `Chunk` instance 
    construction.
    """
    chunk = enum.auto()
    rule = enum.auto()
    chunk_instance = enum.auto()
    rule_instance = enum.auto()


this = Indexical


type Constant = Term | Indexical
type Variable = Var | MatchVar
type Symbol = Constant | Variable