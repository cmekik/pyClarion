from typing import Type, Iterable, Iterator, ClassVar, get_type_hints
from contextlib import contextmanager
from itertools import count

from ..numdicts import ValidationError, Key, KeyForm, ks_parent
from ..numdicts.keyspaces import KSProtocol, KSPath, KSRoot, KSNode, KSChild


class Symbol:
    """
    Base class for data symbols.
    
    Do not directly instantiate or subclass this class.
    """
    pass


class Term(KSChild, Symbol):
    """
    Base class for terms.

    Data terms represent indvidual data elements of a model (e.g., individual 
    features, parameters etc.).
    
    Do not directly instantiate or subclass this class. Use `Atom`, `Chunk`, or 
    `Rule` instead.
    """
    _h_offset_ = 0


class Sort[C: Term](KSNode[C], Symbol):
    """
    A data sort.

    Represents a collection of data terms that are alike in content (e.g., 
    color terms, shape terms, etc.). 

    Direct instantiation or subclassing of this class is not recommended. Use 
    `Atoms`, `Chunks`, or `Rules` instead.
    """
    _h_offset_ = 1
    _mtype_: Type[C]
    _required_: frozenset[Key]
    _counter_: count

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        cls = type(self)
        for name, typ in get_type_hints(cls).items():
            if isinstance(typ, type) and issubclass(typ, self._mtype_):
                self[name] = typ()
                setattr(self, name, self[name])
        self._required_ = frozenset(self._members_)
        self._counter_ = count()

    def __delattr__(self, name: str) -> None:
        if Key(name) in self._required_:
            raise ValidationError(f"Cannot remove required key '{name}'")


class Var[S: Sort](KSProtocol, Symbol):
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

    __mul__ = KSPath.__mul__

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
        if ks_parent(term) != self.sort:
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


class Family(KSNode[Sort], Symbol):
    """
    A family of data sorts.

    Represents a collection of data terms that are alike in content (e.g., 
    color terms, shape terms, etc.). 
    """

    _m_type_: ClassVar[Type[Sort] | tuple[Type[Sort], ...]]
    _h_offset_ = 2
    _required_: frozenset[Key]

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        cls = type(self)
        for name, typ in get_type_hints(cls).items():
            if isinstance(typ, type) and issubclass(typ, self._m_type_):
                self[name] = typ()
                setattr(self, name, self[name])
        self._required_ = frozenset(self._members_)


class Root(KSRoot[Family]):
    """The root of a hierarchy of data symbols."""
    pass
