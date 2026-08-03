__all__ = ["Term", "Sort", "Family", "Root"]


from typing import Generator, Never
from itertools import count

from ..numdicts.keyspaces import Root as KSRoot, Node, Child


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
