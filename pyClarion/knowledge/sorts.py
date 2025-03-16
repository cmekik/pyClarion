from .base import Sort, Var
from .terms import Atom, Compound, Chunk, Rule


class Atoms(Sort[Atom]):
    """
    A data sort for atomic terms.

    Represents a collection of atomic data terms that are alike in content 
    (e.g., color terms, shape terms, etc.).
    """ 
    _vars_: dict[str, Var]

    def __init__(self, name: str = ""):
        super().__init__(name, Atom)
        self._vars_ = {}

    def __call__(self, name: str) -> Var:
        return self._vars_.setdefault(name, Var(name, self))


class Compounds[C: Compound](Sort[C]):
    """A data sort for compound terms."""
    pass


class Chunks(Compounds[Chunk]):
    """
    A data sort for chunk terms.

    Represents a collection of chunk terms. This sort includes a `nil` term as 
    a necessary member.
    """
    nil: Chunk

    def __init__(self, name: str = ""):
        super().__init__(name, Chunk)


class Rules(Compounds[Rule]):
    """
    A data sort for rule terms.

    Represents a collection of rule terms. This sort includes a `nil` term as a 
    necessary member.
    """
    nil: Rule 

    def __init__(self, name: str = ""):
        super().__init__(name, Rule)