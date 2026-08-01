__all__ = ["Parametric", "ParamUpdate"]


from ...events.system import Action, Process
from ...events.sites import Site, State
from ...events.updates import ForwardUpdate
from ...knowledge.base import Root
from ...knowledge.sorts import Params
from ...numdicts.keyspaces import keyform
from ...numdicts.sentinel import Sentinel, UndefinedType


class Parametric[P: Params](Process[Root]):
    """
    A parametric process.
    
    A process is parametric iff it has defined metaparameters. This mixin class
    provides a uniform interface for metaparameter initialziation and updating.
    """
    p: P
    params: Site[UndefinedType] = Site()
    def _init_parametric(self, p: P) -> None:
        self.system.check_root(p)
        i, d = self.system.get_index(keyform(p)), p._init_values_()
        self.p = p
        self.params = State(i, d, Sentinel.UNDEFINED, self.system.depth)


class ParamUpdate[P: Parametric](Action[P]):
    """Updates metaparameters associated with a parametric process."""
    __slots__ = ()
    def __init__(self, source: P, **kwargs: float) -> None:
        super().__init__(source)
        data = {~source.p[param]: value for param, value in kwargs.items()}
        self.updates.append(ForwardUpdate(source.params, data, "write"))
