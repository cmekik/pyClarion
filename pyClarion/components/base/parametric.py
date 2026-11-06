__all__ = ["Parametric", "ParamUpdate"]


from typing import cast, Final

from ...events.system import Action, Event, Process
from ...events.sites import Site, State
from ...events.updates import ForwardUpdate
from ...knowledge.base import Root
from ...knowledge.implicit import Params
from ...numdicts.keyspaces import keyform, find_root
from ...numdicts.sentinel import Sentinel, UndefinedType


class Parametric[P: Params](Process[Root]):
    """
    A parametric process.
    
    A process is parametric iff it has defined metaparameters. This mixin class
    provides a uniform interface for metaparameter initialziation and updating.
    """

    p: Final[P]

    params: Site[UndefinedType] = Site()

    def __init__(self, name: str, p: P) -> None:
        super().__init__(name, cast(Root | None, find_root(p)))
        self.system.check_root(p)
        i, d = self.system.get_index(keyform(p)), p._init_values_()
        self.p = p
        self.params = State(i, d, Sentinel.UNDEFINED, self.system.depth)

    def resolve(self, event: Event) -> None:
        match event.action:
            case ParamUpdate(p) if p is self:
                pass
            case _:
                raise ValueError(f"Unexpected event {event}")


class ParamUpdate[P: Parametric](Action[P]):
    """Updates metaparameters associated with a parametric process."""
    __slots__ = ()
    def __init__(self, source: P, **kwargs: float) -> None:
        data = {~source.p[param]: value for param, value in kwargs.items()}
        super().__init__(source)
        self.updates.append(ForwardUpdate(source.params, data, "write"))
