__all__ = ["Source", "Input"]


from typing import overload, cast, Final

from ...events.system import Action, Event, Process
from ...events.sites import State, Site
from ...events.updates import ForwardUpdate
from ...knowledge.base import Root, Term
from ...knowledge.language import Var, Indexical, Constraint
from ...knowledge.explicit import Chunk, Nodes
from ...numdicts.keys import Key
from ...numdicts.keyspaces import keyform, find_root, Dyad
from ...numdicts.sentinel import UndefinedType


class Source[D: Nodes, C: float | UndefinedType](Process[Root]):
    """
    An activation source.
    
    Receives activations from external sources.
    """

    d: Final[D]

    output: Site[C] = Site()

    def __init__(self, name: str, d: D, *, c: C = 0.0) -> None:
        if isinstance(d, Dyad):
            r = find_root(d.units[0])
        else:
            r = find_root(d)
        super().__init__(name, cast(Root | None, r))
        self.system.check_root(d)
        i, l = self.system.get_index(keyform(d)), self.system.depth
        self.d = d 
        self.output = State(i, {}, c, l)

    def resolve(self, event: Event) -> None:
        match event.action:
            case Input(p) if p is self:
                pass
            case _:
                raise ValueError(f"Unexpected event {event}")


class Input[S: Source[Nodes, float | UndefinedType]](Action[S]):
    __slots__ = ()

    @overload
    def __init__(self, source: S, data: dict[Key, float]) -> None: 
        ...

    @overload
    def __init__[D: Dyad](self, source: S, data: dict[D, float]) -> None: 
        ...

    @overload
    def __init__[T: Term](self, source: S, data: dict[T, float]) -> None: 
        ...

    @overload
    def __init__(self, source: S, data: Chunk) -> None: 
        ...

    def __init__(self, source: S, data: dict | Chunk) -> None:
        super().__init__(source)
        _data = {}
        match data:
            case dict():
                for k, v in data.items():
                    if not isinstance(v, float):
                        raise TypeError(f"Non-float value for key {k}")
                    elif isinstance(k, (Dyad, Term)):
                        source.system.check_root(k)
                        k = ~k
                    elif not isinstance(k, Key):
                        key_type = type(k).__name__
                        raise TypeError(f"Unexpected key type {key_type}")
                    _data[k] = v
            case Chunk():
                for (bus, symbol), weight in data._dyads_.items():
                    if isinstance(symbol, (Var, Constraint)):
                        raise TypeError("Input chunk contains variables.")
                    constant = symbol
                    if isinstance(constant, Indexical):
                        constant = data._interpret_indexical_(constant)
                    if constant is not None:
                        source.system.check_root(bus, constant)
                        _data[~bus * ~constant] = weight
            case _:
                raise TypeError(f"Unexpected type '{type(data).__name__}'")
        for k in _data:
            if k not in source.output.index:
                raise ValueError(f"Unexpected key {k}")
        self.updates.append(ForwardUpdate(self.source.output, _data, "push"))
