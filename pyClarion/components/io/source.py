__all__ = ["Source", "Input"]


from typing import overload

from ...events.system import Action, Process
from ...events.sites import State, Site
from ...events.updates import ForwardUpdate
from ...knowledge.base import Root, Term
from ...knowledge.compounds import Var, Indexical, MatchVar, Chunk
from ...knowledge.data import Nodes
from ...numdicts.keys import Key
from ...numdicts.keyspaces import keyform
from ...numdicts.sentinel import UndefinedType


class Source[D: Nodes, C: float | UndefinedType](Process[Root]):
    """
    An activation source.
    
    Receives activations from external sources.
    """
    output: Site[C] = Site()

    @property
    def d(self) -> D:
        return self._d

    def __init__(self, name: str, d: D, *, c: C = 0.0) -> None:
        super().__init__(name)
        self.system.check_root(d)
        i, l = self.system.get_index(keyform(d)), self.system.depth
        self._d = d 
        self.output = State(i, {}, c, l)


class Input[S: Source[Nodes, float | UndefinedType]](Action[S]):
    __slots__ = ()

    @overload
    def __init__(self, source: S, data: dict[Key, float]) -> None: 
        ...

    @overload
    def __init__[T: Term](self, source: S, data: dict[T, float]) -> None: 
        ...

    @overload
    def __init__(self, source: S, data: Chunk) -> None: 
        ...

    def __init__(self, source, data):
        super().__init__(source)
        _data = {}
        match data:
            case dict():
                for k, v in data.items():
                    if not isinstance(v, float):
                        raise TypeError("Encountered non-float value for"
                            f" key {k}")
                    elif isinstance(k, Term):
                        source.system.check_root(k)
                        k = ~k
                    elif not isinstance(k, Key):
                        raise TypeError("Unexpected key type" 
                            f" {type(k).__name__}")
                    _data[k] = v
            case Chunk():
                for (bus, symbol), weight in data._dyads_.items():
                    if isinstance(symbol, (Var, MatchVar)):
                        raise TypeError("Input chunk contains variables.")
                    constant = symbol
                    if isinstance(constant, Indexical):
                        constant = data._evaluate_indexicals_(constant)
                    source.system.check_root(bus, constant)
                    _data[~bus * ~constant] = weight
            case _:
                raise TypeError(f"Unexpected type '{type(data).__name__}'")
        for k in _data:
            if k not in source.output.index:
                raise ValueError(f"Unexpected key {k}")
        self.updates.append(ForwardUpdate(self.source.output, _data, "push"))
