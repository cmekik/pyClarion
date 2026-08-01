__all__ = ["Source", "Input"]


from typing import Any, overload

from ...events.system import Action, Process
from ...events.sites import State, Site
from ...events.updates import ForwardUpdate
from ...knowledge.base import Root
from ...knowledge.terms import Term, Indexical, Var, MatchVar, Chunk
from ...knowledge.types import Nodes, DVPairs
from ...numdicts.keys import Key
from ...numdicts.keyspaces import keyform
from ...numdicts.sentinel import UndefinedType


class Source[D: Nodes, C: float | UndefinedType](Process[Root]):
    """
    An activation source.
    
    Receives activations from external sources.
    """
    d: D
    main: Site[C] = Site()

    def __init__(self, name: str, d: D, *, c: C = 0.0) -> None:
        super().__init__(name)
        i, l = self.system.get_index(keyform(d)), self.system.depth
        self.d = d 
        self.main = State(i, {}, c, l)


class Input[S: Source](Action[S]):
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
        d = self._parse_input(data)
        self.updates.append(ForwardUpdate(self.source.main, d, "push"))

    def _parse_input(self, d: dict | Chunk) -> dict[Key, float]:
        data = {}
        match d:
            case dict():
                for k, v in d.items():
                    if not isinstance(v, float):
                        raise TypeError("Encountered non-float value for"
                            f" key {k}")
                    elif isinstance(k, Term):
                        self.source.system.check_root(k)
                        k = ~k
                    elif not isinstance(k, Key):
                        raise TypeError("Unexpected key type" 
                            f" {type(k).__name__}")
                    data[k] = v
            case Chunk():
                for (bus, symbol), weight in d._dyads_.items():
                    if isinstance(symbol, (Var, MatchVar)):
                        raise TypeError("Input chunk contains variables.")
                    constant = symbol
                    if isinstance(constant, Indexical):
                        constant = d._evaluate_indexicals_(constant)
                    self.source.system.check_root(bus, constant)
                    data[~bus * ~constant] = weight
            case _:
                raise TypeError(f"Unexpected type '{type(d).__name__}'")
        for k in data:
            if k not in self.source.main.index:
                raise ValueError(f"Unexpected key {k}")
        return data
