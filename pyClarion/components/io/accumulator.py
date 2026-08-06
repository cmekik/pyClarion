
from typing import cast

from ...knowledge.base import Root
from ...knowledge.primitives import Nodes
from ...events.sites import Site, State
from ...events.system import Process, Action, Event
from ...events.updates import ForwardUpdate
from ...numdicts.keyspaces import find_root, Dyad, keyform


class Accumulator[D: Nodes](Process[Root]):
    input: Site = Site(lax=True)
    output: Site = Site()

    @property 
    def d(self) -> D:
        return self._d

    def __init__(self, name: str, d: D) -> None:
        if isinstance(d, Dyad):
            r = find_root(d.units[0])
        else:
            r = find_root(d)
        super().__init__(name, cast(Root | None, r))
        l = self.system.depth
        self.system.check_root(d)
        self._d = d
        index = self.system.get_index(keyform(d))
        self.input = State(index, {}, 0.0, l)
        self.output = State(index, {}, 0.0, l)

    def resolve(self, event: Event) -> None:
        match event.action:
            case AccumulatorUpdate(p) | AccumulatorReset(p) if p is self:
                pass
            case _:
                raise ValueError(f"Unexpected event {event}")


class AccumulatorUpdate[P: Accumulator](Action[P]):
    def __init__(self, source: P) -> None:
        super().__init__(source)
        output = source.output[0].sum(source.input[0])
        self.updates = [ForwardUpdate(source.output, output)]


class AccumulatorReset[P: Accumulator](Action[P]):
    def __init__(self, source: P) -> None:
        super().__init__(source)
        self.updates = [ForwardUpdate(source.output, {})]
