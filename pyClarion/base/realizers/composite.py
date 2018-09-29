from abc import abstractmethod
from typing import Dict, Any, Iterator, Iterable, Type, Optional, Hashable
from pyClarion.base.symbols import Node, Flow, Appraisal, Subsystem
from pyClarion.base.invariants import (
    subsystem_may_connect, subsystem_may_contain
)
from pyClarion.base.packets import ActivationPacket
from pyClarion.base.realizers.abstract import Realizer
from pyClarion.base.links import (
    SubsystemInputMonitor, SubsystemOutputView, PullMethod
)


class SubsystemRealizer(Realizer[Subsystem]):
    """A network of interconnected nodes and flows."""

    def __init__(self, construct: Subsystem) -> None:

        super().__init__(construct)

        self.dict: Dict = dict()
        self._appraisal: Optional[Appraisal] = None
        self.input = SubsystemInputMonitor(self._watch, self._drop)
        self.output = SubsystemOutputView(self._view)        

    def __len__(self) -> int:

        return len(self.dict)

    def __contains__(self, obj: Any) -> bool:

        return obj in self.dict

    def __iter__(self) -> Iterator:

        return self.dict.__iter__()

    def __getitem__(self, key: Any) -> Any:

        return self.dict[key]

    def __setitem__(self, key: Any, value: Any) -> None:

        if not key == value.construct:
            raise ValueError("Mismatch between key and realizer construct.")

        if subsystem_may_contain(key):

            if isinstance(key, Appraisal):
                if self._appraisal:
                    raise Exception("Appraisal already set")
                else:
                    self._appraisal = key
            
            self.dict[key] = value

            if isinstance(key, Node):
                for buffer, pull_method in self.input.input_links.items():
                    value.input.watch(buffer, pull_method)

            for construct, realizer in self.dict.items():
                if subsystem_may_connect(source=construct, target=key):
                    value.input.watch(construct, realizer.output.view)
                if subsystem_may_connect(source=key, target=construct):
                    realizer.input.watch(key, value.output.view)

        else:
            raise TypeError("Unexpected type {}".format(type(key)))

    def __delitem__(self, key: Any) -> None:

        del self.dict[key]
        for construct, construct_link in self.dict.items():
            if subsystem_may_connect(key, construct):
                construct_link.drop(key)

        if isinstance(key, Appraisal):
            self._appraisal = None

    def _watch(self, identifier: Hashable, pull_method: PullMethod) -> None:

        for node in self.nodes:
            self[node].input.watch(identifier, pull_method)

    def _drop(self, identifier: Hashable) -> None:

        for node in self.nodes:
            self[node].input.drop(identifier)

    def _view(self, keys: Iterable[Node] = None) -> ActivationPacket:

        return self[self.appraisal].view(keys)

    @abstractmethod
    def do(self) -> None:
        pass

    @property
    def nodes(self) -> Iterable[Node]:
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, Node)
        }

    @property
    def flows(self) -> Iterable[Flow]:
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, Flow)
        }

    @property
    def appraisal(self) -> Appraisal:
        
        if self._appraisal:
            return self._appraisal
        else:
            raise AttributeError("Appraisal not set.")
