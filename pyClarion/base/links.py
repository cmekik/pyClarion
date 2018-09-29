from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Hashable, Callable, List, Dict, Iterable, Optional
from pyClarion.base.symbols import Node
from pyClarion.base.packets import ActivationPacket


It = TypeVar('It', bound=ActivationPacket)
Ot = TypeVar('Ot', bound=ActivationPacket)
PullMethod = Callable[[Optional[Iterable[Node]]], ActivationPacket]


class InputMonitor(object):

    @abstractmethod
    def pull(self) -> Iterable[ActivationPacket]:
        pass

    @abstractmethod
    def watch(self, identifier: Hashable, pull_method: PullMethod) -> None:
        pass

    @abstractmethod
    def drop(self, identifier: Hashable):
        pass


class OutputView(object):

    @abstractmethod
    def view(self) -> ActivationPacket:
        pass


class BasicInputMonitor(InputMonitor):

    def __init__(self) -> None:

        self.input_links: Dict[Hashable, PullMethod] = dict()

    def pull(self, keys: Iterable[Node] = None) -> Iterable[ActivationPacket]:
        
        return [
            pull_method(keys) for pull_method in self.input_links.values()
        ] 

    def watch(self, identifier: Hashable, pull_method: PullMethod) -> None:

        self.input_links[identifier] = pull_method

    def drop(self, identifier: Hashable):

        del self.input_links[identifier]


class BasicOutputView(OutputView):

    def update(self, output: ActivationPacket) -> None:
        
        self._output_buffer = output

    def view(self, keys: Iterable[Node] = None) -> ActivationPacket:
        
        if keys:
            out = self.output_buffer.subpacket(keys)
        else:
            out = self.output_buffer.copy()
        return out

    @property
    def output_buffer(self) -> ActivationPacket:

        if self._output_buffer:
            return self._output_buffer
        else:
            raise AttributeError()


class SubsystemInputMonitor(BasicInputMonitor):

    def __init__(
        self, 
        watch: Callable[[Hashable, PullMethod], None], 
        drop: Callable[[Hashable], None]
    ) -> None:

        super().__init__()
        self._watch = watch
        self._drop = drop

    def watch(self, identifier: Hashable, pull_method: PullMethod) -> None:

        super().watch(identifier, pull_method)
        self._watch(identifier, pull_method)

    def drop(self, identifier: Hashable):

        super().drop(identifier)
        self._drop(identifier)


class SubsystemOutputView(OutputView):

    def __init__(self, view: PullMethod) -> None:

        self._view = view

    def view(self, keys: Iterable[Node] = None) -> ActivationPacket:
        
        return self._view(keys)
