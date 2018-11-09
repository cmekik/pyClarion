"""Tools for linking basic construct realizer inputs and outputs."""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Hashable, Callable, List, Dict, Iterable, Optional
from pyClarion.base.symbols import Node
from pyClarion.base.packets import ActivationPacket, DefaultActivation


It = TypeVar('It', bound=ActivationPacket)
Ot = TypeVar('Ot', bound=ActivationPacket)
PullMethod = Callable[[Optional[Iterable[Node]]], ActivationPacket]


class InputMonitor(ABC):
    """Listens to outputs of construct realizers of interest"""

    @abstractmethod
    def pull(self) -> Iterable[ActivationPacket]:
        """Return activation packets output by constructs of interest."""

        pass

    @abstractmethod
    def watch(self, identifier: Hashable, pull_method: PullMethod) -> None:
        """Start listening to a new construct of interest."""

        pass

    @abstractmethod
    def drop(self, identifier: Hashable):
        """Stop listening to some construct of interest"""

        pass


class OutputView(ABC):
    """Exposes the output state of a client construct"""

    @abstractmethod
    def view(self) -> ActivationPacket:
        """Return current output of client construct."""

        pass


class BasicInputMonitor(InputMonitor):
    """Listens for inputs to `BasicConstructRealizer` objects."""

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
    """Exposes outputs of `BasicConstructRealizer` objects."""

    def __init__(
        self, default_activation: DefaultActivation = None
    ) -> None:

        self.default_activation = default_activation

    def update(self, output: ActivationPacket) -> None:
        
        self._output_buffer = output

    def view(self, keys: Iterable[Node] = None) -> ActivationPacket:
        
        if keys:
            out = self.output_buffer.subpacket(keys, self.default_activation)
        else:
            out = self.output_buffer.copy()
        return out

    @property
    def output_buffer(self) -> ActivationPacket:

        if self._output_buffer is not None:
            return self._output_buffer
        else:
            raise AttributeError()


class SubsystemInputMonitor(BasicInputMonitor):
    """Listens for inputs to `SubsystemRealizer` objects."""

    def __init__(
        self, 
        watch: Callable[[Hashable, PullMethod], None], 
        drop: Callable[[Hashable], None],
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
    """Exposes outputs of `SubsystemRealizer` objects."""

    def __init__(self, view: PullMethod) -> None:

        self._view = view

    def view(self, keys: Iterable[Node] = None) -> ActivationPacket:
        
        return self._view(keys)
