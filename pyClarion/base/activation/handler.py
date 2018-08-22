import abc
import typing as T
from pyClarion.base.activation.packet import ActivationPacket
from pyClarion.base.activation.junction import Junction

class ActivationHandler(abc.ABC):

    def __init__(self, client : T.Hashable, junction : Junction) -> None:

        self._client = client
        self._junction = junction
        self._buffer : T.Dict[T.Hashable, ActivationPacket] = dict()
        self._listeners : T.Set[ActivationHandler] = set()

    @abc.abstractmethod
    def __call__(self) -> None:
        '''Update listeners with new activation of ``self``.
        '''
        pass

    def register(self, handler : ActivationHandler) -> None:
        '''Register ``handler`` as a listener of ``self``.

        :param handler: An activation handler that listens to self.
        '''
        self.listeners.add(handler)

    def update(
        self, construct : T.Hashable, packet : ActivationPacket
    ) -> None:
        '''Update buffer with a new activation packet.
        '''

        self.buffer[construct] = packet

    @property
    def client(self) -> T.Hashable:
        '''The client construct whose activations are handled by ``self``.
        '''
        return self._client

    @property
    def buffer(self) -> T.Dict[T.Hashable, ActivationPacket]:
        return self._buffer

    @property
    def junction(self) -> Junction:
        return self._junction

    @property
    def listeners(self) -> T.Set[ActivationHandler]:
        return self._listeners

    @staticmethod
    def notify_listeners(
        listeners : T.Iterable[ActivationHandler], 
        construct : T.Hashable, 
        packet : ActivationPacket
    ) -> None:

        for listener in listeners:
            listener.update(construct, packet) 


class NodeHandler(ActivationHandler):
    pass


class ChannelHandler(ActivationHandler):
    pass

