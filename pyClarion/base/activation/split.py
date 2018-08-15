import typing as T
import abc
from ..node import Microfeature, Chunk
from .packet import ActivationPacket

class Split(abc.ABC):
    """An abstract class for handling the splitting of chunk and/or 
    (micro)feature activations into multiple streams.
    """

    @abc.abstractmethod
    def __call__(
        self, input_map: ActivationPacket
    ) -> T.Iterable[ActivationPacket]:
        """Split an activation dict into multiple streams.

        kwargs:
            input_map : A dict mapping nodes (Chunks and/or Features) input 
            activations.
        """

        pass

class NodeTypeSplit(Split):

    def __call__(
        self, input_map : ActivationPacket
    ) -> T.Iterable[ActivationPacket]:

        microfeature_activations = input_map.__class__()
        chunk_activations = input_map.__class__()

        for node_, strength in input_map.items():
            if isinstance(node_, Microfeature):
                microfeature_activations[node_] = strength 
            elif isinstance(node_, Chunk):
                chunk_activations[node_] = strength

        return microfeature_activations, chunk_activations