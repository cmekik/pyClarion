import typing as T
import abc
from ..node import get_nodes
from .packet import ActivationPacket

class Junction(abc.ABC):
    """An abstract class for handling the combination of chunk and/or 
    (micro)feature activations from multiple sources.
    """

    @abc.abstractmethod
    def __call__(self, *input_maps: ActivationPacket) -> ActivationPacket:
        """Return a combined mapping from chunks and/or microfeatures to 
        activations.

        kwargs:
            input_maps : Dicts mapping from chunks and/or microfeatures 
            to input activations.
        """

        pass

class MaxJunction(Junction):
    """An activation junction returning max activations for all input nodes.
    """

    output_type : T.Type[ActivationPacket]

    def __call__(self, *input_maps : ActivationPacket) -> ActivationPacket:
        """Return the maximum activation value for each input node.

        kwargs:
            input_maps : A set of mappings from chunks and/or (micro)features 
            to activations.
        """

        node_set = get_nodes(*input_maps)

        activations = self.output_type()
        for n in node_set:
            for input_map in input_maps:
                if activations[n] < input_map[n]:
                    activations[n] = input_map[n]
        return activations

JunctionType = T.Type[Junction]