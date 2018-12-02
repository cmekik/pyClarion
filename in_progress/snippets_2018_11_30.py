from pyClarion.base.symbols import Microfeature
from pyClarion.base.packets import ActivationPacket, DefaultActivation
from typing import List, Iterable, Dict, Hashable
import numpy as np


class MicrofeatureToVector(object):
    """Converts Microfeature activations to activation vectors and vice versa."""
    
    def __init__(
        self, 
        inputs: List[Microfeature], 
        outputs: List[Microfeature],
        default_activation: DefaultActivation
    ) -> None:
        """
        Initializer a microfeature to vector converter.

        :param inputs: Expected input microfeatures.
        :param outputs: Expected output microfeatures.
        :param default_activation: The default activation.
        """

        self.inputs = inputs
        self.outputs = outputs
        self.default_activation = default_activation

    def embed_microfeatures(self, packet: ActivationPacket):
        """
        Embed target microfeatures in packet into an input vector.

        :param packet: Input activation packet.
        """

        output = np.zeros((len(self.inputs),))
        for idx, mf in enumerate(self.inputs):
            output[idx] = packet.get(mf, self.default_activation(mf))
        return output

    def embed_vector(self, vector: np.array):
        """
        Construct output packet from given activation vector.

        :param vector: An activation vector.
        """ 

        assert len(vector.shape) == 1
        assert len(self.outputs) == vector.shape[0]
        
        output: ActivationPacket = ActivationPacket()
        for idx, val in enumerate(vector):
            output[self.outputs[idx]] = val
        return output


def microfeatures_from_dims(dims: Dict[Hashable, List[Hashable]]):
    """
    Construct microfeatures from a dimension list.

    :param dims: Dimension value sepc.
    """

    mfs = set()
    for dim in dims:
        for val in dims[dim]:
            mfs.add(Microfeature(dim=dim, val=val))
    return mfs


dims: Dict[Hashable, List[Hashable]] = {
    "shape": ["pyramid", "cube", "cone", "sphere", "cylinder"],
    "size": ["small", "medium", "large"],
    "color": ["red", "yellow", "green", "blue"]
}
mfs = microfeatures_from_dims(dims)
for mf in mfs:
    print(mf)
