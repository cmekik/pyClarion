'''Common constructs for the standard implementation of Clarion.'''

from pyClarion.base.symbols import Node
from pyClarion.base.packet import ActivationPacket
from pyClarion.base.processor import UpdateJunction as BaseUpdateJunction


###########################
### DEFAULT ACTIVATIONS ###
###########################


def default_factory(key: Node) -> float:
    
    return 0.0


#################
### JUNCTIONS ###
#################

class UpdateJunction(BaseUpdateJunction[float]):
    """Merges input activation packets using the packet ``update`` method."""

    def __call__(
        self, *input_maps: ActivationPacket[float]
    ) -> ActivationPacket[float]:

        output = super().__call__(*input_maps)
        output.default_factory = default_factory
        return output