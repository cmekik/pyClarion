'''Common constructs for the standard implementation of Clarion.'''

from pyClarion.base.knowledge import Node
from pyClarion.base.packet import ActivationPacket as BaseActivationPacket
from pyClarion.base.processor import Channel as BaseChannel
from pyClarion.base.processor import Junction as BaseJunction

##########################
### ACTIVATION PACKETS ###
##########################

class ActivationPacket(BaseActivationPacket[float]):
    
    def default_activation(self, key : Node) -> float:
        '''
        Return base node activation in standard Clarion.
        
        Returns 0.
        '''

        return 0.0


################
### CHANNELS ###
################

class Channel(BaseChannel[ActivationPacket, ActivationPacket]):
    pass


#################
### JUNCTIONS ###
#################

class UpdateJunction(BaseJunction[ActivationPacket, ActivationPacket]):
    """Merges input activation packets using the packet ``update`` method."""

    def __call__(self, *input_maps: ActivationPacket) -> ActivationPacket:

        output = ActivationPacket()
        output.update(*input_maps)
        return output