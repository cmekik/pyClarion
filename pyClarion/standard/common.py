'''Common constructs for the standard implementation of Clarion.'''

from pyClarion.base.knowledge import Node
from pyClarion.base.packet import ActivationPacket as BaseActivationPacket
from pyClarion.base.channel import Channel as BaseChannel
from pyClarion.base.junction import Junction as BaseJunction

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

class TopLevelPacket(ActivationPacket):
    pass

class BottomLevelPacket(ActivationPacket):
    pass

class TopDownPacket(ActivationPacket):
    pass

class BottomUpPacket(ActivationPacket):
    pass


################
### CHANNELS ###
################

class Channel(BaseChannel[ActivationPacket]):
    pass


#################
### JUNCTIONS ###
#################

class UpdateJunction(BaseJunction[ActivationPacket]):
    """Merges input activation packets using the packet ``update`` method."""

    def __call__(self, *input_maps: ActivationPacket) -> ActivationPacket:

        output = ActivationPacket()
        output.update(*input_maps)
        return output