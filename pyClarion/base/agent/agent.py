import abc
from ..activation.packet import ActivationPacket

class Agent(object):
    """An abstraction for representing Clarion agents.

    Subject objects facilitate the interface between subsystems and the 
    environment. The main responsibility of these objects is to distribute 
    sensory input to subsystems. They also serve to bind together all 
    subsystems associated with a given subject.

    It may also be useful to define action callbacks affecting the environment 
    as methods of this class. Action-centered subsystems would be passed sets 
    of these methods as the callbacks to execute following an action decision. 
    Internal actions may be defined within the relevant subsystem class 
    definition and passed to relevant subsystems in the same way.
    """

    @abc.abstractmethod
    def __call__(self, input_map : ActivationPacket) -> None:
        """Receive and process a new set of sensory/world information.
        """
        pass