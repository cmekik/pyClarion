import abc
import typing as T
import enum
from . import node
from . import activation
from . import action


####### CLASS DEFINITIONS #######

class Subsystem(abc.ABC):

    junction_type : activation.JunctionType
    selector_type : action.SelectorType
    action_handler_type : action.HandlerType = action.Handler

    def __init__(
        self,
        channels : activation.ChannelType,  
        action_map : node.Chunk2Callable
    ) -> None:

        self.channels = channels
        self.junction =  self.junction_type()
        self.selector = self.selector_type(set(action_map.keys()))
        self.action_handler = self.action_handler_type(action_map)

    @abc.abstractmethod
    def __call__(
        self, 
        input_map : node.Node2Float
    ) -> node.Node2Float:
        pass


####### TYPE ALIASES #######

SubsystemSet = T.Set[Subsystem]