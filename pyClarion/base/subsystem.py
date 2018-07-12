import abc
import typing as T
import enum
from . import node
from . import activation
from . import action


class Subsystem(abc.ABC):

    junction_type : T.Type[activation.Junction]
    selector_type : T.Type[action.Selector]
    filter_type : T.Type[node.Node2ValueFilter] = node.Node2ValueFilter
    action_handler_type : T.Type[action.Handler] = action.Handler

    def __init__(
        self,  
        channels : T.Set[activation.Channel],  
        action_map : node.Chunk2Callable
    ) -> None:

        self.channels = channels
        self.junction =  self.junction_type()
        self.selector = self.selector_type(set(action_map.keys()))
        self.filter = self.filter_type()
        self.action_handler = self.action_handler_type(action_map)

    @abc.abstractmethod
    def __call__(
        self, 
        input_map : node.Node2Float
    ) -> node.Node2Float:
        pass

SubsystemSet = T.Set[Subsystem]