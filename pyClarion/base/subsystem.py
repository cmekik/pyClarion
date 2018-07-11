import abc
import typing as T
import enum
from . import nodes
from . import activation
from . import action

class SubsystemMode(enum.Flag):
    pass

class Subsystem(abc.ABC):

    @abc.abstractproperty
    junction_type : type

    @abc.abstractproperty
    selector_type : type

    @abc.abstractproperty
    action_handler_type : type

    def __init__(
        self, 
        nodes : nodes.NodeSet, 
        channels : T.Set[activation.ActivationChannel],  
        action_map : T.Mapping[nodes.Chunk, T.Callable]
    ) -> None:

        self.nodes = nodes
        self.channels = channels
        self.junction =  self.junction_type()
        self.selector = self.selector_type(set(action_map.keys()), .1)
        self.action_handler = self.action_handler_type(action_map)

    # Be more specific about output type.
    @abc.abstractmethod
    def __call__(
        self, 
        input_map : nodes.Node2Float, 
        mode : SubsystemMode
    ) -> T.Any:
        pass