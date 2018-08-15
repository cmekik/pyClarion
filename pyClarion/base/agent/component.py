import abc.ABC
import typing as T

class Component(abc.ABC):
    """An abstraction for managing some class of activation channels associated 
    with a subsystem.

    Components are abstractions meant to capture learning and forgetting 
    routines. They monitor the activity of the subsystem to which they belong 
    and modify its members (channels and/or parameters).
    """
    pass

ComponentSet = T.Set[Component]