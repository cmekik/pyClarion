'''
Tools for building Clarion agents.

.. warning:: 
   This module is very experimental. Definitions may change substantially.
'''

import abc
import typing as T
from pyClarion.base.node import Node
from pyClarion.base.packet import ActivationPacket
from pyClarion.base.channel import Channel
from pyClarion.base.selector import Selector
from pyClarion.base.effector import Effector
from pyClarion.base.network import Network


class Statistic(abc.ABC):
    """Tracks a statistic.

    Contains and updates some relevant statistic(s). Does not store additional 
    information about related constructs, such as references to the objects of 
    stored statistics.
    """
    pass


class Component(abc.ABC):
    """Manages some class of activation flows associated with a subsystem.

    Components are abstractions meant to capture learning and forgetting 
    routines. They monitor the activity of the subsystem to which they belong 
    and modify its members (channels and/or parameters).
    """
    
    @abc.abstractmethod
    def spawn_channels(self) -> T.Union[Channel, T.Iterable[Channel]]:
        '''Create and return channel(s) managed by ``self``.'''

        pass


class Subsystem(abc.ABC):
    '''A Clarion subsystem.'''

    @property
    @abc.abstractmethod
    def network(self) -> Network:
        '''A set of channels representing knowledge stored in ``self``.'''
        
        pass

    @property
    @abc.abstractmethod
    def components(self) -> T.Set[Component]:
        '''Components handling learning and forgetting in ``self``.'''

        pass


class Agent(object):
    """Represents a Clarion agent.

    ``Agent`` objects facilitate the interface between subsystems and the 
    environment. The main responsibility of these objects is to distribute 
    sensory input to subsystems. They also serve to bind together all 
    subsystems associated with a given subject.

    It may also be useful to define action callbacks affecting the environment 
    as methods of this class. Action-centered subsystems would be passed sets 
    of these methods as the callbacks to execute following an action decision. 
    Internal actions may be defined within the relevant subsystem class 
    definition and passed to relevant subsystems in the same way.
    """

    def __call__(self) -> None:
        """Receive and process a new set of sensory/world information.
        """
        pass

    @property
    def subsystems(self) -> T.Set[Subsystem]:
        pass