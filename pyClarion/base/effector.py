'''
Tools for controlling action execution.

Usage
=====

The ``Effector`` class is an abstract generic class providing an interface 
between the outcome of an action selection cycle and action callbacks. It has 
one abstract method, ``__call__``, which must be implemented before 
instantiation.

>>> Effector()
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class Effector with abstract methods __call__

``Effector.__call__`` expects as input a ``SelectorPacket`` instance, which it 
may use to execute selected actions, compute reaction times etc. The method 
returns ``None``.

The primary responsibility of an ``Effector`` object is to ensure proper 
execution of selected actions. This includes effecting calls to appropriate 
callbacks, providing correct reaction times, and resuming cognitive processing 
in the event that a suitable action candidate has not yet been found (e.g., 
based on internal confidence levels).

The ``MappingEffector`` provides a very simple concrete effector class. This 
class may be used to directly bind individual chunks to callbacks. 

>>> def callback_1():
...     print('Executed callback_1.') 
... 
>>> def callback_2():
...     print('Executed callback_2')
... 
>>> ch1, ch2 = Chunk(1), Chunk(2)
>>> effector = MappingEffector({ch1: callback_1, ch2: callback_2})
>>> selector_packet = SelectorPacket({ch1 : .7, ch2 : .3}, chosen={ch1})
>>> effector(selector_packet)
Executed callback_1.

Some applications may require parametrized calls to various callback that are 
contingent on chosen chunks. Such sophisticated effectors may be implemented by 
overriding ``Effector.__call__``
'''

import abc
from typing import Generic, TypeVar, Mapping, Callable
from pyClarion.base.knowledge import Chunk, Node
from pyClarion.base.packet import SelectorPacket

St = TypeVar('St', bound=SelectorPacket)

class Effector(Generic[St], abc.ABC):
    '''An abstract class for linking actionable chunks to action callbacks.'''
    
    @abc.abstractmethod
    def __call__(self, selector_packet : St) -> None:
        '''
        Execute actions associated with given actionable chunk.

        :param selector_packet: The output of an action selection cycle.
        '''
        pass


class MappingEffector(Effector[St]):
    '''A simple effector built on a map from actionable chunks to callbacks.'''

    def __init__(self, chunk2callback : Mapping[Chunk, Callable]) -> None:
        '''Initialize a ``MappingEffector`` instance.

        :param chunk2callback: Defines mapping from actionable chunks to 
            callbacks.
        '''

        self.chunk2callback = dict(chunk2callback)

    def __call__(self, selector_packet : St) -> None:
        '''
        Execute callbacks associated with each chosen chunk.

        :param selector_packet: The output of an action selection cycle.
        '''
        
        if selector_packet.chosen:
            for chunk in selector_packet.chosen:
                self.chunk2callback[chunk]()