'''
This module provides utilities for action selection.

Usage
=====

This module exports the abstract ``Selector`` class and related constructs.

The ``Selector`` class cannot be directly instantiated.

>>> Selector()
Traceback (most recent call last):
    ....
TypeError: Can't instantiate abstract class Selector with abstract methods __call__

``Selector`` objects provide a ``__call__`` method, which takes an activation 
packet as input and outputs a ``SelectorPacket`` object. ``SelectorPacket`` 
objects summarize the result of an action selection routine: they contain 
information about which actionable chunks are selected for firing and the 
strengths associated with each actionable chunk.

The following example defines a simple ``MaxSelector`` and demonstrates its use.

>>> class MyPacket(BaseActivationPacket):
... 
...     def default_activation(self, key) -> float:
...         return 0.0
... 
>>> class MaxSelector(Selector):
...
...     def __init__(self, actionable_chunks : T.Set[Chunk]) -> None:
...
...         self.actionable_chunks = actionable_chunks
...
...     def __call__(self, input_map : BaseActivationPacket) -> SelectorPacket:
...          
...         selected = self.get_max(input_map)
...         activations = {
...             chunk : input_map[chunk] for chunk in self.actionable_chunks
...         }
...         return SelectorPacket(selected, activations)
...
...     def get_max(self, input_map : BaseActivationPacket) -> Chunk:
...         """
...         Return actionable chunk with maximum activation in current input.
...         
...         In case of multiple maxima, picks the first one encoutered.
...         """
... 
...         for chunk in self.actionable_chunks:
...             try:
...                 if input_map[selected] < input_map[chunk]:
...                     selected = chunk
...             except NameError:
...                 selected = chunk
...         return set([selected])
...
>>> from pyClarion.base.node import Node
>>> ch1, ch2, ch3 = Chunk(1), Chunk(2), Chunk(3)
>>> n1, n2, n3 = Node(), Node(), Node()
>>> selector = MaxSelector(actionable_chunks={ch1, ch2, ch3})
>>> p = MyPacket({ch1 : .2, ch2 :  .3, n1 : 1.})
>>> selector(p) == SelectorPacket(
...     choices={ch2}, activations={ch1 : .2, ch2 : .3, ch3 : 0.}
... )
...
True

Activations reported by a ``Selector`` object need not be the identical to 
received input activations. They may, for instance, be normalized in some way, 
as shown in the example below.

>>> class NormalizedMaxSelector(MaxSelector):
...
...     def __call__(self, input_map : BaseActivationPacket) -> SelectorPacket:
...         
...         output = super().__call__(input_map)
...         total = sum(output.activations.values())
...         for chunk in output.activations:
...             output.activations[chunk] /= total
...         return output
... 
>>> selector = NormalizedMaxSelector(actionable_chunks={ch1, ch2, ch3})
>>> selector(p) == SelectorPacket(
...     choices={ch2}, activations=MyPacket({ch1 : .4, ch2 : .6, ch3 : 0.})
... )
...
True
'''

import typing as T
import abc
import numpy as np
from pyClarion.base.node import Node, Chunk
from pyClarion.base.packet import BaseActivationPacket


At = T.TypeVar('At', bound=BaseActivationPacket)


class SelectorPacket(object):
    '''
    Represents the output of an action selection routine.

    Contains information about the selected actions and strengths of actionable 
    chunks. 
    '''
    
    def __init__(
        self, choices : T.Set[Chunk], activations : T.Dict[Node, T.Any]
    ) -> None:
        '''
        Initialize a ``SelectorPacket`` instance.

        :param choices: The set of actions to be fired.
        :param activations: Activation strengths of actionable chunks.
        '''

        self.choices = choices
        self.activations = activations

    def __eq__(self, other : T.Any) -> bool:

        if (
            isinstance(other, SelectorPacket) and
            self.choices == other.choices and
            self.activations == other.activations
        ):
            return True
        else:
            return False


class Selector(abc.ABC):
    """Selects actionable chunks based on chunk strengths.
    """

    @abc.abstractmethod
    def __call__(
        self, 
        input_map: BaseActivationPacket
    ) -> SelectorPacket:
        """Identify chunks that are currently actionable based on their 
        strengths.

        :param input_map: Strengths of input nodes.
        """

        pass


class BoltzmannSelector(Selector):
    """Select a chunk according to a Boltzmann distribution.
    """

    def __init__(self, chunks : T.Iterable[Chunk], temperature: float) -> None:
        """Initialize a ``BoltzmannSelector`` instance.

        
        :param chunks: An iterable of (potentially) actionable chunks.
        :param temperature: Temperature of the Boltzmann distribution.
        """

        self.actionable_chunks = chunks
        self.temperature = temperature

    def __call__(
        self, 
        input_map: BaseActivationPacket
    ) -> SelectorPacket:
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.

        :param input_map: Strengths of input nodes.
        """

        boltzmann_distribution : T.Dict[Node, float] = (
            self.get_boltzmann_distribution(input_map)
        )
        chunk_list, probabilities = zip(*list(boltzmann_distribution.items()))
        choices = self.choose(chunk_list, probabilities)
        return SelectorPacket(choices, boltzmann_distribution)

    def get_boltzmann_distribution(
        self, 
        input_map: BaseActivationPacket
    ) -> T.Dict[Node, float]:
        """Construct and return a boltzmann distribution.
        """

        terms = dict()
        divisor = 0.
        for chunk in self.actionable_chunks:
            terms[chunk] = np.exp(input_map[chunk] / self.temperature)
            divisor += terms[chunk]
        probabilities = [
            terms[chunk] / divisor for chunk in self.actionable_chunks
        ]
        return dict(zip(self.actionable_chunks, probabilities))

    def choose(self, chunks, probabilities):
        '''Choose a chunk given some selection probabilities.'''

        choice = np.random.choice(chunk_list, p=probabilities)
        return {choice}


if __name__ == '__main__':
    import doctest
    doctest.testmod()