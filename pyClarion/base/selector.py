'''
Tools for action selection.

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

>>> class MyPacket(ActivationPacket):
... 
...     def default_activation(self, key) -> float:
...         return 0.0
... 
>>> class MaxSelector(Selector):
...
...     def __init__(self, actionable_chunks : Set[Chunk]) -> None:
...
...         self.actionable_chunks = actionable_chunks
...
...     def __call__(self, input_map : ActivationPacket) -> SelectorPacket:
...          
...         selected = self.get_max(input_map)
...         activations = {
...             chunk : input_map[chunk] for chunk in self.actionable_chunks
...         }
...         return SelectorPacket(activations, selected)
...
...     def get_max(self, input_map : ActivationPacket) -> Chunk:
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
>>> from pyClarion.base.knowledge import Node
>>> ch1, ch2, ch3 = Chunk(1), Chunk(2), Chunk(3)
>>> n1, n2, n3 = Node(), Node(), Node()
>>> selector = MaxSelector(actionable_chunks={ch1, ch2, ch3})
>>> p = MyPacket({ch1 : .2, ch2 :  .3, n1 : 1.})
>>> selector(p) == SelectorPacket({ch1 : .2, ch2 : .3, ch3 : 0.}, chosen={ch2})
True

Activations reported by a ``Selector`` object need not be the identical to 
received input activations. They may, for instance, be normalized in some way, 
as shown in the example below.

>>> class NormalizedMaxSelector(MaxSelector):
...
...     def __call__(self, input_map : ActivationPacket) -> SelectorPacket:
...         
...         output = super().__call__(input_map)
...         total = sum(output.values())
...         for chunk in output:
...             output[chunk] /= total
...         return output
... 
>>> selector = NormalizedMaxSelector(actionable_chunks={ch1, ch2, ch3})
>>> selector(p) == SelectorPacket({ch1 : .4, ch2 : .6, ch3 : 0.}, chosen={ch2})
True
'''

from typing import TypeVar, Generic, Iterable, Dict, Set, Any
import abc
import numpy as np
from pyClarion.base.knowledge import Node, Chunk
from pyClarion.base.packet import ActivationPacket, SelectorPacket


At = TypeVar('At')
Pt = TypeVar('Pt', bound=ActivationPacket)

class Selector(Generic[Pt, At], abc.ABC):
    """Selects actionable chunks based on chunk strengths.
    """

    @abc.abstractmethod
    def __call__(self, input_map: Pt) -> SelectorPacket[At]:
        """Identify chunks that are currently actionable based on their 
        strengths.

        :param input_map: Strengths of input nodes.
        """

        pass


class BoltzmannSelector(Selector[Pt, float]):
    """Select a chunk according to a Boltzmann distribution.
    """

    def __init__(self, chunks : Iterable[Chunk], temperature: float) -> None:
        """Initialize a ``BoltzmannSelector`` instance.

        
        :param chunks: An iterable of (potentially) actionable chunks.
        :param temperature: Temperature of the Boltzmann distribution.
        """

        self.actionable_chunks = chunks
        self.temperature = temperature

    def __call__(self, input_map: Pt) -> SelectorPacket[float]:
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.

        :param input_map: Strengths of input nodes.
        """

        boltzmann_distribution : Dict[Node, float] = (
            self.get_boltzmann_distribution(input_map)
        )
        chunk_list, probabilities = zip(*list(boltzmann_distribution.items()))
        choices = self.choose(chunk_list, probabilities)
        return SelectorPacket(boltzmann_distribution, choices)

    def get_boltzmann_distribution(self, input_map: Pt) -> Dict[Node, float]:
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