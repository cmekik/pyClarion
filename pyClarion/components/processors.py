from typing import Set, Dict, Mapping, Callable
from pyClarion.base import *
import numpy as np


class UpdateJunction(Junction[At]):
    """Merges input activation packets using the packet ``update`` method."""

    def __call__(
        self, *input_maps: ActivationPacket[At]
    ) -> ActivationPacket[At]:

        output: ActivationPacket = ActivationPacket()
        for input_map in input_maps:
            output.update(input_map)
        return output


class MaxJunction(Junction[At]):
    """An activation junction returning max activations for all input nodes.
    """

    def __call__(
        self, *input_maps: ActivationPacket[At]
    ) -> ActivationPacket[At]:
        """Return the maximum activation value for each input node.

        kwargs:
            input_maps : A set of mappings from chunks and/or (micro)features 
            to activations.
        """

        node_set: Set[Node] = set()
        node_set.update(*input_maps)
        output : ActivationPacket = ActivationPacket()
        for n in node_set:
            for input_map in input_maps:
                try :
                    found_new_max = output[n] < input_map[n]
                except KeyError:
                    found_new_max = (n in input_map) and (not n in output)
                if found_new_max:
                    output[n] = input_map[n]
        return output


class BoltzmannSelector(Selector[float]):
    """Select a chunk according to a Boltzmann distribution.
    """

    def __init__(self, temperature: float) -> None:
        """Initialize a ``BoltzmannSelector`` instance.

        
        :param chunks: An iterable of (potentially) actionable chunks.
        :param temperature: Temperature of the Boltzmann distribution.
        """

        self.temperature = temperature

    def __call__(
        self, input_map: ActivationPacket[float]
    ) -> DecisionPacket[float]:
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.

        :param input_map: Strengths of input nodes.
        """

        choices: Set[Chunk] = set()
        boltzmann_distribution : Dict[Node, float] = (
            self.get_boltzmann_distribution(input_map)
        )
        if boltzmann_distribution:
            chunk_list, probabilities = zip(*list(boltzmann_distribution.items()))
            choices = self.choose(chunk_list, probabilities)
        return DecisionPacket(boltzmann_distribution, chosen=choices)

    def get_boltzmann_distribution(
        self, input_map: ActivationPacket[float]
    ) -> Dict[Node, float]:
        """Construct and return a boltzmann distribution.
        """

        terms = dict()
        divisor = 0.
        chunks = [node for node in input_map if isinstance(node, Chunk)]
        for chunk in chunks:
            terms[chunk] = np.exp(input_map[chunk] / self.temperature)
            divisor += terms[chunk]
        probabilities = [
            terms[chunk] / divisor for chunk in chunks
        ]
        return dict(zip(chunks, probabilities))

    def choose(self, chunks, probabilities):
        '''Choose a chunk given some selection probabilities.'''

        choice = np.random.choice(chunks, p=probabilities)
        return {choice}


class MappingEffector(Effector[At]):
    '''A simple effector built on a map from actionable chunks to callbacks.'''

    def __init__(self, chunk2callback : Mapping[Chunk, Callable]) -> None:
        '''Initialize a ``MappingEffector`` instance.

        :param chunk2callback: Defines mapping from actionable chunks to 
            callbacks.
        '''

        self.chunk2callback = dict(chunk2callback)

    def __call__(self, selector_packet : DecisionPacket[At]) -> None:
        '''
        Execute callbacks associated with each chosen chunk.

        :param selector_packet: The output of an action selection cycle.
        '''
        
        if selector_packet.chosen:
            for chunk in selector_packet.chosen:
                self.chunk2callback[chunk]()


class ConstantSource(Source[At]):
    """Simple source outputting a constant activation packet."""

    def __init__(self, packet: ActivationPacket[At] = None) -> None:

        if packet:
            self.packet = packet
        else:
            self.packet = ActivationPacket()

    def __call__(self) -> ActivationPacket[At]:

        return self.packet.copy()

    def update(self, packet: ActivationPacket[At]) -> None:

        self.packet.update(packet)

    def clear(self) -> None:

        self.packet.clear()
