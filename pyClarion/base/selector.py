import typing as T
import abc
import numpy as np
from pyClarion.base.node import Chunk
from pyClarion.base.packet import BaseActivationPacket

class Selector(abc.ABC):
    """Selects actionable chunks based on chunk strengths.
    """

    @abc.abstractmethod
    def __call__(
        self, 
        input_map: BaseActivationPacket, 
        actionable_chunks: T.Set[Chunk]
    ) -> T.Set[Chunk]:
        """Identify chunks that are currently actionable based on their 
        strengths.

        kwargs:
            input_map : A dict mapping nodes (Chunks and/or Features) to 
            input activations.
        """

        pass


class BoltzmannSelector(Selector):
    """Select a chunk according to a Boltzmann distribution.
    """

    def __init__(self, temperature: float) -> None:
        """Initialize a BoltzmannSelector.

        kwargs:
            chunks : A set of (potentially) actionable chunks.
            temperature : Temperature of the Boltzmann distribution.
        """

        super().__init__()
        self.temperature = temperature

    def __call__(
        self, 
        input_map: BaseActivationPacket, 
        actionable_chunks: T.Set[Chunk]
    ) -> T.Set[Chunk]:
        """Identify chunks that are currently actionable based on their 
        strengths according to a Boltzmann distribution.

        Note: If an expected input chunk is missing, it is assumed to have 
        activation 0.

        kwargs:
            chunk2strength : A mapping from chunks to their strengths.
        """

        boltzmann_distribution : T.Dict[Chunk, float] = (
            self.get_boltzmann_distribution(input_map, actionable_chunks)
        )
        chunk_list, probabilities = zip(*list(boltzmann_distribution.items()))
        choices = self.choose(chunk_list, probabilities)
        return choices

    def get_boltzmann_distribution(
        self, 
        input_map: BaseActivationPacket, 
        actionable_chunks: T.Set[Chunk]
    ) -> T.Dict[Chunk, float]:
        """Construct and return a boltzmann distribution.
        """

        terms = dict()
        divisor = 0.
        for chunk in actionable_chunks:
            terms[chunk] = np.exp(input_map[chunk] / self.temperature)
            divisor += terms[chunk]
        chunk_list = list(actionable_chunks)
        probabilities = [terms[chunk] / divisor for chunk in chunk_list]
        return dict(zip(chunk_list, probabilities))

    def choose(self, chunk_list, probabilities):
        '''Choose a chunk given some selection probabilities.'''

        choice = np.random.choice(chunk_list, p=probabilities)
        return {choice}