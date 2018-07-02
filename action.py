"""This module provides tools for handling actions in the Clarion cognitive 
architecture. 

There are two steps to action handling: action selection and action execution. 
These two steps are handled separately by the ChunkSelector and ActionHandler 
classes, respectively.
"""


import abc
import nodes
import numpy as np


####### ABSTRACTIONS #######

class ChunkSelector(abc.ABC):
    """An abstract class defining the interface for selection of actionable 
    chunks based on chunk strengths.
    """

    def __init__(self, chunks: nodes.ChunkSet) -> None:
        """Initialize a chunk selector.

        kwargs:
            chunks : A set of (potentially) actionable chunks.
        """

        self.chunks = chunks

    @abc.abstractmethod
    def __call__(self, chunk2strength: nodes.Chunk2Float) -> nodes.ChunkSet:
        """Identify chunks that are currently actionable based on their 
        strengths.

        kwargs:
            chunk2strength : A mapping from chunks to their strengths.
        """

        pass

class ActionHandler(object):
    """Generic class for handling chunk-driven action execution.

    Can be used out of the box to link action chunks to callbacks implementing 
    relevant actions.
    """

    def __init__(self, chunk2action : nodes.Chunk2Callable) -> None:
        """Initialize an action handler.

        kwargs:
            chunk2action : A mapping from (action) chunks to actions.
        """

        self.chunk2action = chunk2action
    
    def __call__(self, chunks : nodes.ChunkSet) -> None:
        """Execute selected actions.

        kwargs:
            chunks : A set of chunks representing selected actions.
        """
        
        for chunk in chunks:
            try:
                self.chunk2action[chunk].__call__()
            except KeyError:
                continue


####### STANDARD CHUNK SELECTORS #######

class BoltzmannSelector(ChunkSelector):
    """Select a chunk according to a Boltzmann distribution.
    """

    def __init__(self, chunks: nodes.ChunkSet, temperature: float) -> None:
        """Initialize a BoltzmannSelector.

        kwargs:
            chunks : A set of (potentially) actionable chunks.
            temperature : Temperature of the Boltzmann distribution.
        """

        super().__init__(chunks)
        self.temperature = temperature

    def __call__(self, chunk2strength: nodes.Chunk2Float) -> nodes.ChunkSet:
        """Identify chunks that are currently actionable based on their 
        strengths according to a Boltzmann distribution.

        Note: If an expected input chunk is missing, it is assumed to have 
        activation 0.

        kwargs:
            chunk2strength : A mapping from chunks to their strengths.
        """

        terms = dict()
        divisor = 0.
        for chunk in self.chunks:
            try:
                terms[chunk] = np.exp(
                    chunk2strength[chunk] / self.temperature
                )
            except KeyError:
                # By assumption, chunk2strength[chunk] == 0. and exp(0. / t) 
                # is 1.0.
                terms[chunk] = 1.0
            divisor += terms[chunk]

        chunk_list = list(self.chunks)
        probabilities = [terms[chunk] / divisor for chunk in chunk_list]
        choice = np.random.choice(chunk_list, p=probabilities)

        return {choice}