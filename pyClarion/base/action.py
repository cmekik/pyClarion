"""This module provides tools for handling actions in the Clarion cognitive 
architecture. 

There are two steps to action handling: action selection and action execution. 
These two steps are respectively handled by the Selector and Handler classes.

For basic details on action selection, see Chapter 3.1.2.3 of Sun (2016). See
also Section 3.4.2 on memory retrieval actions, Section 4.2.3 for possible 
goal-setting actions and Section 4.3.2 for other possible metacognitive actions.

References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
"""


import abc
import typing as T
from . import node


class Selector(abc.ABC):
    """An abstract class defining the interface for selection of actionable 
    chunks based on chunk strengths.
    """

    def __init__(self, chunks: node.ChunkSet) -> None:
        """Initialize a chunk selector.

        kwargs:
            chunks : A set of (potentially) actionable chunks.
        """

        self.chunks = chunks

    @abc.abstractmethod
    def __call__(self, input_map: node.Node2Float) -> node.ChunkSet:
        """Identify chunks that are currently actionable based on their 
        strengths.

        kwargs:
            input_map : A dict mapping nodes (Chunks and/or Features) to 
            activations.
        """

        pass

class Handler(object):
    """Generic class for handling chunk-driven action execution.

    Can be used out of the box to link action chunks to callbacks implementing 
    relevant actions.
    """

    def __init__(self, chunk2action : node.Chunk2Callable) -> None:
        """Initialize an action handler.

        kwargs:
            chunk2action : A mapping from (action) chunks to actions.
        """

        self.chunk2action = chunk2action
    
    def __call__(self, chunks : node.ChunkSet) -> None:
        """Execute selected actions.

        kwargs:
            chunks : A set of chunks representing selected actions.
        """
        
        for chunk in chunks:
            try:
                # Try executing action attached to chunk
                self.chunk2action[chunk]()
            except KeyError:
                continue


####### TYPE ALIASES #######

SelectorType = T.Type[Selector]
HandlerType = T.Type[Handler]