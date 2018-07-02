import abc
import typing as T
from feature import Feature2Float
from chunk import Chunk2Callable, ChunkSet, Chunk2Float

class ChunkSelector(abc.ABC):
    """An abstract class defining the interface for selection of actionable 
    chunks based on chunk strengths.
    """

    def __init__(self, chunks: ChunkSet) -> None:
        """Initialize a chunk selector.

        kwargs:
            chunks : A set of (potentially) actionable chunks.
        """

        self.chunks = chunks

    @abc.abstractmethod
    def __call__(self, chunk2strength: Chunk2Float) -> ChunkSet:
        """Identify chunks that are currently actionable based on their 
        strengths.

        kwargs:
            chunk2strength : A mapping from chunks to their strengths.
        """

        pass

class ActionHandler(object):
    """Handles chunk-driven action execution.

    Use this class to link action chunks to callbacks implementing relevant 
    actions.
    """

    def __init__(self, chunk2action : Chunk2Callable) -> None:
        """Initialize an action handler.

        kwargs:
            chunk2action : A mapping from (action) chunks to actions.
        """

        self.chunk2action = chunk2action
    
    def __call__(self, chunks : ChunkSet) -> None:
        """Execute selected actions.

        kwargs:
            chunks : A set of chunks representing selected actions.
        """
        
        for chunk in chunks:
            try:
                self.chunk2action[chunk].__call__()
            except KeyError:
                continue