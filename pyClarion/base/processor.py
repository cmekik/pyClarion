import abc
import numpy as np
from typing import Generic, TypeVar, Iterable, Dict, Mapping, Callable
from pyClarion.base.knowledge import get_nodes, Node, Chunk
from pyClarion.base.packet import Packet, ActivationPacket, SelectorPacket


################
# ABSTRACTIONS #
################

Mt = TypeVar('Mt', bound=Packet)
Pt = TypeVar('Pt', bound=ActivationPacket)
St = TypeVar('St', bound=SelectorPacket)
At = TypeVar('At')


class Channel(Generic[Pt, At], abc.ABC):
    """An abstract generic class for capturing activation flows.

    This class that provides an interface for handling basic activation flows. 
    Activation flows are implemented in the ``__call__`` method. 

    See module documentation for examples and details.
    """
    
    @abc.abstractmethod
    def __call__(self, input_map: Pt) -> ActivationPacket[At]:
        """Compute and return activations resulting from an input to this 
        channel.

        .. note::
            Assumptions about missing expected nodes in the input map should 
            be explicitly specified/documented, along with behavior for handling 
            such cases. 

        :param input_map: An activation packet representing the input to this 
            channel.
        """

        pass


class Junction(Generic[Pt, At], abc.ABC):
    """Combines activations flows from multiple sources."""

    @abc.abstractmethod
    def __call__(self, *input_maps: Pt) -> ActivationPacket[At]:
        """Return a combined mapping from chunks and/or microfeatures to 
        activations.

        kwargs:
            input_maps : Dicts mapping from chunks and/or microfeatures 
            to input activations.
        """

        pass


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


class Effector(Generic[St], abc.ABC):
    '''An abstract class for linking actionable chunks to action callbacks.'''
    
    @abc.abstractmethod
    def __call__(self, selector_packet : St) -> None:
        '''
        Execute actions associated with given actionable chunk.

        :param selector_packet: The output of an action selection cycle.
        '''
        pass


################
### GENERICS ###
################


class MaxJunction(Junction[Pt, At]):
    """An activation junction returning max activations for all input nodes.
    """

    def __call__(self, *input_maps : Pt) -> ActivationPacket:
        """Return the maximum activation value for each input node.

        kwargs:
            input_maps : A set of mappings from chunks and/or (micro)features 
            to activations.
        """

        node_set = get_nodes(*input_maps)
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