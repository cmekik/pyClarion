"""
Tools for creating and manipulating activation and decision patterns.
"""

import abc
import numpy as np
from typing import Generic, TypeVar, Iterable, Dict, Mapping, Callable, Union
from pyClarion.base.symbols import get_nodes, Node, Chunk
from pyClarion.base.packet import Packet, ActivationPacket, DecisionPacket


######################
### TYPE VARIABLES ###
######################


Pt = TypeVar('Pt', bound=Packet)
It = TypeVar('It', bound=ActivationPacket)
Ot = TypeVar('Ot', bound=ActivationPacket)
St = TypeVar('St', bound=DecisionPacket)


################
# ABSTRACTIONS #
################


class Channel(Generic[It, Ot], abc.ABC):
    """
    Transforms an activation pattern.

    ``Channel`` instances are callable objects whose ``__call__`` method 
    expects a single ``ActivationPacket`` and outputs a single 
    ``ActivationPacket``.

    Channels may be defined to capture various constructs such as activation 
    flows and buffers.
    """
    
    @abc.abstractmethod
    def __call__(self, input_map: It) -> Ot:
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


class Junction(Generic[It, Ot], abc.ABC):
    """Combines activations flows from multiple sources.

    ``Junction`` instances are callable objects whose ``__call__`` method 
    expects multiple ``ActivationPacket``s and outputs a single combined 
    ``ActivationPacket``.

    Junctions may be defined to capture constructs such as Chunk or Microfeature 
    nodes.
    """

    @abc.abstractmethod
    def __call__(self, *input_maps: It) -> Ot:
        """Return a combined mapping from chunks and/or microfeatures to 
        activations.

        kwargs:
            input_maps : Dicts mapping from chunks and/or microfeatures 
            to input activations.
        """

        pass


class Selector(Generic[It, St], abc.ABC):
    """
    Selects output chunks.

    ``Selector`` instances are callable objects whose ``__call__`` method 
    expects a single ``ActivationPacket``s and outputs a single 
    ``DecisionPacket``.

    Selectors may be defined to capture Appraisal constructs.
    """

    @abc.abstractmethod
    def __call__(self, input_map: It) -> St:
        """
        Select output chunk(s).

        :param input_map: Strengths of input nodes.
        """

        pass


class Effector(Generic[St], abc.ABC):
    """
    Links output chunks to callbacks.

    ``Effector`` instances are callable objects whose ``__call__`` method 
    expects a single ``DecisionPacket`` and outputs nothing. The method executes 
    callbacks according to the contents of its input. 

    Effectors may be defined to capture Behavior constructs.
    """
    
    @abc.abstractmethod
    def __call__(self, selector_packet : St) -> None:
        '''
        Execute actions associated with given output.

        :param selector_packet: The output of an action selection cycle.
        '''
        pass


#################################
### ACTIVATION PROCESSOR TYPE ###
#################################


ActivationProcessor = Union[Channel, Junction, Selector, Effector]


################
### GENERICS ###
################


class MaxJunction(Junction[It, ActivationPacket]):
    """An activation junction returning max activations for all input nodes.
    """

    def __call__(self, *input_maps : It) -> ActivationPacket:
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


class BoltzmannSelector(Selector[It, DecisionPacket[float]]):
    """Select a chunk according to a Boltzmann distribution.
    """

    def __init__(self, chunks : Iterable[Chunk], temperature: float) -> None:
        """Initialize a ``BoltzmannSelector`` instance.

        
        :param chunks: An iterable of (potentially) actionable chunks.
        :param temperature: Temperature of the Boltzmann distribution.
        """

        self.actionable_chunks = chunks
        self.temperature = temperature

    def __call__(self, input_map: Pt) -> DecisionPacket[float]:
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
        return DecisionPacket(boltzmann_distribution, choices)

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