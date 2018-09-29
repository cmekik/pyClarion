"""
Tools for creating and manipulating activation and decision patterns.
"""

import abc
import numpy as np
from typing import Generic, TypeVar, Iterable, Dict, Mapping, Callable, Union, Set
from pyClarion.base.symbols import get_nodes, Node, Chunk
from pyClarion.base.packets import ActivationPacket, DecisionPacket, At


################
# ABSTRACTIONS #
################


class Channel(Generic[At], abc.ABC):
    """
    Transforms an activation pattern.

    ``Channel`` instances are callable objects whose ``__call__`` method 
    expects a single ``ActivationPacket`` and outputs a single 
    ``ActivationPacket``.

    Channels may be defined to capture various constructs such as activation 
    flows and buffers.
    """
    
    @abc.abstractmethod
    def __call__(self, input_map: ActivationPacket[At]) -> ActivationPacket[At]:
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


class Junction(Generic[At], abc.ABC):
    """Combines activations flows from multiple sources.

    ``Junction`` instances are callable objects whose ``__call__`` method 
    expects multiple ``ActivationPacket``s and outputs a single combined 
    ``ActivationPacket``.

    Junctions may be defined to capture constructs such as Chunk or Microfeature 
    nodes.
    """

    def __init__(self, default_activation: Callable = None) -> None:

        self.default_activation = default_activation

    @abc.abstractmethod
    def __call__(self, *input_maps: ActivationPacket[At]) -> ActivationPacket:
        """Return a combined mapping from chunks and/or microfeatures to 
        activations.

        kwargs:
            input_maps : Dicts mapping from chunks and/or microfeatures 
            to input activations.
        """

        pass


class Selector(Generic[At], abc.ABC):
    """
    Selects output chunks.

    ``Selector`` instances are callable objects whose ``__call__`` method 
    expects a single ``ActivationPacket``s and outputs a single 
    ``DecisionPacket``.

    Selectors may be defined to capture Appraisal constructs.
    """

    @abc.abstractmethod
    def __call__(self, input_map: ActivationPacket[At]) -> DecisionPacket[At]:
        """
        Select output chunk(s).

        :param input_map: Strengths of input nodes.
        """

        pass


class Effector(Generic[At], abc.ABC):
    """
    Links output chunks to callbacks.

    ``Effector`` instances are callable objects whose ``__call__`` method 
    expects a single ``DecisionPacket`` and outputs nothing. The method executes 
    callbacks according to the contents of its input. 

    Effectors may be defined to capture Behavior constructs.
    """
    
    @abc.abstractmethod
    def __call__(self, selector_packet : DecisionPacket[At]) -> None:
        '''
        Execute actions associated with given output.

        :param selector_packet: The output of an action selection cycle.
        '''
        pass


class Buffer(Generic[At], abc.ABC):
    """ """

    @abc.abstractmethod
    def __call__(self, selector_packet: DecisionPacket[At]) -> ActivationPacket[At]:
        """ """

        pass


#################################
### ACTIVATION PROCESSOR TYPE ###
#################################


ActivationProcessor = Union[Channel, Junction, Selector, Effector]


################
### GENERICS ###
################


class UpdateJunction(Junction[At]):
    """Merges input activation packets using the packet ``update`` method."""

    def __call__(
        self, *input_maps: ActivationPacket[At]
    ) -> ActivationPacket[At]:

        output: ActivationPacket = ActivationPacket(
            default_factory=self.default_activation
        )
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

        node_set = get_nodes(*input_maps)
        output : ActivationPacket = ActivationPacket(
            default_factory=self.default_activation
        )
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