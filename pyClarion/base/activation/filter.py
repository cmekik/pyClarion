"""This module provides generic tools for filtering nodes and attaching them to 
various activation handlers.
"""

import abc
from ..node import NodeSet
from .packet import ActivationPacket
from .channel import Channel


####### FILTERING #######

class Filter(Channel):
    """An activation filter.
    """

    def __init__(self):

        self._nodes : NodeSet = set()

    def __call__(self, input_map : ActivationPacket) -> ActivationPacket:
        """Filter given activation dict.
        
        Sets any nodes matching the filter to have the default activation value.
        
        kwargs:
            input_map : A dict mapping nodes (Chunks and/or Microfeatures) to 
            input activations.
        """
        
        output = activations_2_default(
            input_map, self.nodes
        ) 
        return output

    @property
    def nodes(self) -> NodeSet:
        """The set of nodes caught by this filter
        """
        return self._nodes

    @nodes.setter
    def nodes(self, value : NodeSet) -> None:
        self._nodes = value

class InputFilterer(Channel):
    """A mixin for binding an input filter to a channel.
    """

    @property
    @abc.abstractmethod
    def input_filter(self) -> Filter:
        pass

    def __call__(self, input_map : ActivationPacket) -> ActivationPacket:
        """Compute and return activations that result from a filtered input to 
        this channel.

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """

        filtered_input = self.input_filter(input_map)
        output = super().__call__(filtered_input)
        return output

class OutputFilterer(Channel):
    """A mixin for binding an output filter to a channel.
    """

    @property
    @abc.abstractmethod
    def output_filter(self) -> Filter:
        pass

    def __call__(self, input_map : ActivationPacket) -> ActivationPacket:
        """Compute, filter, and return activations that result from an input to 
        this channel.

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """

        raw_output = super().__call__(input_map)
        output = self.output_filter(raw_output)
        return output

def activations_2_default(
    activation_map : ActivationPacket,
    target_nodes : NodeSet
) -> ActivationPacket:
    """Sets activations of target nodes to default value.
    
    Leaves other activations unchanged.

    kwargs:
        activation_map : A dict mapping nodes to activations.
        target_nodes : Nodes to be matched.
        default_activation : Assumed default vaule.
    """
        
    filtered = activation_map.__class__()
    for node, activation in activation_map.items():
        if node in target_nodes:
            continue
        else:
            filtered[node] = activation
    return filtered