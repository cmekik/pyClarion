"""This module provides tools for computing and handling node activations in the 
Clarion cognitive architecture. 

Activations may propagate within the top level (chunks to chunks; rules), 
top-down (chunks to microfeatures), bottom-up (microfeatures to chunks), and 
within the bottom-level (microfeatures to chunks or features; implicit 
activation). Activations from different sources may also be combined. 

The processes described above are captured by means of two abstractions: 
activation channels (Channel class) and junctions (Junction class). Channels 
implement mappings from node activations to node activations. Junctions 
implement routines for combining inputs from multiple channels.

For details of activation flows, see Chapter 3 of Sun (2016). Also, see Chapter 
4 for a discussion of filtering capabilities of MCS.

References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
"""


import abc
import typing as T
from . import node


####### ABSTRACTIONS #######

class Channel(abc.ABC):
    """An abstract class for capturing activation flows.

    This class provides a uniform interface for handling activation flows. It 
    is assumed that activations will be represented as dicts mapping nodes to 
    activation values. Thus, activation channels expect such mappings as input, 
    and return such mappings as output. Output mappings are allowed to be empty 
    when such behavior is sensible.
    
    It is assumed that an activation channel will pay attention only to the 
    activations that are relevant to the computation it implements. For 
    instance, if an activation class implementing a bottom-up connection is 
    passed a bunch of chunk activations, it should simply ignore these and look 
    for matching microfeatures. 
    
    Likewise if an activation channel is handed an input that does not contain 
    a complete activation mapping for expected nodes (e.g., due to filtering), 
    it should not fail. Instead, it should have a well-defined default behavior
    for such cases. 
    """
    
    @abc.abstractmethod
    def __call__(self, input_map : node.Node2Float) -> node.Node2Float:
        """Compute and return activations resulting from an input to this 
        channel.

        Note: Assumptions about missing expected nodes in the input map should 
        be explicitly specified/documented, along with behavior for handling 
        such cases. 

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """

        pass

class Junction(abc.ABC):
    """An abstract class for handling the combination of chunk and/or 
    (micro)feature activations from multiple sources.
    """

    @abc.abstractmethod
    def __call__(self, *input_maps: node.Node2Float) -> node.Node2Float:
        """Return a combined mapping from chunks and/or microfeatures to 
        activations.

        kwargs:
            input_maps : A set of mappings from chunks and/or microfeatures 
            to activations.
        """

        pass

# Type Aliases

ChannelSet = T.Set[Channel]
Channel2Iterable = T.Dict[Channel, T.Iterable]
ChannelType = T.Type[Channel]
ChannelTypeSet = T.Set[ChannelType]

JunctionType = T.Type[Junction]


####### BASIC CHANNEL TYPES #######

class TopDown(Channel):
    """A base class for top-down activation channels.

    This is an abstract interface for various possible implementations of 
    top-down activation channels. 
    """

    pass

class BottomUp(Channel):
    """A base class for bottom-up activation channels.

    This is an abstract interface for various possible implementations of 
    bottom-up activation channels. 
    """

    pass

class TopLevel(Channel):
    """A base class for top-level (i.e., explicit) activation channels.

    This is an abstract interface for various possible implementations of 
    top-level activation channels. 
    """

    pass

class BottomLevel(Channel):
    """A base class for bottom-level (i.e., implicit) activation channels.

    This is an abstract interface for various possible implementations of 
    bottom-level activation channels.
    """

    pass


####### FUNCTIONS #######

def select_channels_by_type(
    channels : ChannelSet, 
    channel_types : ChannelTypeSet
) -> ChannelSet:
    """Return a subset of channels that match the desired types.

    kwargs:
        channels : A set of channels to be filtered.
        channel_types : A set of channel types to be included in the output.
    """

    selected : ChannelSet = set()
    for channel in channels:
        for channel_type in channel_types:
            if isinstance(channel, channel_type):
                selected.add(channel)
            else:
                continue
    return selected

def propagate(
    input_map : node.Node2Float, 
    channels : ChannelSet, 
    junction : Junction
) -> node.Node2Float:
    """Propagate inputs through a set of channels, combine their outputs and 
    return the result.

    kwargs:
        input_map : A mapping from nodes to their current activations.
        channels : A set of activation channels.
        junction : An activation junction.
    """

    return junction(*[channel(input_map) for channel in channels])