"""This module provides tools for computing and handling node activations in the 
Clarion cognitive architecture. 

Activations may flow from chunks to chunks (rules), chunks to features 
(top-down activation), features to chunks (bottom-up activation), and features 
to chunks or features (implicit activation). Activations may also be filtered at 
various stages based on input from the meta-cognitive subsystem (MCS) or other 
sources. Furthermore, activations from several different sources may need to be 
combined. For example, activations from the top and bottom levels of the 
action-centered subsystem (ACS) may need to be combined in order to complete an 
action-decision making cycle. 

Here, the various processes described above are captured by means of two 
abstractions: activation channels and junctions. Activation channels implement 
mappings from node activations to node activations. They may be used to 
represent, among others, filtering, top-down activation, bottom-up activation, 
rule-based activation, and implicit activation processes. Activation junctions 
implement routines for combining inputs from various sources. They may be used, 
for example, for combining activations from the top and botom levels of the ACS 
for action decision-making.

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
    is assumed that activations will be represented as mappings from nodes to 
    activation values. Thus, activation channels expect such mappings as input, 
    and return such mappings as output. Output mappings are allowed to be empty 
    when such behavior is sensible.
    
    It is assumed that an activation channel will pay attention only to the 
    activations that are relevant to the computation it implements. For 
    instance, if an activation class implementing a bottom-up connection is 
    passed a bunch of chunk activations, it should simply ignore these and look 
    for matching (micro)features. 
    
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
        be explicitly documented, along with behavior for handling such cases. 

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """

        pass

ChannelSet = T.Set[Channel]
ChannelTypeSet = T.Set[T.Type[Channel]]

class Junction(abc.ABC):
    """An abstract class for handling the combination of chunk and/or 
    (micro)feature activations from multiple sources.

    For instance, this class may be used to combine chunk strengths from 
    multiple sources for action decision making. 
    """

    @abc.abstractmethod
    def __call__(self, *input_maps: node.Node2Float) -> node.Node2Float:
        """Return a combined mapping from chunks and/or (micro)features to 
        activations.

        kwargs:
            input_maps : A set of mappings from chunks and/or (micro)features 
            to activations.
        """

        pass


####### GENERIC FUNCTIONS #######

def select_channels_by_type(
    channels : ChannelSet, 
    channel_types : ChannelTypeSet
) -> ChannelSet:
    """
    """

    selected : T.Set[Channel] = set()
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

def filter_call(
    input_map : node.Node2Float,
    channel : Channel,
    node_map_filter : node.Node2ValueFilter,  
    input_keys : T.Iterable = None, 
    output_keys : T.Iterable = None
) -> node.Node2Float:
    """Passes input through channel with given input and output filters.
    """

    filtered_input = node_map_filter(input_map, input_keys)
    raw_output = channel(filtered_input)
    filtered_output = node_map_filter(raw_output, output_keys)
    return filtered_output

def filter_propagate(
    input_map : node.Node2Float,
    channels : ChannelSet,
    node_map_filter : node.Node2ValueFilter,
    junction : Junction,
    input_key_map : T.Dict[Channel, T.Iterable] = None,
    output_key_map : T.Dict[Channel, T.Iterable] = None
) -> node.Node2Float:
    """
    """

    if input_key_map is None:
        input_key_map = dict()
    if output_key_map is None:
        output_key_map = dict()

    channel_outputs = []
    for channel in channels:
        input_keys = input_key_map.get(channel)
        output_keys = output_key_map.get(channel)
        channel_output = filter_call(
            input_map, channel, node_map_filter, input_keys, output_keys
        )
        channel_outputs.append(channel_output) 
    return junction(*channel_outputs)


####### BASE ACTIVATION CHANNELS #######

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

class Rule(Channel):
    """A base class for rule activation channels.

    This is an abstract interface for various possible implementations of 
    rule activation channels. 
    """

    pass

class Implicit(Channel):
    """A base class for implicit activation channels.

    This is an abstract interface for various possible implementations of 
    implicit activation channels. These may include multi-layer perceptrons for 
    action decision-making, autoassociative networks for implicit reasoning, and 
    others. 
    """

    pass