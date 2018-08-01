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

class ActivationHandler(abc.ABC):
    """An abstract class for objects that handle node activations. 
    """

    @property
    @abc.abstractmethod
    def default_activation(self) -> float:
        """The assumed default value for activations. 
        """
        pass

class Channel(ActivationHandler):
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

class Junction(ActivationHandler):
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

class Split(ActivationHandler):
    """An abstract class for handling the splitting of chunk and/or 
    (micro)feature activations into multiple streams.
    """

    @abc.abstractmethod
    def __call__(
        self, input_map: node.Node2Float
    ) -> T.Iterable[node.Node2Float]:
        """Split a mapping from chunks and/or microfeatures to activations.

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """

        pass

class Selector(ActivationHandler):
    """An abstract class defining the interface for selection of actionable 
    chunks based on chunk strengths.
    """

    @abc.abstractmethod
    def __call__(
        self, input_map: node.Node2Float, actionable_chunks: node.ChunkSet
    ) -> node.ChunkSet:
        """Identify chunks that are currently actionable based on their 
        strengths.

        kwargs:
            input_map : A dict mapping nodes (Chunks and/or Features) to 
            activations.
        """

        pass

# Type Aliases

ChannelSet = T.Set[Channel]
Channel2Iterable = T.Dict[Channel, T.Iterable]
ChannelType = T.Type[Channel]
ChannelTypeSet = T.Set[ChannelType]

JunctionType = T.Type[Junction]
SelectorType = T.Type[Selector]


####### SPLITTING #######

class NodeTypeSplit(Split):

    def __call__(
        self, input_map : node.Node2Float
    ) -> T.Iterable[node.Node2Float]:

        microfeature_activations : node.Node2Float = dict()
        chunk_activations : node.Node2Float = dict()

        for node_, strength in input_map.items():
            if isinstance(node_, node.Microfeature):
                microfeature_activations[node_] = strength 
            elif isinstance(node_, node.Chunk):
                chunk_activations[node_] = strength

        return microfeature_activations, chunk_activations


####### FILTERING #######

class Filter(Channel):
    """An activation filter.
    """

    def __init__(self):

        self._nodes : node.NodeSet = set()

    def __call__(self, input_map : node.Node2Float) -> node.Node2Float:
        """Filter given activation map.
        
        Sets any nodes matching the filter to have the default activation value.
        
        kwargs:
            activation_map : A dict mapping nodes (Chunks and/or Features) to 
            activations.
        """
        
        output = activations_2_default(
            input_map, self.nodes, self.default_activation
        ) 
        return output

    @property
    def nodes(self) -> node.NodeSet:
        """The set of nodes caught by this filter
        """
        return self._nodes

    @nodes.setter
    def nodes(self, value : node.NodeSet) -> None:
        self._nodes = value

class InputFilterer(Channel):
    """A mixin for binding an input filter to a channel.
    """

    @property
    @abc.abstractmethod
    def input_filter(self) -> Filter:
        pass

    def __call__(self, input_map : node.Node2Float) -> node.Node2Float:
        
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

    def __call__(self, input_map : node.Node2Float) -> node.Node2Float:
        
        raw_output = super().__call__(input_map)
        output = self.output_filter(raw_output)
        return output


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

def propagate_channel_types(
    input_map : node.Node2Float,
    channels : ChannelSet,
    channel_types : ChannelTypeSet,
    junction : Junction
) -> node.Node2Float:
    """Propagate inputs only through channels of selected type.


    kwargs:
        input_map : A mapping from nodes to their current activations.
        channels : A set of activation channels.
        channel_types : A set of channel types to be matched.
        junction : An activation junction.
    """

    selected_channels = select_channels_by_type(
        channels = channels,
        channel_types = channel_types 
    )
    output = propagate(
        input_map = input_map,
        channels = selected_channels,
        junction = junction
    )
    return output

def activations_2_default(
    activation_map : node.Node2Float,
    target_nodes : node.NodeSet,
    default_activation : float
) -> node.Node2Float:
    """Sets activations of target nodes to default value.
    
    Leaves other activations unchanged.

    kwargs:
        activation_map : A dict mapping nodes to activations.
        target_nodes : Nodes to be matched.
        default_activation : Assumed default vaule.
    """
        
    filtered = dict()
    for node, activation in activation_map.items():
        if node in target_nodes:
            filtered[node] = default_activation
        else:
            filtered[node] = activation
    return filtered

def keep_microfeatures(
    activation_map : node.Node2Float
) -> node.Node2Float:
    """Return an activation map with all but microfeatures removed.

    kwargs:
        activation_map : A dict mapping nodes to activations.
    """
    
    output : node.Node2Float = {
        n : s for n, s in activation_map.items() 
        if isinstance(n, node.Microfeature)
    }
    return output

def keep_chunks(
    activation_map : node.Node2Float
) -> node.Node2Float:
    """Return an activation map with all but chunks removed.

    kwargs:
        activation_map : A dict mapping nodes to activations.
    """
    
    output : node.Node2Float = {
        n : s for n, s in activation_map.items() if isinstance(n, node.Chunk)
    }
    return output