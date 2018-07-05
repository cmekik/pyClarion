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
import nodes


####### ABSTRACTIONS #######

class ActivationChannel(abc.ABC):
    """An abstract class for capturing activation flows.

    This class provides a uniform interface for handling activation flows. It 
    is assumed that activations will be represented as mappings from nodes to 
    activation values. Thus, activation channels expect such mappings as input, 
    and return such mappings as output. Output mappings are allowed to be empty 
    when such behavior is sensible (see e.g., TopDown).
    
    It is assumed that an activation channel will pay attention only to the 
    activations that are relevant to the computation it implements. For 
    instance, if an activation class implementing a bottom-up connection is 
    passed a bunch of chunk activations, it should simply ignore these and look 
    for matching (micro)features. 
    
    Likewise if an activation channel is handed an input that does not contain 
    a complete activation mapping for expected nodes (e.g., due to filtering), 
    it should not fail. Instead, it should have a well-defined default behavior
    for such cases. 
    
    In general, it may be assumed that when an expected node is missing in the 
    input, its activation is equal to 0. ActivationChannel subclasses in this 
    module have been written according to this assumption.
    """
    
    @abc.abstractmethod
    def __call__(self, input_map : nodes.Node2Float) -> nodes.Node2Float:
        """Compute and return activations resulting from an input to this 
        channel.

        Note: Assumptions about missing expected nodes in the input map should 
        be explicitly documented, along with behavior for handling such cases. 

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """

        pass

class ActivationJunction(abc.ABC):
    """An abstract class for handling the combination of chunk and/or 
    (micro)feature activations from multiple sources.

    For instance, this class may be used to combine chunk strengths from 
    multiple sources for action decision making. 
    """

    @abc.abstractmethod
    def __call__(self, *input_maps: nodes.Node2Float) -> nodes.Node2Float:
        """Return a combined mapping from chunks and/or (micro)features to 
        activations.

        kwargs:
            input_maps : A set of mappings from chunks and/or (micro)features 
            to activations.
        """

        pass


####### STANDARD ACTIVATION CHANNELS #######

class TopDown(ActivationChannel):
    """A top-down (from a chunk to its microfeatures) activation channel.

    For details, see Section 3.2.2.3 para 2 of Sun (2016).

    References:
        Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(self, chunk : nodes.Chunk):
        """Set up the top-down activation flow from chunk to its microfeatures.

        kwargs:
            chunk : Source chunk for top-down activation.
        """
        
        self.chunk = chunk
        self.val_counts = self.count_values(self.chunk.microfeatures)

    def __call__(self, input_map : nodes.Node2Float) -> nodes.Node2Float:
        """Compute bottom-level activations given current chunk strength.

        Note: If the expected chunk is missing from the input, this channel 
        outputs an empty mapping.

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """

        try:
            strength = input_map[self.chunk]
        except KeyError:
            return dict()
        else:        
            activations = dict()
            for f in self.chunk.microfeatures:
                w_dim = self.chunk.dim2weight[f.dim()] 
                n_val = self.val_counts[f.dim()]
                activations[f] = strength * w_dim / n_val
            return activations

    @staticmethod
    def count_values(microfeatures: nodes.FeatureSet) -> nodes.Dim2Float:
        """Count the number of features in each dimension
        
        kwargs:
            microfeatures : A set of dv-pairs.
        """
        
        counts = dict()
        for f in microfeatures:
            try:
                counts[f.dim()] += 1
            except KeyError:
                counts[f.dim()] = 1
        return counts

class BottomUp(ActivationChannel):
    """A top-down (from microfeatures to a chunk to its microfeatures) 
    activation channel.

    For details, see Section 3.2.2.3 para 3 of Sun (2016).

    References:
        Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(self, chunk : nodes.Chunk):
        """Set up the bottom-up activation flow to chunk from its microfeatures.

        kwargs:
            chunk : Target chunk for bottom-up activation.
        """
        
        self.chunk = chunk

    def __call__(self, input_map : nodes.Node2Float) -> nodes.Node2Float:
        """Compute chunk strength given current bottom-level activations.

        Note: If an expected (micro)feature is missing from the input, its 
        activation is assumed to be 0.

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """

        # For each dimension, pick the maximum available activation of the 
        # target chunk microfeatures in that dimension. 
        dim2activation = dict()
        for f in self.chunk.microfeatures:
            try:
                if dim2activation[f.dim()] < input_map[f]:
                    dim2activation[f.dim()] = input_map[f]
            except KeyError:
                # A KeyError may arise either because f.dim() is not in 
                # dim2activation, or because f is not in input_map. 
                # If the former is the case, add f.dim() to dim2activation, 
                # otherwise move on to the next microfeature.
                try: 
                    dim2activation[f.dim()] = input_map[f]
                except KeyError:
                    continue
        # Compute chunk strength based on dimensional activations.
        strength = 0.
        for dim in dim2activation:
            strength += self.chunk.dim2weight[dim] * dim2activation[dim]
        else:
            strength /= sum(self.chunk.dim2weight.values()) ** 1.1
        return {self.chunk: strength}

class Rule(ActivationChannel):
    """A basic Clarion associative rule.

    Rules have the form:
        chunk_1 chunk_2 chunk_3 -> chunk_4
    Chunks in the left-hand side are condition chunks, the single chunk in the 
    right-hand side is the conclusion chunk. The strength of the conclusion 
    chunk resulting from rule application is a weighted sum of the strengths 
    of the condition chunks.

    This implementation is based on Chapter 3 of Sun (2016). 

    References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(
        self,
        chunk2weight : nodes.Chunk2Float,
        conclusion_chunk : nodes.Chunk
    ) -> None:
        """Initialize a Clarion associative rule.

        kwargs:
            chunk2weight : A mapping from condition chunks to their weights.
            conclusion_chunk : The conclusion chunk of the rule.
        """
        
        self.chunk2weight = chunk2weight
        self.conclusion_chunk = conclusion_chunk
    
    def __call__(self, input_map : nodes.Node2Float) -> nodes.Node2Float:
        """Return strength of conclusion chunk resulting from an application of 
        current associative rule.

        Note: If an expected chunk is missing from the input, its activation is 
        assumed to be 0.

        kwargs:
            input_map : A mapping from nodes (Chunks and/or Features) to 
            input activations.
        """
        
        strength = 0. 
        for chunk in self.chunk2weight:
            try:
                strength += input_map[chunk] * self.chunk2weight[chunk]
            except KeyError:
                continue
        return {self.conclusion_chunk: strength}

class Implicit(ActivationChannel):
    """An implicit activation channel.

    This is an abstract interface for various possible implementations of 
    implicit activation channels. These may include multi-layer perceptrons for 
    action decision-making, autoassociative networks for implicit reasoning, and 
    others. 
    """
    pass


####### STANDARD ACTIVATION JUNCTIONS #######

class MaxJunction(ActivationJunction):
    """An activation junction returning max activations for all input nodes.
    """

    def __call__(self, *input_maps):
        """Return the maximum activation value for each input node.

        kwargs:
            input_maps : A set of mappings from chunks and/or (micro)features 
            to activations.
        """

        node_set = nodes.get_nodes(*input_maps)

        activations = dict()
        for node in node_set:
            for input_map in input_maps:
                try:
                    if activations[node] < input_map[node]:
                        activations[node] = input_map[node]
                except KeyError:
                    # Two cases: either node not in activations or node not in 
                    # input_map. Assume node not in activations, but in input 
                    # map, if this fails continue.
                    try:
                        activations[node] = input_map[node]
                    except KeyError:
                        continue

        return activations


####### FILTERING #######

class ActivationFilter(ActivationChannel):
    """Class for filtering inputs of an activation channel.
    """

    def __init__(self, node_set : nodes.NodeSet = None):
        """Initialize an activation filter.

        kwargs:
            node_set : A set of nodes to be filtered out.
        """

        if node_set is None:
            node_set = set()

        self.filter_nodes = node_set

    def __call__(self, input_map : nodes.Node2Float) -> nodes.Node2Float:
        """Return a filtered activation map.

        kwargs:
            input_map : An activation map to be filtered.
        """

        filtered = {
            k:v for (k,v) in input_map.items() if k not in self.filter_nodes
        }
        return filtered


####### FUNCTIONS #######

def with_channels(input_map, channels, junction):

    return junction(*[channel(input_map) for channel in channels])

def _with_filter(activation_filter, input_map):
    """Helper function for with_filter.
    """

    if activation_filter is None:
        return input_map
    else:
        return activation_filter(input_map)

def with_filter(
    channel : ActivationChannel, 
    input_map : nodes.Node2Float, 
    input_filter : ActivationFilter = None, 
    output_filter : ActivationFilter = None
) -> nodes.Node2Float:
    """Passes input through channel with given input and output filters.
    """

    filtered_input = _with_filter(input_filter, input_map)
    raw_output = channel(filtered_input)
    filtered_output = _with_filter(output_filter, raw_output)
    return filtered_output