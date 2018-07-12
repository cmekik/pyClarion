"""This module defines the basic representational constructs of the Clarion 
cognitive architecture.

Overall, the most basic representational construct in Clarion is a connectionist 
node: an individual unit that may be connected to other units and whose 
activation depends on the activation of other units in its network and the 
strength of incoming connections. 

Clarion has a dual representational architecture, meaning that there are, 
broadly, two kinds of node: (micro)features and chunks. These two types of node 
are represented by the Feature and Chunk classes respectively. A number of 
types related to these two classes are also provided for facilitating type 
hints in collaborator classes. There is no generic Node class, however a Node 
type is defined (as the union of Feature and Chunk). Node-related utility types 
are also provided for convenience.

For details on representation in Clarion, see Chapter 3 of Sun (2016).

References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press. 
"""


import typing as T
import abc
import enum
import weakref


####### (MICRO)FEATURES #######

@enum.unique
class Feature(enum.Enum):
    """An abstract Clarion (micro)feature.

    In Clarion, (micro)features are implicit, connectionist representations. 
    They are represented as dimension-value pairs:
        e.g. (Color: White), (Shape: Star)
    In this implementation, dimensions are represented by subclasses of the 
    Feature class, which is itself an enumeration class. Members of Feature 
    subclasses correspond to individual dimension-value pairs.

    For details on dimension-value pairs, see Chapter 3 of Sun (2016), starting 
    with Section 3.1.2.1.

    References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press. 
    """
    pass

    def __repr__(self):
        """Return a string representation of self.

        Should look like this:
            (Dimension, Value)
        """
        return (self.__class__.__name__ + ": " + self.name).join(["(",")"])

    def dim(self) -> T.Type:
        """Return dimension associated with self.
        """

        return type(self)

def all_features() -> T.Set[Feature]:
    """Return a set containing all existing microfeatures.

    Note: Searches all direct subclasses of Feature for microfeatures.
    """

    microfeatures = set()
    for subclass in Feature.__subclasses__():
        microfeatures.update(list(subclass))
    return microfeatures

# (Micro)Feature-Related Types

Dim2Float = T.Dict[enum.EnumMeta, float]
FeatureSet = T.Set[Feature]
Feature2Float = T.Dict[Feature, float]


####### CHUNKS #######

class Chunk(object):
    """A basic Clarion chunk. 

    In Clarion, chunks are explicit, localist representations. They live in the 
    top level and may be connected to (micro)features, which live in the bottom 
    level. Activation may flow from (micro)features to linked chunks (bottom-up 
    activation), or it may flow from chunks to (micro)features (top-down 
    activation).

    This implementation is based on Chapter 3 of Sun (2016).

    References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    _instances : weakref.WeakSet = weakref.WeakSet()

    def __init__(
        self, 
        microfeatures: FeatureSet,
        dim2weight: T.Optional[Dim2Float] = None,
        label: T.Optional[str] = None
    ) -> None:
        """Initialize a Clarion Chunk.

        By default, top-down weights are set to 1.0 if no weights are provided.

        Args:
            microfeatures : Set of dv-pairs.
            dim2weight : A mapping from each chunk dimension to its top-down 
                weight.
            label : Semantic label of chunk.
        """
        
        self.microfeatures = microfeatures
        self.dim2weight = self.initialize_weights(dim2weight)
        self.label = label

        # Add self to global chunk instance collection.
        self._instances.add(self)

    def __repr__(self):
        """Return a string representation of self.

        Should look like this:
            <Chunk: 'chunk-label' {(Dim1, Val1), (Dim2, Val2), ...}>
        """

        header = self.__class__.__name__ + ": "
        if self.label is not None:
            header += self.label.join(["'","'"]) + " "
        return (header + self.microfeatures.__repr__()).join(["<", ">"])

    def initialize_weights(
        self, 
        dim2weight: T.Optional[Dim2Float]
    ) -> Dim2Float:
        """Initialize top-down weights.

        If input is None, weights are initialized to 1.0.

        Args:
            dim2weight: A mapping from each chunk dimension to its top-down 
                weight.
        """
        
        if dim2weight is None:
            dim2weight = dict()
            for f in self.microfeatures:
                dim2weight[f.dim()] = 1.
        return dim2weight

def all_chunks() -> T.Set[Chunk]:
    """Return a set containing all existing chunks.
    """

    return set(Chunk._instances)

# Chunk-Related Types

ChunkSet = T.Set[Chunk]
Chunk2Float = T.Dict[Chunk, float]
Chunk2Callable = T.Dict[Chunk, T.Callable]


####### NODES #######
    # Since (Micro)Features and Chunks have been defined, nodes can be defined 
    # as the union type of these two types.

Node = T.Union[Chunk, Feature]

# Node-Related Types

NodeIterable = T.Iterable[Node]
NodeSet = T.Set[Node]
Node2Any = T.Dict[Node,T.Any]
Node2Float = T.Dict[Node, float]

Any2NodeSet = T.Dict[T.Any, NodeSet]

# Node-Related Classes

class Node2ValueFilter(object):
    """Filters mappings from nodes to values.

    This filter can be used for input/output filtering, assuming that 
    downstream activation channels have well-defined default behavior for 
    coping with missing expected inputs.
    """

    def __init__(
        self, filter_map : Any2NodeSet = None
    ) -> None:
        """Initialize an activation filter.

        kwargs:
            filter_map : A mapping from filter keys to node sets to be filtered. 
                There is no particular restriction on the type of filter keys.
        """

        if filter_map is None:
            filter_map = dict()

        self.map = filter_map

    def __call__(
        self, input_map : Node2Any, filter_keys : T.Iterable = None
    ) -> Node2Any:
        """Return a filtered mapping from nodes to values.

        Suppresses nodes designated by the filter keys.
        
        Warnings: 
            - Removes nodes to be filtered when encountered in the input map.
            - If a filter key is not in the filter map, the key is ignored. 

        kwargs:
            input_map : An activation map to be filtered.
            filter_key : An iterable collection of keys for selecting nodes to 
                be filtered.
        """

        if filter_keys is not None:
            filter_nodes : NodeSet = set.union(
                *[self.map.get(key, set()) for key in filter_keys]
            )
            filtered = {
                k:v for (k,v) in input_map.items() if k not in filter_nodes
            }
        else:
            filtered = input_map
        return filtered

# Node-Related Functions

def all_nodes() -> NodeSet:
    """Return a set containing all existing nodes.
    """

    microfeatures : NodeSet = T.cast(NodeSet, all_features())
    chunks : NodeSet = T.cast(NodeSet, all_chunks())

    return set.union(microfeatures, chunks)

def get_nodes(*node_iterables: NodeIterable) -> NodeSet:
    """Return a set containing all nodes appearing in at least once in the 
    input mappings.

    kwargs:
        node_iterables : A sequence iterables containing nodes.
    """

    node_set = set()
    for node_iterable in node_iterables:
        for node in node_iterable:
            node_set.add(node)
    return node_set