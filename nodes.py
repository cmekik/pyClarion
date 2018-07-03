"""This module defines the basic representational constructs of the Clarion 
cognitive architecture.

Overall, the most basic representational construct in Clarion is a connectionist 
node: an individual unit that may be connected to other units whose 
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
import enum


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

    def dim(self) -> enum.EnumMeta:
        """Return dimension associated with self.
        """

        return type(self)

# (Micro)Feature-Related Types

Dim2Float = T.Mapping[enum.EnumMeta, float]
FeatureSet = T.Set[Feature]
Feature2Float = T.Mapping[Feature, float]

# (Micro)Feature-Related Functions

def get_all_microfeatures() -> FeatureSet:
    """Return all defined microfeatures.

    Note: Searches all direct subclasses of Feature for dimension-value pairs.
    """

    microfeatures = set()
    for subclass in Feature.__subclasses__():
        microfeatures.update(list(subclass))
    return microfeatures


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
        if label is not None:
            self.label = label

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

# Chunk-Related Types

ChunkSet = T.Set[Chunk]
Chunk2Float = T.Mapping[Chunk, float]
Chunk2Callable = T.Mapping[Chunk, T.Callable]


####### NODES #######
    # Since (Micro)Features and Chunks have been defined, nodes can be defined 
    # as the union type of these two types.

Node = T.Union[Chunk, Feature]

# Node-Related Types

NodeSet = T.Set[Node]
Node2Float = T.Mapping[Node, float]