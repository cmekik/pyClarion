"""This module defines the basic representational constructs of the Clarion 
cognitive architecture.

The most basic representational construct in Clarion is a connectionist node: 
an individual unit that may receive activation from and propagate activation to 
other units. 

Clarion has a dual representational architecture. There are two kinds of node: 
microfeatures and chunks. These two types of node are represented by the 
Microfeature and Chunk classes respectively. A number of types related to these 
two classes are also provided for facilitating type hints in collaborator 
classes.

For details on representation in Clarion, see Chapter 3 of Sun (2016).

References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press. 
"""


import typing as T
import abc
import collections


####### MICROFEATURES ########

class Microfeature(T.NamedTuple):
    """A Clarion microfeature.

    Microfeatures are implicit, connectionist representations. They are 
    represented as dimension-value pairs:
        e.g. (Color: White), (Shape: Star)

    For details on dimension-value pairs, see Chapter 3 of Sun (2016), starting 
    with Section 3.1.2.1.

    References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press. 
    """
    
    dim : T.Hashable
    val : T.Hashable

# Type Aliases

FeatureSet = T.Set[Microfeature]
Feature2Float = T.Dict[Microfeature, float]
Dim2Float = T.Dict[T.Hashable, float]


####### CHUNKS #######

class Chunk(object):
    """A basic Clarion chunk. 

    Chunks are explicit, localist representations. They may be linked to 
    microfeatures.

    This implementation is based on Chapter 3 of Sun (2016).

    References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(
        self, 
        microfeatures: FeatureSet,
        dim2weight: T.Dict[T.Any, float],
        label: str = None
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
        self.dim2weight = dim2weight
        self.label = label

    def __repr__(self):
        """Return a string representation of self.
        """

        header = self.__class__.__name__ + ": "
        if self.label is not None:
            header += self.label.join(["'","'"]) + " "
        return (header + self.microfeatures.__repr__()).join(["<", ">"])

# Type Aliases

ChunkSet = T.Set[Chunk]
Chunk2Float = T.Dict[Chunk, float]
Chunk2Callable = T.Dict[Chunk, T.Callable]


####### NODES #######

# Type Aliases

Node = T.Union[Microfeature, Chunk]
NodeIterable = T.Iterable[Node]
NodeSet = T.Set[Node]
Node2Any = T.Dict[Node,T.Any]
Node2Float = T.Dict[Node, float]
Any2NodeSet = T.Dict[T.Any, NodeSet]

# Functions

def get_nodes(*node_iterables: NodeIterable) -> NodeSet:
    """Return a set containing all nodes appearing in at least once in the 
    input iterables.

    kwargs:
        node_iterables : A sequence of iterables containing nodes.
    """

    node_set = set()
    for node_iterable in node_iterables:
        for node in node_iterable:
            node_set.add(node)
    return node_set