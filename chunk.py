import typing as T
from feature import Dim2Float, Feature2Float, FeatureSet

class Chunk(object):
    """A basic Clarion chunk. 

    In Clarion, chunks are explicit, localist representations. They live in the 
    top level and may be connected to (micro)features, which live in the bottom 
    level. Activation may flow from (micro)features to linked chunks (bottom-up 
    activation), or it may flow from chunks to (micro)features (top-down 
    activation).

    Only essential features of Clarion chunks are represented in this class. 
    More advanced or specialized features may be added using specialized Mixin 
    classes (see, e.g., bla.BLAMixin).

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

# Types
ChunkSet = T.Set[Chunk]
Chunk2Float = T.Mapping[Chunk, float]
Chunk2Callable = T.Mapping[Chunk, T.Callable]