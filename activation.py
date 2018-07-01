import abc
from chunk import Chunk, Chunk2Float, ChunkSet
from feature import Dim2Float, FeatureSet, Feature2Float


class Activation(abc.ABC):
    """An abstract class for capturing activation flows.
    """
    
    def __call__(self, *args, **kwargs):
        pass

class TopDown(Activation):
    """A top-down (from a chunk to its microfeatures) activation channel.
    """

    def __init__(self, chunk : Chunk):
        """Set up the top-down activation flow from chunk to its microfeatures.

        kwargs:
            chunk : Source chunk for top-down activation.
        """
        
        self.chunk = chunk
        self.val_counts = self.count_values(self.chunk.microfeatures)

    def __call__(
        self, 
        chunk2strength : Chunk2Float, 
        microfeatures : FeatureSet
    ) -> Feature2Float:
        """Compute bottom-level activations given current chunk strength.

        For details, see Section 3.2.2.3 para 2 of Sun (2016).

        Args:
            chunk2strength : A mapping from chunks to their current strengths.
            microfeatures : A set of microfeatures for which top-down 
                activations may be computed.
        """

        try:
            strength = chunk2strength[self.chunk]
        except KeyError:
            return dict()
        else:        
            activations = dict()
            for f in self.chunk.microfeatures:
                if f in microfeatures:
                    w_dim = self.chunk.dim2weight[f.dim()] 
                    n_val = self.val_counts[f.dim()]
                    activations[f] = strength * w_dim / n_val
            return activations

    @staticmethod
    def count_values(microfeatures: FeatureSet) -> Dim2Float:
        """Count the number of features in each dimension
        
        Args:
            microfeatures : A set of dv-pairs.
        """
        
        counts = dict()
        for f in microfeatures:
            try:
                counts[f.dim()] += 1
            except KeyError:
                counts[f.dim()] = 1
        return counts

class BottomUp(Activation):
    """A top-down (from microfeatures to a chunk to its microfeatures) 
    activation channel.
    """

    def __init__(self, chunk : Chunk):
        """Set up the bottom-up activation flow to chunk from its microfeatures.

        kwargs:
            chunk : Target chunk for bottom-up activation.
        """
        
        self.chunk = chunk

    def __call__(
        self, 
        feature2activation : Feature2Float, 
        chunks : ChunkSet
    ) -> Chunk2Float:
        """Compute chunk strength given current bottom-level activations.

        For details, see Section 3.2.2.3 para 3 of Sun (2016).

        Args:
            feature2activation : Current activations at the botom level.
            chunks : A set of chunks for which bottom-up activations may be 
                computed.
        """

        if self.chunk in chunks:
            # For each dimension, pick the maximum available activation of the 
            # target chunk microfeatures in that dimension. 
            dim2activation = dict()
            for f in self.chunk.microfeatures:
                try:
                    if dim2activation[f.dim()] < feature2activation[f]:
                        dim2activation[f.dim()] = feature2activation[f]
                except KeyError:
                    # A KeyError may arise either because f.dim() is not in 
                    # dim2activation, or because f is not in feature2activation. 
                    # If the former is the case, add f.dim() to dim2activation, 
                    # otherwise move on to the next microfeature.
                    try: 
                        dim2activation[f.dim()] = feature2activation[f]
                    except KeyError:
                        continue
            # Compute chunk strength based on dimensional activations.
            strength = 0.
            for dim in dim2activation:
                strength += self.chunk.dim2weight[dim] * dim2activation[dim]
            else:
                # What is the purpose of superlinearity? 
                # Is it just to ensure that bottom-up activation is < 1?
                strength /= sum(self.chunk.dim2weight.values()) ** 1.1
            return {self.chunk: strength}
        else:
            return dict()

class Rule(Activation):
    """A basic Clarion associative rule.

    Rules have the form:
        chunk_1 chunk_2 chunk_3 -> chunk_4
    Chunks in the left-hand side are condition chunks, the single chunk in the 
    right-hand side is the conclusion chunk. The strength of the conclusion 
    chunk resulting from rule application is a weighted sum of the strengths 
    of the condition chunks.

    Only essential features of Clarion rules are represented in this class. 
    More advanced or specialized features may be added using specialized Mixin 
    classes (see, e.g., bla.BLAMixin).

    This implementation is based on Chapter 3 of Sun (2016). 

    References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(
        self,
        chunk2weight : Chunk2Float,
        conclusion_chunk : Chunk
    ) -> None:
        """Initialize a Clarion associative rule.

        kwargs:
            chunk2weight : A mapping from condition chunks to their weights.
            conclusion_chunk : The conclusion chunk of the rule.
        """
        
        self.chunk2weight = chunk2weight
        self.conclusion_chunk = conclusion_chunk
    
    def __call__(self, chunk2strength : Chunk2Float, chunks) -> Chunk2Float:
        """Return strength of conclusion chunk resulting from an application of 
        current associative rule.

        kwargs:
            chunk2strength : A mapping from chunks to their current strengths.
            chunks : A set of chunks for which rule activations may be 
                computed.
        """
        
        if self.conclusion_chunk in chunks:
            strength = 0. 
            for chunk in self.chunk2weight:
                try:
                    strength += chunk2strength[chunk] * self.chunk2weight[chunk]
                except KeyError:
                    continue
            return {self.conclusion_chunk, strength}
        else:
            return dict()

class Implicit(Activation):
    pass