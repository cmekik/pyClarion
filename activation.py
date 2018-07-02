import abc
import typing as T
from chunk import Chunk, ChunkSet, Chunk2Float
from feature import Feature, Dim2Float, FeatureSet

Node = T.Union[Chunk, Feature]
NodeSet = T.Set[Node]
Node2Float = T.Mapping[Node, float]

class ActivationChannel(abc.ABC):
    """An abstract class for capturing activation flows.
    """
    
    def __call__(
        self, 
        input_map : Node2Float
    ) -> Node2Float:
        """Compute and return activations resulting from an input to this 
        channel.

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
    def __call__(
        self, 
        *input_maps: Node2Float
    ) -> Node2Float:
        """Return a combined mapping from chunks and/or (micro)features to 
        activations.

        kwargs:
            activation_maps : A set of mappings from chunks and/or 
            (micro)features to activations.
        """

        pass

class ActivationFilter(ActivationChannel):
    """Filters activation maps against a set of nodes.
    """

    def __init__(self, nodes : NodeSet):
        """Initialize an activation filter.

        kwargs:
            nodes : A set of nodes whose activations may pass the filter.
        """

        self.nodes = nodes

    def __call__(self, input_map : Node2Float) -> Node2Float:
        """Return a filtered activation map.

        kwargs:
            input_map : An activation map to be filtered.
        """

        return {k:v for (k,v) in input_map if k in self.nodes}

class TopDown(ActivationChannel):
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
        input_map : Node2Float
    ) -> Node2Float:
        """Compute bottom-level activations given current chunk strength.

        For details, see Section 3.2.2.3 para 2 of Sun (2016).

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

class BottomUp(ActivationChannel):
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
        input_map : Node2Float, 
    ) -> Node2Float:
        """Compute chunk strength given current bottom-level activations.

        For details, see Section 3.2.2.3 para 3 of Sun (2016).

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
                # dim2activation, or because f is not in feature2activation. 
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
            # What is the purpose of superlinearity? 
            # Is it just to ensure that bottom-up activation is < 1?
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
    
    def __call__(self, input_map : Node2Float) -> Node2Float:
        """Return strength of conclusion chunk resulting from an application of 
        current associative rule.

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
    pass