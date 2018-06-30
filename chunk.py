import typing as T
from feature import Dim2Float, Feature2Float, FeatureSet

class Chunk(object):
    """A Clarion chunk. 

    This implementation is based on Chapter 3 of Sun (2016).

    References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(
        self, 
        microfeatures: FeatureSet,
        top_down_weights: T.Optional[Dim2Float] = None,
        label: T.Optional[str] = None
    ) -> None:
        """Initialize a Clarion Chunk.

        Args:
            microfeatures : Set of dv-pairs.
            top_down_weights : A dict mapping each dimension to its top-down 
                weight.
            label : Semantic label of chunk.
        """
        
        self.microfeatures = microfeatures
        self.top_down_weights = self.initialize_weights(top_down_weights)
        self._val_counts = self.count_values(microfeatures)
        if label is not None:
            self.label = label

    def top_down(self, strength: float) -> Feature2Float:
        """Compute bottom-level activations given current chunk strength.

        For details, see Section 3.2.2.3 para 2 of Sun (2016).

        Args:
            strength : Current chunk strength.
        """
 
        activations = dict()
        for f in self.microfeatures:
            w_dim = self.top_down_weights[f.dim()] 
            n_val = self._val_counts[f.dim()]
            activations[f] = strength * w_dim / n_val
        return activations

    def bottom_up(self, activations: Feature2Float) -> float:
        """Compute chunk strength given current bottom-level activations.

        For details, see Section 3.2.2.3 para 3 of Sun (2016).

        Args:
            activations : Current activations at the botom level.
        """

        dim_activations = dict()
        for f in self.microfeatures:
            try:
                if dim_activations[f.dim()] < activations[f]:
                    dim_activations[f.dim()] = activations[f]
            except KeyError:
                dim_activations[f.dim()] = activations[f]
        
        strength = 0.
        for dim in dim_activations:
            strength += self.top_down_weights[dim] * dim_activations[dim]
        else:
            # What is the purpose of superlinearity? 
            # Is it just to ensure that bottom-up activation is < 1?
            strength /= sum(self.top_down_weights.values()) ** 1.1

        return strength

    def initialize_weights(
        self, 
        top_down_weights: T.Optional[Dim2Float]
    ) -> Dim2Float:
        """Initialize top-down weights.

        If input is None, weights are initialized to 1.0.

        Args:
            top_down_weights: A dict mapping each dimension to its top-down 
                weight.
        """
        
        if top_down_weights is None:
            top_down_weights = dict()
            for f in self.microfeatures:
                top_down_weights[f.dim()] = 1.
        return top_down_weights

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