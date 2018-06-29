import enum
import typing as T

@enum.unique
class Feature(enum.Enum):
    """An abstract Clarion (micro)feature.

    In Clarion, (micro)features are represented as dimension-value pairs:
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

# Some useful types
# Note: Type hints are mainly for ease of reading, code has not been tested 
# with a type checker yet.
Dim2Float = T.Dict[enum.EnumMeta, float]
FeatureSet = T.Set[Feature]
Feature2Float = T.Dict[Feature, float]

class Chunk(object):
    """A Clarion chunk. 

    This implementation is based on Chapter 3 of Sun (2016).

    References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(self, 
                 microfeatures: FeatureSet,
                 top_down_weights: Dim2Float,
                 label: str="") -> None:
        """Initialize a Clarion Chunk.

        Args:
            label : Semantic label of chunk.
            microfeatures : Set of dv-pairs.
        """
        
        self.label = label
        self.microfeatures = microfeatures
        self.top_down_weights = top_down_weights
        self._val_counts = self.count_values(microfeatures)

    def top_down(self, strength: float) -> Feature2Float:
        """Compute bottom-level activations given current chunk strength.

        For details, see Section 3.2.2.3 para 2 of Sun (2016).

        Args:
            strength : Current chunk strength.
        """
 
        activations = dict()
        for f in self.microfeatures:
            dim = type(f)
            w_dim, n_val = self.top_down_weights[dim], self._val_counts[dim]
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
            dim = type(f)
            try:
                if dim_activations[dim] < activations[f]:
                    dim_activations[dim] = activations[f]
            except KeyError:
                dim_activations[dim] = activations[f]
        
        strength = 0.
        for dim in dim_activations:
            strength += self.top_down_weights[dim] * dim_activations[dim]
        else:
            # What is the purpose of superlinearity? 
            # Is it just to ensure that bottom-up activation is < 1?
            strength /= sum(self.top_down_weights.values()) ** 1.1

        return strength

    @staticmethod
    def count_values(microfeatures: FeatureSet) -> Dim2Float:
        """Count the number of features in each dimension
        
        Args:
            microfeatures : A set of dv-pairs.
        """
        
        counts = dict()
        for f in microfeatures:
            try:
                counts[type(f)] += 1
            except KeyError:
                counts[type(f)] = 1
        return counts