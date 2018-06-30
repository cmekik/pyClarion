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

    def dim(self) -> enum.EnumMeta:
        """Return dimension associated with self.
        """

        return type(self)

# Some useful types
Dim2Float = T.Dict[enum.EnumMeta, float]
FeatureSet = T.Set[Feature]
Feature2Float = T.Dict[Feature, float]