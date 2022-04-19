from dataclasses import dataclass
from typing import List, Tuple, Callable

from .. import dev as cld
from ..base import feature
from ..numdicts import NumDict


class NAM(cld.Process):
    """
    A neural associative memory.
    
    Implements a single fully connected layer. 
    
    For validation, each weight and bias key must belong to a client fspace.

    May be used as a static network or as a base for various associative 
    learning models such as Hopfield nets.
    """

    initial = NumDict()

    w: NumDict[Tuple[feature, feature]]
    b: NumDict[feature]

    def __init__(
        self,
        f: Callable[[NumDict[feature]], NumDict[feature]] = cld.eye
    ) -> None:
        self.w = NumDict()
        self.b = NumDict()
        self.f = f

    def validate(self):
        if self.fspaces:
            fspace = set(f for fspace in self.fspaces for f in fspace())
            if (any(k1 not in fspace or k2 not in fspace for k1, k2 in self.w) 
                or any(k not in fspace for k in self.b)):
                raise ValueError("Parameter key not a member of set fspaces.")

    def call(self, x: NumDict[feature]) -> NumDict[feature]:
        return (self.w
            .mul_from(x, kf=cld.first)
            .sum_by(kf=cld.second)
            .add(self.b)
            .pipe(self.f))
