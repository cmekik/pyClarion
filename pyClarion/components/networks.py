from typing import Tuple, Callable, TypeVar

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


T = TypeVar("T")

class NDRAM(NAM):

    p: int
    lr: float
    delta: float
    xi: float

    def __init__(self, 
        p: int = 10, 
        lr: float = 1e-3, 
        delta: float = .49, 
        xi: float = 9999e-4
    ) -> None:
        self.p = p
        self.lr = lr
        self.delta = delta
        self.xi = xi
        super().__init__(f=self.activation)

    def call(self, x: NumDict[feature]) -> NumDict[feature]:
        output = super().call(x)
        self.update(x)
        return output

    def update(self, x: NumDict[feature]) -> None:
        y = x
        for _ in range(self.p):
            y = super().call(y)
        xx = x.outer(x)
        yy = y.outer(y)
        self.w = (self.w
            .mul(self.xi)
            .sub(self.lr * (xx - yy)))
        
    def activation(self, x: NumDict[T]) -> NumDict[T]:
        return (x
            .mul(1 + self.delta)
            .sub(x.pow(3).mul(self.delta))
            .max(-1)
            .min(1))