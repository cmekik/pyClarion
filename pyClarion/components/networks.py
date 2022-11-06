from typing import Sequence, Tuple, Callable, TypeVar

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

class NDRAM(cld.Process):

    initial = NumDict()

    lr: float
    t: int
    delta: float
    xi: float
    w: NumDict[Tuple[feature, feature]]

    def __init__(
        self, 
        lr: float = 1e-3, 
        t: int = 1, 
        delta: float = .2, 
        xi: float = 1.
    ) -> None:
        self.lr = lr
        self.t = t
        self.delta = delta
        self.xi = xi
        self.w = NumDict()

    def call(
        self, c: NumDict[feature], x: NumDict[feature]
    ) -> NumDict[feature]:
        output = self.forward_pass(x)
        self.update(c, x)
        return output

    def update(self, c: NumDict[feature], x: NumDict[feature]) -> None:
        apply = c[self.cmds[1]] # controls whether to update weights
        y = self.forward_pass(x, iter=self.t)
        xx = x.outer(x)
        yy = y.outer(y)
        self.w = (self.w
            .mul(self.xi)
            .add(apply * self.lr * (xx - yy)))
        
    def forward_pass(self, x: NumDict[feature], iter=1) -> NumDict[feature]:
        y = x
        for _ in range(iter):
            y = self.activation(self.w
                .mul_from(y, kf=cld.first)
                .sum_by(kf=cld.second))
        return y

    def activation(self, x: NumDict[T]) -> NumDict[T]:
        return (x
            .mul(1 + self.delta)
            .sub(x.pow(3).mul(self.delta))
            .max(-1).min(1)) # clip output to be between [-1, 1]

    @property
    def cmds(self) -> Tuple[feature, ...]:
        return (
            feature(cld.prefix("ud", self.prefix), None), 
            feature(cld.prefix("ud", self.prefix), "apply"))

    @property
    def nops(self) -> Tuple[feature, ...]:
        return (feature(cld.prefix("ud", self.prefix), None),)