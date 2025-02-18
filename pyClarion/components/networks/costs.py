from ...numdicts import NumDict
from .base import Cost


class LeastSquares(Cost):

    def __call__(self, est: NumDict, tgt: NumDict, mask: NumDict) -> NumDict:
        return est.sub(tgt).pow(x=2).mul(mask)
    
    def grad(self, est: NumDict, tgt: NumDict, mask: NumDict) -> NumDict:
        return est.sub(tgt).mul(mask)
