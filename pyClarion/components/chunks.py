from typing import Tuple, Callable

from ..numdicts import NumDict
from ..base.constructs import Process
from ..base.symbols import F, D, C
from .. import sym


class ChunkStore(Process):
    """Basic store for chunks."""
    cf: NumDict[Tuple[C, F]] # (C, F): 1.0, c=0
    cd: NumDict[Tuple[C, D]] # (C, D): w, c=0
    cn: NumDict[C]           # C: sum(|w|), c=0

    def __init__(
        self, 
        path: str = "", 
        g: Callable[[NumDict[C]], NumDict[C]] = sym.eye,
    ) -> None:
        super().__init__(path)
        self.g = g
        self.cf = NumDict() 
        self.cd = NumDict()
        self.cn = NumDict()

    def initial(self) \
    -> Tuple[NumDict[Tuple[C, F]], NumDict[Tuple[C, D]], NumDict[C]]:
        return self.cf, self.cd, self.g(self.cn).set_c(0)

    call = initial

    def add(
        self, cf: NumDict[Tuple[C, F]], cd: NumDict[Tuple[C, D]]
    ) -> None:
        self.cf = self.cf.merge(cf)
        self.cd = self.cd.merge(cd)
        self.cn = self.cn.merge(cd.abs().sum_by(kf=sym.first))


class BottomUp(Process):
    """Propagates bottom-up activations."""

    def initial(self) -> NumDict[C]:
        return NumDict()

    def call(
        self, 
        cf: NumDict[Tuple[C, F]], 
        cd: NumDict[Tuple[C, D]], 
        cn: NumDict[C], 
        d: NumDict[F]
    ) -> NumDict[C]:
        """
        Propagate bottom-up activations.
        
        :param cf: Chunk-feature associations (binary).
        :param cd: Chunk-dimension associations (i.e., top-down weights).
        :param cn: Normalization terms for each chunk. For each chunk, expected 
            to be equal to g(sum(|w|)), where g is some superlinear function.
        :param d: Feature strengths in the bottom level.
        """
        return (cf
            .put(d, kf=sym.second, strict=True)
            .cam_by(kf=sym.cf2cd)
            .mul_from(cd, kf=sym.eye)
            .sum_by(kf=sym.first)
            .squeeze()
            .div_from(cn, kf=sym.eye, strict=True))


class TopDown(Process):
    """Propagates top-down activations."""

    def initial(self) -> NumDict[F]:
        return NumDict()

    def call(
        self, 
        cf: NumDict[Tuple[C, F]], 
        cd: NumDict[Tuple[C, D]], 
        d: NumDict[C]
    ) -> NumDict[F]:
        """
        Propagate top-down activations.
        
        :param cf: Chunk-feature associations (binary).
        :param cd: Chunk-dimension associations (i.e., top-down weights).
        :param d: Chunk strengths in the top level.
        """
        return (cf
            .mul_from(d, kf=sym.first, strict=True)
            .mul_from(cd, kf=sym.cf2cd, strict=True)
            .cam_by(kf=sym.second) 
            .squeeze())
