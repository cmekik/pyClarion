from typing import Callable, ClassVar

from ..numdicts.ops.funcs import collect
from ..numdicts.ops.base import OpBase, GradientTape
from ..numdicts.ops.base import Unary,  Aggregator
from ..numdicts import KeyForm, NumDict


class Cost(OpBase[NumDict]):
    kernel: ClassVar[Callable[[float, float, float], float]]

    def __call__(self, est: NumDict, tgt: NumDict, msk: NumDict, /, by: KeyForm | tuple[KeyForm, KeyForm] | None = None) -> NumDict:
        if isinstance(by, KeyForm):
            by = (by, by)
        it = collect(est, tgt, msk, mode="match", branches=by)
        if (not isinstance(est._c, float) 
            or not isinstance(tgt._c, float) 
            or not isinstance(msk._c, float)):
            raise ValueError()
        new_c = type(self).kernel(est._c, tgt._c, msk._c)
        new_d = {k: v for k, (v1, v2, v3) in it 
            if (v := type(self).kernel(v1, v2, v3)) != new_c}
        r = type(est)(est._i, new_d, new_c, False)
        tape = GradientTape.STACK.get()
        if tape is not None:
            tape.record(self, r, est, tgt, msk, by=by)
        return r

    def grad(self, g: NumDict, r: NumDict, est: NumDict, tgt: NumDict, msk: NumDict, /, by: KeyForm | tuple[KeyForm, KeyForm] | None = None) -> tuple[NumDict, NumDict, NumDict]:
        raise NotImplementedError()


class CAM(Aggregator[NumDict]):
    kernel = lambda xs: max(0.0, *xs) + min(0.0, *xs)
    eye = 0.0


class LeastSquaresCost(Cost):
    kernel = lambda est, tgt, msk: msk * (est - tgt) ** 2
    def grad(self, g: NumDict, r: NumDict, est: NumDict, tgt: NumDict, msk: NumDict, /, by: KeyForm | tuple[KeyForm, KeyForm] | None = None) -> tuple[NumDict, NumDict, NumDict]:
        if by is None or isinstance(by, KeyForm):
            _by = (by, by)
        else:
            _by = by
        return est.sub(tgt, by=_by[0]).mul(msk).mul(g), tgt.zeros(), msk.zeros()

cam = CAM()
least_squares_cost = LeastSquaresCost()