from typing import TypeVar, Optional, List, Tuple, ClassVar

from ..numdicts import NumDict
from ..base.symbols import F
from ..base.constructs import Process, subprocesses


T = TypeVar("T")
class CAM(Process[T]):
    """Computes the combined-add-max activation for each node in a pool."""

    def initial(self) -> NumDict[T]:
        return NumDict()

    def call(self, *inputs: NumDict[T]) -> NumDict[T]:
        return NumDict.eltwise_cam(*inputs)


class WeightedCAM(Process[T]):
    pool: CAM[T]
    _prms: NumDict[F]
    _d_weight: ClassVar[str] = "weight"

    def __init__(
        self, 
        path: str = "", 
        inputs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(path, inputs)
        self.__validate()
        with subprocesses():
            self.pool = CAM()
        self._prms = NumDict({f: 1.0 for f in self._init_prms()})

    def __validate(self) -> None:
        if len(self.inputs) == 0:
            raise ValueError("Must have at least one input.") 

    def initial(self) -> Tuple[NumDict[T], NumDict[F]]:
        return self.pool.initial(), self._prms

    def call(
        self, p: NumDict[F], *inputs: NumDict[T]
    ) -> Tuple[NumDict[T], NumDict[F]]:
        weighted = (p.isolate(key=f).mul(d) for f, d in zip(self._prms, inputs))
        return self.pool.call(*weighted), self._prms

    def _init_prms(self) -> Tuple[F, ...]:
        return tuple(F(self._d_weight, m=i, p=self.path)    
            for i in self.inputs[1:])
