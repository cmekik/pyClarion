from typing import List, Tuple, Deque, Optional
from collections import deque

from ..numdicts import NumDict
from ..base.constructs import Process
from ..base.symbols import F, D, V
from ..nn import Optimizer, NeuralNet, ADAM, sequential, sum_squares


class SQNet(Process):
    layers: List[int]
    gamma: float
    optimizer: Optimizer
    n: List[NumDict]
    net: NeuralNet
    _xio: Deque[Tuple[NumDict[F], NumDict[F], NumDict[V]]]
    _ar: Deque[Tuple[NumDict[V], NumDict[D]]]

    def __init__(
        self, 
        path: str = "", 
        inputs: Optional[List[str]] = None, 
        layers: Optional[List[int]] = None, 
        gamma: float = 0.7,
        optimizer: Optional[Optimizer] = None
    ) -> None:
        super().__init__(path, inputs)
        self.layers = layers or []
        self.gamma = gamma
        self.optimizer = optimizer or ADAM()
        self.__validate()
        self.n = [NumDict({i: 1.0 for i in range(n)}) for n in self.layers]
        self.net = self._init_net()
        self._xio = deque([(NumDict(),) * 3] * 2, maxlen=2)
        self._ar = deque([(NumDict(),) * 2], maxlen=1)

    def __validate(self) -> None:
        if not 0 <= self.gamma or not self.gamma <= 1:
            raise ValueError("Discount factor must be in interval [0, 1].")

    def initial(self) -> NumDict[V]:
        return NumDict()
    
    def call(
        self, 
        x_curr: NumDict[F], 
        i_curr: NumDict[F], 
        o_curr: NumDict[V],
        a_prev: NumDict[V], 
        r_prev: NumDict[D],
    ) -> NumDict[V]:
        self._xio.append((x_curr, i_curr, o_curr))
        self._ar.append((a_prev, r_prev))
        output = self.net.call(
            *self._collect_inputs(*self._xio[1], NumDict(), NumDict()))
        q, (a, r) = output[-3], self._ar[0]
        y = r + self.gamma * q.max_by(kf=F.dim.fget)
        self.optimizer.update(self.net, 
            *self._collect_inputs(*self._xio[0], a, y))
        return q.tanh()

    def _collect_inputs(self, x, i, o, a, y):
        return [x, i, *self.n, o, a, y]

    def _init_net(self):
        net = sequential(len(self.layers), NumDict.tanh)
        l = len(net.spec) - 1
        i = len(self.layers) + 3
        net.add((self._td_err, (f"l{l}", f"i{i}", f"i{i + 1}")), 
            (sum_squares, (f"l{l+1}",)))
        return net

    @staticmethod
    def _td_err(q, a, y):
        y_hat = q.mul(a).sum_by(kf=F.dim.fget)
        return y.sub(y_hat)
