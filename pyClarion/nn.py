from typing import TypeVar, Tuple, List, Callable, Sequence
from typing_extensions import Protocol

from .numdicts import NumDict, GradientTape
from .sym import second


class NeuralNet:
    params: List[NumDict]
    spec: List[Tuple[Callable, Sequence[str]]]

    def __init__(self, spec: List[Tuple[Callable, Sequence[str]]]) -> None:
        self.params = []
        self.spec = []
        self.add(*spec)

    def add(self, *args: Tuple[Callable, Sequence[str]]):
        for f, seq in args:
            for arg in seq:
                if arg[0] == "l" and len(self.spec) <= int(arg[1:]):
                    raise ValueError(f"Input '{arg}' undefined")
                if arg[0] == "p" and len(self.params) < int(arg[1:]):
                    raise ValueError(f"Param '{arg}' undefined")
                elif arg[0] == "p" and len(self.params) == int(arg[1:]):
                    self.params.append(NumDict())
            self.spec.append((f, seq))

    def call(self, *inputs: NumDict) -> List[NumDict]:
        i, l = inputs, []
        for f, seq in self.spec:
            l.append(f(*self._get_args(seq, i, l, self.params)))
        return l
        
    def _get_args(self, seq, i, l, p):
        d, args = {"i": i, "l": l, "p": p}, []
        for symbol in seq:
            args.append(d[symbol[0]][int(symbol[1:])])
        return args


class Optimizer(Protocol):
    def update(self, net: NeuralNet, *inputs: NumDict) -> List[NumDict]:
        ...


class SGD:
    lr: float
    eta: float
    gamma: float
    _t: int

    def __init__(
        self, lr: float = 1e-1, eta: float = 1, gamma: float = .55
    ) -> None:
        self.lr = lr
        self.eta = eta
        self.gamma = gamma
        self.__validate()
        self._t = 0

    def update(self, net: NeuralNet, *inputs: NumDict) -> List[NumDict]:
        with GradientTape() as tape:
            outputs = net.call(*inputs)
        params = tuple(net.params)
        _, grads = tape.gradients(outputs[-1], params)
        self._t += 1
        sigma = self.eta / self._t ** self.gamma
        for i, (param, grad) in enumerate(zip(params, grads)):
            net.params[i] = param.sub(
                grad.add_normal_noise(sigma=sigma).mul(self.lr))
        return outputs

    def __validate(self) -> None:
        if self.lr <= 0:
            raise ValueError("Learning rate must be greater than zero.")
        if not 0 <= self.eta:
            raise ValueError("Eta must be greater than zero.")
        if not 0 <= self.gamma:
            raise ValueError("Gamma must be greater than zero.")


class ADAM:
    lr: float
    beta1: float
    beta2: float
    eta: float
    gamma: float
    m_t: List[NumDict]
    v_t: List[NumDict]
    _t: int
    _epsilon: float = 1e-8

    def __init__(
        self, 
        lr: float = 1e-1, 
        beta1: float = .9, 
        beta2: float = .999, 
        eta: float = 1, 
        gamma: float = .55
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.gamma = gamma
        self.__validate()
        self.m_t = []
        self.v_t = []
        self._t = 0

    def update(self, net: NeuralNet, *inputs: NumDict) -> List[NumDict]:
        with GradientTape() as tape:
            outputs = net.call(*inputs)
        params = tuple(net.params)
        _, grads = tape.gradients(outputs[-1], params)
        if self._t == 0:
            for param in params:
                self.m_t.append(NumDict())
                self.v_t.append(NumDict())
        self._t += 1
        sigma = self.eta / self._t ** self.gamma
        beta1, beta2 = self.beta1, self.beta2
        beta1_compl, beta2_compl = 1 - beta1, 1 - beta2
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m_t[i] = self.m_t[i] * beta1 + grad * beta1_compl
            self.v_t[i] = self.v_t[i] * beta2 + grad * grad * beta2_compl
            m_t_hat = self.m_t[i] / (1 - beta1 ** self._t)
            v_t_hat = self.v_t[i] / (1 - beta2 ** self._t)
            net.params[i] = param.sub(
                m_t_hat.div(v_t_hat.pow(0.5).add(self._epsilon))
                .add_normal_noise(sigma=sigma)
                .mul(self.lr))
        return outputs

    def __validate(self) -> None:
        if self.lr <= 0:
            raise ValueError("Learning rate must be greater than zero.")
        if not 0 <= self.beta1 < 1:
            raise ValueError("Beta1 must be in [0, 1).")
        if not 0 <= self.beta2 < 1:
            raise ValueError("Beta2 must be in [0, 1).")
        if not 0 <= self.eta:
            raise ValueError("Eta must be greater than zero.")
        if not 0 <= self.gamma:
            raise ValueError("Gamma must be greater than zero.")


T1, T2 = TypeVar("T1"), TypeVar("T2")
def layer(
    x: NumDict[T1], 
    l: NumDict[T2], 
    w: NumDict[Tuple[T1, T2]], 
    b: NumDict[T2]
) -> NumDict[T2]:
    return x.outer(l).mul(w).sum_by(kf=second).add(b)


def sum_squares(err: NumDict) -> NumDict:
    return err.mul(err).div(2).reduce_sum()


def sequential(n_hl: int, f_h: Callable) -> NeuralNet:
    net = NeuralNet([])
    net.add((layer, ("i0", "i1", "p0", "p1")))
    for i in range(0, n_hl + 1):
        net.add((f_h, (f"l{2*i}",)))
        net.add(
            (layer, (f"l{2*i + 1}", f"i{i+2}", f"p{2*i + 2}", f"p{2*i + 3}")))
    return net
