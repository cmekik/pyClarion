from ...numdicts import NumDict
from .base import Layer
from .base import Activation


class Tanh(Activation):
    """A tanh activation function for neural networks."""

    def __call__(self, d: NumDict) -> NumDict:
        return d.tanh()

    def grad(self, d: NumDict) -> NumDict:
        return d.cosh().inv().pow(x=2.0)

    def scale(self, layer: Layer) -> float:
        return 1 / (1 + len(layer.input) + len(layer.main))
