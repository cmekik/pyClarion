'''Common constructs for the standard implementation of Clarion.'''

from pyClarion.base.symbols import Node
from pyClarion.base.processors import UpdateJunction, MaxJunction


def default_activation(key: Node = None) -> float:
    
    return 0.0


class StandardUpdateJunction(UpdateJunction[float]):

    def __init__(self, default_activation=default_activation):

        super().__init__(default_activation)


class StandardMaxJunction(MaxJunction[float]):

    def __init__(self, default_activation=default_activation):

        super().__init__(default_activation)