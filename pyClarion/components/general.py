from pyClarion.base.symbols import *
from pyClarion.base.packets import *
from pyClarion.components.utils import *


class MaxNode(object):
    """Simple node returning maximum strength for given construct."""

    def __call__(self, construct, inputs):

        packets = (pull_func() for pull_func in inputs.values())
        strengths = max_strength(construct, packets)
        return ActivationPacket(strengths=strengths)


class BoltzmannSelector(object):
    """Selects a chunk according to a Boltzmann distribution."""

    def __init__(self, temperature, k=1):
        """Initialize a ``BoltzmannSelector`` instance.

        :param temperature: Temperature of the Boltzmann distribution.
        """

        self.temperature = temperature
        self.k = k

    def __call__(self, construct, inputs):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.

        :param strengths: Mapping of node strengths.
        """

        packets = (pull_func() for pull_func in inputs.values())
        strengths = simple_junction(packets)
        probabilities = boltzmann_distribution(strengths, self.temperature)
        selection = select(probabilities, self.k)
        dpacket = DecisionPacket(strengths=probabilities, selection=selection)
        return dpacket


class MappingEffector(object):
    """Links actionable chunks to callbacks."""

    def __init__(self, callbacks) -> None:
        """
        Initialize a SimpleEffector instance.

        :param chunk2callback: Mapping from actionable chunks to callbacks.
        """

        self.callbacks = callbacks

    def __call__(self, dpacket) -> None:
        """
        Execute callbacks associated with each chosen chunk.

        :param dpacket: A decision packet.
        """
        
        for chunk in dpacket.selection:
            self.callbacks[chunk].__call__()


class ConstantSource(object):
    """Outputs a stored activation packet."""

    def __init__(self, strengths = None) -> None:

        self.strengths = strengths or dict()

    def __call__(self, construct, inputs):
        """Return stored strengths."""

        return ActivationPacket(strengths=self.strengths)

    def update(self, strengths):
        """Update self with contents of dict-like strengths."""

        self.strengths = self.strengths.copy()
        self.strengths.update(strengths)

    def clear(self) -> None:
        """Clear stored node strengths."""

        self.strengths = {}
