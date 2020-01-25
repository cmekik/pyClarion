from pyClarion.base import ActivationPacket, DecisionPacket
from pyClarion.utils.funcs import (
    max_strength, simple_junction, boltzmann_distribution, select, 
    multiplicative_filter
)
from pyClarion.base.realizers import Proc


__all__ = [
    "MaxNode", "BoltzmannSelector", "ConstantProc", "StimulusProc", 
    "FilteredProc"
]


class MaxNode(Proc):
    """Simple node returning maximum strength for given construct."""

    def call(self, construct, inputs, **kwargs):

        packets = inputs.values()
        strength = max_strength(construct, packets)
        return ActivationPacket(strengths=strength)


class BoltzmannSelector(Proc):
    """Selects a chunk according to a Boltzmann distribution."""

    def __init__(self, temperature, k=1):
        """
        Initialize a ``BoltzmannSelector`` instance.

        :param temperature: Temperature of the Boltzmann distribution.
        """

        self.temperature = temperature
        self.k = k

    def call(self, construct, inputs, **kwargs):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.

        :param strengths: Mapping of node strengths.
        """

        packets = inputs.values()
        strengths = simple_junction(packets)
        probabilities = boltzmann_distribution(strengths, self.temperature)
        selection = select(probabilities, self.k)
        dpacket = DecisionPacket(strengths=probabilities, selection=selection)
        return dpacket


class ConstantProc(Proc):
    """Outputs a stored activation packet."""

    def __init__(self, strengths = None) -> None:

        self.strengths = strengths or dict()

    def call(self, construct, inputs, **kwargs):
        """Return stored strengths."""

        return ActivationPacket(strengths=self.strengths)

    def update(self, strengths):
        """Update self with contents of dict-like strengths."""

        self.strengths = self.strengths.copy()
        self.strengths.update(strengths)

    def clear(self) -> None:
        """Clear stored node strengths."""

        self.strengths = {}


class StimulusProc(Proc):
    """Propagates externally provided stimulus."""

    def call(self, construct, inputs, stimulus=None, **kwargs):

        packet = stimulus or ActivationPacket()
        return packet


class FilteredProc(Proc):
    """
    Filters input and output activations of proc.

    Filters activation packets only. Will throw exception if other packet types 
    are encountered.

    Typically, filters will be applied only to chunk/feature nodes and 
    response realizers.
    """
    
    # Dev note:
    # This object should inherit type signature of base_proc, not sure how to 
    # do this. -CSM

    def __init__(
        self, base_proc, input_filter=None, output_filter=None, fdefault=0
    ):

        self.base_proc = base_proc
        # Expected types for input_filter and output_filter should be same as 
        # `matches` argument to realizers.
        self.input_filter = input_filter
        self.output_filter = output_filter
        self.fdefault = fdefault

    def call(self, construct, inputs, **kwargs):

        # Get filter settings and remove filter info from inputs dict so they 
        # are not processed by self.base_proc
        # Technically filters should be activation packets, but this is not 
        # enforced (for now at least).
        if self.input_filter is not None:
            input_weights = inputs.pop(self.input_filter)
        if self.output_filter is not None:
            output_weights = inputs.pop(self.output_filter)

        # Filter inputs to base_proc
        if self.input_filter is not None:
            # Make sure inputs has type Dict[ConstructSymbol, ActivationPacket]
            # This is an expensive test (O(n) in number of packets).
            if not all(
                [
                    isinstance(packet, ActivationPacket) 
                    for packet in inputs.values()
                ]
            ):
                raise TypeError(
                    "Input filtering must act on ActivationPackets."
                )
            inputs = {
                source: ActivationPacket(
                    strengths=multiplicative_filter(
                        filter_weights=input_weights, 
                        strengths=packet, 
                        fdefault=self.fdefault
                    )
                ) for source, packet in inputs.items()
            }
        
        # Call base_proc on (potentially) filtered inputs
        # Note that call is to `Proc.call()` instead of `Proc.__call__()`. This 
        # is because we rely on `self.__call__()` instead.
        output = self.base_proc.call(construct, inputs, **kwargs)

        # Filter outputs of base_proc
        if self.output_filter is not None:
            if not isinstance(output, ActivationPacket):
                raise TypeError(
                "Output filtering must act on an ActivationPacket."
            )
            output = ActivationPacket(
                strengths=multiplicative_filter(
                    filter_weights=output_weights, 
                    strengths=output.strengths, 
                    fdefault=self.fdefault
                )
            )

        return output