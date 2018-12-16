"""Generally useful non-basic pyClarion components."""

from pyClarion.base import *
import math
import random


class SimpleJunction(object):
    """Merges node strengths from multiple packets using pure dict update."""

    def __call__(self, packets):

        d = {}
        for packet in packets:
            d.update(packet.strengths)
        return d


class MaxJunction(object):
    """Returns max activation for each input node."""

    def __call__(self, packets):
        """
        Process packets.

        Assumes activations >= 0.

        :param packets: An iterable of activation packets.
        """

        d = {}
        for packet in packets:
            for n, s in packet.strengths.items():
                new_max = d.get(n, 0) < s
                if new_max:
                    d[n] = s
        return d


class SimpleNodeJunction(object):
    """Determines node output based on given recommendations."""

    def __init__(self, csym, default_strength):
        """
        Initialize a SimpleNodeJunction instance.

        :param csym: Client node.
        :param default_strength: Callable taking a single construct symbol. 
            Returns default strength of given construct.
        """

        self.csym = csym
        self.default_strength = default_strength

    def __call__(self, packets):
        """
        Output maximum recommended strength for client node.
        
        :param packets: An iterable of activation packets.
        """

        d = {}
        for s in self._iter_packets(packets):
            d[self.csym] = max(
                s, d.get(self.csym, self.default_strength(self.csym))
            )
        return d

    def _iter_packets(self, packets):

        for packet in packets:
            for n, s in packet.strengths.items():
                if n is self.csym:
                    yield s


class BoltzmannSelector(object):
    """Selects a chunk according to a Boltzmann distribution."""

    def __init__(self, temperature):
        """Initialize a ``BoltzmannSelector`` instance.

        :param temperature: Temperature of the Boltzmann distribution.
        """

        self.temperature = temperature

    def __call__(self, strengths):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.

        :param strengths: Mapping of node strengths.
        """

        choices = ()
        bd = self.get_boltzmann_distribution(strengths)
        chunks, probabilities = tuple(zip(*bd.items())) or ((), ())
        chosen = tuple(random.choices(chunks, weights=probabilities))
        return bd, chosen

    def get_boltzmann_distribution(self, strengths):
        """Construct and return a boltzmann distribution."""

        terms, divisor = {}, 0
        for ck, s in self._iter_chunk_strengths(strengths):
            terms[ck] = math.exp(s / self.temperature)
            divisor += terms[ck]
        probabilities = {ck: s / divisor for ck, s in terms.items()}
        return probabilities

    @staticmethod
    def _iter_chunk_strengths(strengths):

        for csym, s in strengths.items():
            if csym.ctype is ConstructType.Chunk:
                yield (csym, s)            


class CategoricalSelector(object):
    """
    Selects a chunk according to a categorical distribution.
    
    May be interpreted as a Boltzmann distribution applied to the log of chunk 
    strengths.
    """

    def __init__(self, temperature = 1.):
        """Initialize a ``CategoricalSelector`` instance.

        :param temperature: Temperature of the categorical distribution. Behaves 
            similarly to temperature parameter in boltzmann distribution. 
        """

        self.temperature = temperature

    def __call__(self, strengths):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        categorical distribution:
            p(ch) = strength(ch) ** - temp / (sum_i strength(i) ** - temp)

        :param strengths: Mapping of node strengths.
        """

        choices = ()
        bd = self.get_categorical_distribution(strengths)
        chunks, probabilities = tuple(zip(*bd.items())) or ((), ())
        chosen = tuple(random.choices(chunks, weights=probabilities))
        return bd, chosen

    def get_categorical_distribution(self, strengths):
        """Construct and return a chunk selection distribution."""

        terms, divisor = {}, 0
        for ck, s in self._iter_chunk_strengths(strengths):
            terms[ck] = pow(s, -self.temperature)
            divisor += terms[ck]
        probabilities = {ck: s / divisor for ck, s in terms.items()}
        return probabilities

    @staticmethod
    def _iter_chunk_strengths(strengths):

        for csym, s in strengths.items():
            if csym.ctype is ConstructType.Chunk:
                yield (csym, s)            


class MappingEffector(object):
    """Links actionable chunks to callbacks."""

    def __init__(self, chunk2callback) -> None:
        """
        Initialize a MappingEffector.

        :param chunk2callback: Mapping from actionable chunks to callbacks.
        """

        self.chunk2callback = chunk2callback

    def __call__(self, dpacket) -> None:
        """
        Execute callbacks associated with each chosen chunk.

        :param dpacket: A decision packet.
        """
        
        for chunk in dpacket.chosen:
            self.chunk2callback[chunk]()


class ConstantSource(object):
    """Outputs a stored activation packet."""

    def __init__(self, strengths = None) -> None:

        self.strengths = strengths or dict()

    def __call__(self):
        """Return stored strengths."""

        return self.strengths

    def update(self, strengths):
        """Update self with contents of dict-like strengths."""

        self.strengths = self.strengths.copy()
        self.strengths.update(strengths)

    def clear(self) -> None:
        """Clear stored node strengths."""

        self.strengths = {}
