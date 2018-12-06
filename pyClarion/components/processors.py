import typing as typ
from pyClarion.base import *
import numpy as np


class UpdateJunction(object):
    """Merges input activation packets using the packet ``update`` method."""

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


class NodeMaxJunction(object):

    def __init__(self, csym, default_strength):

        self.csym = csym
        self.default_strength = default_strength

    def __call__(self, packets):

        d = {}
        for s in self.iter_packets(packets):
            d[self.csym] = max(
                s, d.get(self.csym, self.default_strength(self.csym))
            )
        return d

    def iter_packets(self, packets):

        for packet in packets:
            for n, s in packet.strengths.items():
                if n is self.csym:
                    yield s


class BoltzmannSelector(object):
    """Select a chunk according to a Boltzmann distribution."""

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
        chunk_list, probabilities = tuple(zip(*bd.items())) or ((), ())
        chosen = self.choose(chunk_list, probabilities)
        return bd, chosen

    def get_boltzmann_distribution(self, strengths):
        """Construct and return a boltzmann distribution."""

        terms, divisor = {}, 0
        for ck, s in self.iter_chunk_strengths(strengths):
            terms[ck] = np.exp(s / self.temperature)
            divisor += terms[ck]
        probabilities = {ck: s / divisor for ck, s in terms.items()}
        return probabilities

    def choose(self, chunks, probabilities):
        '''Choose a chunk given some selection probabilities.'''

        # Numpy complains that chunks is not 1-dimensional since it is nested 
        # tuples. So use range(len()) instead.
        choice = np.random.choice(range(len(chunks)), p=probabilities)
        return (chunks[choice],)

    @staticmethod
    def iter_chunk_strengths(strengths):

        for csym, s in strengths.items():
            if csym.ctype is ConstructType.Chunk:
                yield (csym, s)            


class MappingEffector(object):
    '''A simple effector built on a map from actionable chunks to callbacks.'''

    def __init__(self, chunk2callback) -> None:
        '''Initialize a ``MappingEffector`` instance.

        :param chunk2callback: Mapping from actionable chunks to callbacks.
        '''

        self.chunk2callback = chunk2callback

    def __call__(self, dpacket) -> None:
        '''
        Execute callbacks associated with each chosen chunk.

        :param dpacket: A decision packet.
        '''
        
        for chunk in dpacket.chosen:
            self.chunk2callback[chunk]()


class ConstantSource(object):
    """Simple source outputting a constant activation packet."""

    def __init__(self, strengths = None) -> None:

        self.strengths = strengths or dict()

    def __call__(self):

        return self.strengths

    def update(self, strengths):

        self.strengths.update(strengths)

    def clear(self) -> None:

        self.strengths.clear()
