__all__ = ["ChunkConstructor"]


from typing import Iterable
from pyClarion.base import MatchSpec, Packet
from pyClarion.components.datastructures import Chunks
from collections import namedtuple


class ChunkConstructor(object):

    def __init__(self, threshold, op="max"):
        """
        Initialize a chunk constructor object.

        Collaborates with `Chunks` database.

        :param op: Default op for strength aggregation w/in dimensions.
        :param subsystem: Target subsystem to be monitored. If None, it is 
            assumed that the client realizer is the target subsystem.
        """

        self.threshold = threshold
        self.op = op

    def __call__(self, strengths: Packet) -> dict:
        """Create candidate chunk forms based on given strengths and filter."""

        eligible = (f for f, s in strengths.items() if s > self.threshold)
        form = Chunks.update_form({}, *eligible, op=self.op) # weights?
        
        return form
