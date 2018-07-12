import abc
import typing as T
from . import node
from . import subsystem

class Subject(object):

    def __init__(
        self, nodes : node.NodeSet, subsystems : subsystem.SubsystemSet
    ) -> None:

        self.nodes = nodes
        self.subsystems = subsystems

    def __call__(self, input_map : node.Node2Float) -> None:
        pass