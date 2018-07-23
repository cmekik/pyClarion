import abc
import typing as T
from . import subsystem

class Subject(object):

    def __init__(
        self, 
        subsystems : subsystem.SubsystemSet
    ) -> None:

        self.subsystems = subsystems

    def __call__(self, input_map : node.Node2Float) -> None:
        pass