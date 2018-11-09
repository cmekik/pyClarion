"""Enums for capturing basic info about constructs and/or activations"""

from enum import Enum, auto


class Level(Enum):
    """Symbolically represents the top or bottom level in Clarion."""

    Top = auto()
    Bot = auto()


class FlowType(Enum):
    """Signals the activation flow direction of a `Flow` construct."""

    Top2Top = auto()
    Bot2Bot = auto()
    Top2Bot = auto()
    Bot2Top = auto()