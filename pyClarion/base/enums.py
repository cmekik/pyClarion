from enum import Enum, auto


class Level(Enum):

    Top = auto()
    Bot = auto()


class FlowType(Enum):
    """An enumeration of level types."""

    Top2Top = auto()
    Bot2Bot = auto()
    Top2Bot = auto()
    Bot2Top = auto()