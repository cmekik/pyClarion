from typing import Literal
import enum


class Sentinel(enum.Enum):
    UNDEFINED = enum.auto()

type UndefinedType = Literal[Sentinel.UNDEFINED]
