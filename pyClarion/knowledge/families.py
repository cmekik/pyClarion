__all__ = ["DataFamily", "AtomFamily", "ChunkFamily", "RuleFamily", "BusFamily", 
    "ParamFamily"]


from .base import Family, Sort
from .sorts import Buses, Atoms, Chunks, Rules, Params


class DataFamily(Family[Sort]):
    @classmethod
    def _get_member_type_(cls) -> type[Sort]:
        return Sort


class AtomFamily(Family[Atoms]):
    @classmethod
    def _get_member_type_(cls) -> type[Atoms]:
        return Atoms


class ChunkFamily(Family[Chunks]):
    @classmethod
    def _get_member_type_(cls) -> type[Chunks]:
        return Chunks


class RuleFamily(Family[Rules]):
    @classmethod
    def _get_member_type_(cls) -> type[Rules]:
        return Rules


class BusFamily(Family[Buses]):
    @classmethod
    def _get_member_type_(cls) -> type[Buses]:
        return Buses


class ParamFamily(Family[Params]):
    @classmethod
    def _get_member_type_(cls) -> type[Params]:
        return Params