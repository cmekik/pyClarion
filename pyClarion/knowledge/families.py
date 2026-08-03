__all__ = ["DataFamily", "AtomFamily", "ChunkFamily", "RuleFamily", "BusFamily", 
    "ParamFamily"]


from .base import Family, Sort
from .sorts import Buses, Atoms, Params, Compounds, Chunks, Rules


class DataFamily(Family[Sort]):
    def __init__(self, name: str = ""):
        super().__init__(name, skip=(Compounds,))

    @classmethod
    def _get_member_type_(cls) -> type[Sort]:
        return Sort


class AtomFamily(Family[Atoms]):
    @classmethod
    def _get_member_type_(cls) -> type[Atoms]:
        return Atoms

class BusFamily(Family[Buses]):
    @classmethod
    def _get_member_type_(cls) -> type[Buses]:
        return Buses


class ParamFamily(Family[Params]):
    @classmethod
    def _get_member_type_(cls) -> type[Params]:
        return Params


class ChunkFamily(Family[Chunks]):
    _init_required_ = False

    @classmethod
    def _get_member_type_(cls) -> type[Chunks]:
        return Chunks


class RuleFamily(Family[Rules]):
    _init_required_ = False

    @classmethod
    def _get_member_type_(cls) -> type[Rules]:
        return Rules
