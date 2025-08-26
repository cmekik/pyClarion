from .base import Family
from .sorts import Buses, Atoms, Chunks, Rules


class DataFamily(Family):
    _m_type_ = (Atoms, Chunks, Rules)


class AtomFamily(Family):
    _m_type_ = Atoms


class ChunkFamily(Family):
    _m_type_ = Chunks


class RuleFamily(Family):
    _m_type_ = Rules


class BusFamily(Family):
    _m_type_ = Buses
