__all__ = ["AtomUpdate", "ChunkUpdate", "RuleUpdate"]


from ...knowledge.primitives import Atom, Atoms
from ...knowledge.compounds import Chunk, Rule, Chunks, Rules
from ...events.updates import KeyspaceUpdate


class AtomUpdate(KeyspaceUpdate[Atoms, Atom]):
    __slots__ = ()


class ChunkUpdate(KeyspaceUpdate[Chunks, Chunk]):
    __slots__ = ()


class RuleUpdate(KeyspaceUpdate[Rules, Rule]):
    __slots__ = ()