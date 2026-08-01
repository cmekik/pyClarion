__all__ = ["AtomUpdate", "ChunkUpdate", "RuleUpdate"]


from ...knowledge.terms import Atom, Chunk, Rule
from ...knowledge.sorts import Atoms, Chunks, Rules
from ...events.updates import KeyspaceUpdate


class AtomUpdate(KeyspaceUpdate[Atoms, Atom]):
    __slots__ = ()


class ChunkUpdate(KeyspaceUpdate[Chunks, Chunk]):
    __slots__ = ()


class RuleUpdate(KeyspaceUpdate[Rules, Rule]):
    __slots__ = ()