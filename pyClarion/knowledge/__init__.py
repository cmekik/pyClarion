from .base import Symbol, Term, Sort, Var, Family, Root
from .terms import Bus, Atom, Compound, Chunk, Rule
from .sorts import Buses, Atoms, Compounds, Chunks, Rules
from .families import DataFamily, AtomFamily, ChunkFamily, RuleFamily, BusFamily


type SemanticTerm = Atom | Chunk | Rule
type SemanticSort = Atoms | Chunks | Rules
type SemanticFamily = DataFamily | AtomFamily | ChunkFamily | RuleFamily
type SemanticSubspace = SemanticFamily | SemanticSort
type SemanticKeySpace = SemanticFamily | SemanticSort | SemanticTerm
type StructuralKeySpace = BusFamily | Buses | Bus
type DVPairs = tuple[Buses | Bus, SemanticSubspace]
type Nodes = DVPairs | Chunks | Rules

__all__ = [
    "Symbol", "Term", "Sort", "Var", "Family", "Root", 
    "Bus", "Atom", "Compound", "Chunk", "Rule",
    "Buses", "Atoms", "Compounds", "Chunks", "Rules",
    "DataFamily", "AtomFamily", "ChunkFamily", "RuleFamily", "BusFamily",
    "SemanticKeySpace", "StructuralKeySpace", "SemanticSubspace",
    "SemanticFamily", "SemanticSort", "SemanticTerm", "DVPairs", "Nodes"
]