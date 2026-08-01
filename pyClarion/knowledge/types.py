__all__ = ["StructuralKS", "DVPairs", "Nodes"]


from .base import Family, Sort, Term
from .families import BusFamily
from .sorts import Buses
from .terms import Bus
from ..numdicts.keyspaces import Dyad

type StructuralKS = BusFamily | Buses | Bus
type DVPairs = Dyad[StructuralKS, Family[Sort[Term]] | Sort[Term]]
type Nodes = DVPairs | Sort
