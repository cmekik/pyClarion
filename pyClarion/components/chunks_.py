"""Tools for defining, managing, and processing chunks."""


__all__ = [
    "Chunk", "Chunks", "ChunkDBUpdater", "TopDown", "BottomUp", "ChunkExtractor"
]


from ..base.symbols import (
    ConstructType, Symbol, SymbolTrie, SymbolicAddress, 
    feature, chunk, terminus, subsystem
)
from .. import numdicts as nd
from ..base.components import Process, Domain
from ..base.realizers import Construct

from .propagators import ThresholdSelector

from typing import (
    Mapping, Iterable, Union, Tuple, Set, Hashable, FrozenSet, Collection, 
    List, Optional, TypeVar, Type, Generic, MutableMapping, cast, overload, 
    Any, Dict
)
from contextlib import contextmanager
from collections import namedtuple
from statistics import mean
from itertools import count, chain
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
import operator


class Chunk(object):
    """
    Represents the form of a chunk.
    
    Specifies features and dimensional weights.
    """

    __slots__ = ("_features", "_weights")

    def __init__(
        self, 
        features: Collection[feature], 
        weights: Mapping[Tuple[Hashable, int], Union[float, int]] = None
    ):

        dims = set(f.dim for f in features)
        if not (weights is None or dims.issuperset(weights)):
            raise ValueError("Weight dims do not match features.")

        ws = nd.MutableNumDict(weights) 
        ws.extend(dims, value=1.0)

        assert set(ws) == dims

        self._features = frozenset(features)
        self._weights = ws

    def __repr__(self):

        template = "{}(features={}, weights={})"
        name = type(self).__name__
        frepr, wrepr = repr(self.features), repr(self.weights)

        return template.format(name, frepr, wrepr)

    def __eq__(self, other):

        if isinstance(other, Chunk):
            b = (
                self.features == other.features and
                nd.isclose(self.weights, other.weights)
            )
            return b
        else:
            return NotImplemented

    @property
    def features(self):
        """Features associated with chunk."""
        
        return self._features
    
    @property
    def weights(self):
        """Dimensional weights."""

        return self._weights

    def top_down(self, strength):
        """
        Compute top-down strengths for features of self.
        
        Multiplies strength by dimensional weights to get dimensional 
        strengths. Features are then activated to the level of the 
        corresponding dimensional strength. 

        Implementation is based on p. 77-78 of Anatomy of the Mind.
        """

        weighted = strength * self.weights

        d = nd.MutableNumDict(default=0.0)
        d.extend(self.features)
        d.set_by(weighted, feature.dim.fget)

        return d

    def bottom_up(self, strengths):
        """
        Compute bottom up strength for chunk associated with self.

        The bottom-up strength is computed as the weighted average of 
        dimensional strengths. Dimensional strengths are given by the 
        strength of the maximally activated feature in each dimension. 

        Implementation is based on p. 77-78 of Anatomy of the Mind. However, 
        no nonlinearity is included in the denominator of the equation.
        """

        d = nd.keep(strengths, keys=self.features)
        d = nd.max_by(d, keyfunc=feature.dim.fget) # get maxima by dims
        weighted = d * self.weights 
        strength = nd.val_sum(weighted) / nd.val_sum(self.weights)

        return strength

    def support(self, *domains: Domain) -> bool:
        """
        Return True iff domains support self.
        
        A set of domains is considered to support a chunk if every feature of 
        the chunk is in at least one of the domains.
        """
        
        vals = all(any(f in d.features for d in domains) for f in self.features)   
        
        return vals


Ct = TypeVar("Ct", bound="Chunk")
class Chunks(MutableMapping[chunk, Ct]):
    """
    A simple chunk database.

    Maps chunk symbols to chunk forms. Provides tools for requesting, applying, 
    and examining deferred updates. 
    """

    @overload
    def __init__(self: "Chunks[Chunk]") -> None:
        ...

    @overload
    def __init__(self: "Chunks[Ct]", *, chunk_type: Type[Ct]) -> None:
        ...

    @overload
    def __init__(self, data: Mapping[chunk, Ct], chunk_type: Type[Ct]) -> None:
        ...

    def __init__(
        self: "Chunks[Ct]", 
        data: Mapping[chunk, Ct] = None, 
        chunk_type: Type[Ct] = None
    ) -> None:

        if data is not None and chunk_type is None:
            msg = "Must specify chunk type explicitly when data is passed."
            raise ValueError(msg)

        _data: Dict[chunk, Ct]
        if data is None:
            _data = dict()
        else:
            _data = dict(data)
        self._data: MutableMapping[chunk, Ct] = _data
        self._domains: Tuple[Domain, ...] = ()
        self._enforce_support: bool = False

        if chunk_type is None:
            self.Chunk = Chunk
        else:
            self.Chunk = chunk_type
        
        self._add_promises: MutableMapping[chunk, Ct] = dict()
        self._del_promises: Set[chunk] = set()

    def __repr__(self):

        repr_ = "{}({})".format(type(self).__name__, repr(self._data))
        return repr_

    def __len__(self):

        return len(self._data)

    def __iter__(self):

        yield from iter(self._data)

    def __getitem__(self, key):

        return self._data[key]

    def __delitem__(self, key):

        del self._data[key]

    def __setitem__(self, key, val):

        if isinstance(val, self.Chunk):
            if self._enforce_support and not val.support(*self._domains):
                msg = "Chunk {} contains unexpected features."
                raise ValueError(msg.format(key.cid))
            self._data[key] = val
        else:
            msg = "This chunk database expects chunks of type '{}'." 
            TypeError(msg.format(type(self.Chunk.__name__)))

    @property
    def add_promises(self):
        """A view of promised additions."""

        return MappingProxyType(self._add_promises)

    @property
    def del_promises(self):
        """A view of promised deletions."""

        return frozenset(self._del_promises)

    def define(self, ch, *features, weights=None) -> chunk:
        """
        Create a new entry linking chunk to features.
        
        Returns the chunk symbol.
        """

        self[ch] = self.Chunk(features=features, weights=weights)

        return ch

    def find_form(self, form, check_promises=True):
        """
        Return the set of chunks matching the given form.
        
        If check_promises is True, will match form against promised chunks (see 
        Chunks.request_add()).
        """

        # This may need a faster implementation in the future. - Can
        chunks = set()
        for ch, form_ch in self.items():
            if form_ch == form:
                chunks.add(ch)
        for ch, form_ch in self._add_promises.items():
            if form_ch == form:
                chunks.add(ch)

        return chunks

    def contains_form(self, form, check_promises=True):
        """
        Return true if given chunk form matches at least one chunk.

        If check_promises is True, will match form against promised chunks (see 
        Chunks.request_add()).
        """

        found = self.find_form(form, check_promises)

        return 0 < len(found) 

    @contextmanager
    def enforce_support(self, *domains: Domain):
        """
        Reject new chunks that are not supported by domains.
        
        Use this method as a context manager when initializing complex 
        knowledge to automatically check for coding errors.
        """

        self._domains = domains
        self._enforce_support = True
        yield
        self._enforce_support = False
        self._domains = ()

    def request_add(self, ch, form):
        """
        Inform self of a new chunk to be added at update time.
        
        The new chunk will be added on call to self.step().

        If ch is already member of self, will overwrite the existing entry. Does 
        not check for duplicate forms. 
        
        If an update is already registered for chunk ch, will throw an error.
        """

        if ch in self._add_promises or ch in self._del_promises:
            msg = "Chunk {} already registered for a promised update."
            raise ValueError(msg.format(ch))
        else:
            self._add_promises[ch] = form

    def request_del(self, ch):
        """
        Inform self of an existing chunk to be removed at update time.

        The chunk will be removed on call to step(). 
        
        If ch is not already a member of self, will raise an error.
        """

        if ch in self._add_promises or ch in self._del_promises:
            msg = "Chunk {} already registered for a promised update."
            raise ValueError(msg.format(ch))
        elif ch not in self:
            raise ValueError("Cannot delete non-existent chunk.")
        else:
            self._del_promises.add(ch)

    def step(self):
        """
        Apply any promised updates (see request_add() and request_del()).
        
        Clears promised update dict upon completion.
        """

        for ch in self._del_promises:
            del self[ch]
        self._del_promises.clear()

        self.update(self._add_promises)
        self._add_promises.clear()


class ChunkDBUpdater(Process):
    """Applies requested updates to a client Chunks instance."""

    _serves = ConstructType.updater

    def __init__(self, chunks: "Chunks") -> None:
        """Initialize a ChunkDBUpdater instance."""

        super().__init__()
        self.chunks = chunks

    def call(
        self, inputs: Mapping[Tuple[Symbol, ...], nd.NumDict]
    ) -> nd.NumDict:
        """Resolve all outstanding chunk database update requests."""

        self.chunks.step()

        return super().call(inputs)


class TopDown(Process):
    """Computes top-down activations."""

    _serves = ConstructType.flow_tb | ConstructType.flow_in

    def __init__(self, source: Symbol, chunks: Chunks):

        super().__init__(expected=(source,))
        self.chunks = chunks 

    def call(self, inputs):
        """Execute a top-down activation cycle."""

        strengths, = self.extract_inputs(inputs)
        d = nd.MutableNumDict(default=0.0)
        for ch, form in self.chunks.items():
            d.max(form.top_down(strengths[ch]))
        d.squeeze()

        return d


class BottomUp(Process):
    """Computes bottom-up activations."""

    _serves = ConstructType.flow_bt | ConstructType.flow_in

    def __init__(self, source: Symbol, chunks: Chunks):

        super().__init__(expected=(source,))
        self.chunks = chunks 

    def call(self, inputs): 
        """
        Execute a bottom-up activation cycle.

        :param inputs: Dictionary mapping input constructs to their pull 
            methods.
        """

        strengths, = self.extract_inputs(inputs)
        d = nd.MutableNumDict(default=0.0)
        for ch, form in self.chunks.items():
            d[ch] = form.bottom_up(strengths)
        d.squeeze()

        return d


class ChunkExtractor(Process):
    """
    Extracts chunks from the bottom level.
    
    Extracted chunks contain all features in the bottom level above threshold. 
    If the corresponding chunk form does not exist in the client chunk database,
    a request to add the new chunk to the database is issued. Execution of the 
    request is not enforced by the extractor.
    """

    _serves = ConstructType.terminus

    def __init__(
        self, 
        source: Symbol, 
        chunks: Chunks, 
        prefix: str,
        threshold: float = 0.85
    ) -> None:
        
        super().__init__(expected=(source,))
        self.chunks = chunks
        self.prefix = prefix
        self.threshold = threshold

        self._counter = count(start=1, step=1)
        self._to_add: Optional[Tuple[chunk, Chunk]] = None

    def call(self, inputs):
        """Extract a chunk from bottom-level activations."""

        d = nd.MutableNumDict(default=0.0)
        strengths, = self.extract_inputs(inputs)
        fs = nd.threshold(strengths, th=self.threshold)
        if len(fs) > 0:
            form = self.chunks.Chunk(fs)
            name = "{}_{}".format(self.prefix, next(self._counter))
            ch = chunk(name)
            d[ch] = 1.0
            self.chunks.request_add(ch, form)

        return d
