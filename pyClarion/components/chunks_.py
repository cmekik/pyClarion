"""Objects associated with defining, managing, and processing chunks."""


__all__ = ["Chunk", "Chunks", "TopDown", "BottomUp", "ChunkExtractor"]


from ..base.symbols import (
    ConstructType, Symbol, 
    feature, chunk, terminus, subsystem
)
from ..base import numdicts as nd
from ..base.components import Propagator, FeatureInterface, UpdaterS 
from ..base.realizers import Construct

from .propagators import ThresholdSelector

from typing import (
    Mapping, Iterable, Union, Tuple, Set, Hashable, FrozenSet, Collection, 
    List, Optional, TypeVar, Type, Generic
)
from collections.abc import MutableMapping
from collections import namedtuple
from statistics import mean
from itertools import count
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
        weights: Mapping[Tuple[Hashable, int], float] = None
    ):

        # why is this annoying mypy?
        func = feature.dim.fget # type: ignore
        dims = tuple(map(func, features))

        if weights is not None:
            ws = nd.NumDict(weights) 
            ws = nd.restrict(ws, dims)
        else:
            ws = nd.NumDict()

        ws.extend(dims, value=1.0)

        self._features = frozenset(features)
        self._weights = nd.FrozenNumDict(ws)

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
        """Features associated with chunk defined by self."""
        
        return self._features
    
    @property
    def weights(self):
        """Dimensional weights for chunk defined by self."""

        return self._weights

    def top_down(self, strength):
        """
        Compute top-down strengths for features of self.
        
        Multiplies strength by dimensional weights to get dimensional 
        strengths. Features are then activated to the level of the 
        corresponding dimensional strength. 

        Implementation is based on p. 77-78 of Anatomy of the Mind.
        """

        d = nd.NumDict()
        weighted = strength * self.weights
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

        d = nd.restrict(strengths, self.features)
        d = d.by(feature.dim.fget, max) # get maxima by dimensions
        weighted = d * self.weights / nd.val_sum(self.weights)
        strength = nd.val_sum(weighted)

        return strength


Ct = TypeVar("Ct", bound="Chunk")
class Chunks(MutableMapping, Generic[Ct]):
    """
    A simple chunk database.

    Stores chunks in a dict of the form:
    
    {
        chunk(1): Chunk(
            features={f1, ..., fn},
            weights={d1: w1, ..., dk: wk}
        ),
        ... # other chunks
    } 

    Provides mapping methods to access and manipulate this dict and defines the 
    entry type Chunk. 

    Also, this object provides tools for requesting, applying, and examining 
    deferred updates. These methods are useful for cases where multiple 
    constructs may want to add chunks to the same database.
    """

    class Updater(UpdaterS):
        """
        Applies requested updates to a client Chunks instance.
        
        Assumes any updates will be issued by constructs subordinate to 
        self.client.
        """

        _serves = ConstructType.container_construct

        def __init__(self, chunks: "Chunks") -> None:
            """Initialize a Chunks.Updater instance."""

            self.chunks = chunks

        @property
        def expected(self):

            return frozenset()

        def __call__(self, inputs, output, update_data):
            """Resolve all outstanding chunk database update requests."""

            self.chunks.resolve_update_requests()

    def __init__(
        self, 
        data: Mapping[chunk, Ct] = None, 
        chunk_type: Type[Ct] = None
    ) -> None:

        if data is None:
            data = dict()
        else:
            data = dict(data)

        self._data: MutableMapping[chunk, Ct] = data
        self.Chunk = chunk_type if chunk_type is not None else Chunk
        
        self._promises: MutableMapping[chunk, Chunk] = dict()
        self._promises_proxy = MappingProxyType(self._promises)
        self._updater = type(self).Updater(self)

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
            self._data[key] = val
        else:
            msg = "This chunk database expects chunks of type '{}'." 
            TypeError(msg.format(type(self.Chunk.__name__)))

    @property
    def updater(self):
        """Updater object for bound to self."""

        return self._updater

    @property
    def promises(self):
        """A view of promised updates."""

        return self._promises_proxy

    def link(self, ch, *features, weights=None):
        """Link chunk to features."""

        self[ch] = self.Chunk(features=features, weights=weights)

    def find_form(self, form, check_promises=True):
        """
        Return the set of chunks matching the given form.
        
        If check_promises is True, will match form against promised chunks (see 
        Chunks.request_update()).
        """

        # This may need a faster implementation in the future. - Can
        chunks = set()
        for ch, form_ch in self.items():
            if form_ch == form:
                chunks.add(ch)
        for ch, form_ch in self._promises.items():
            if form_ch == form:
                chunks.add(ch)

        return chunks

    def contains_form(self, form, check_promises=True):
        """
        Return true if given chunk form matches at least one chunk.

        If check_promises is True, will match form against promised chunks (see 
        Chunks.request_update()).
        """

        found = self.find_form(form, check_promises)

        return 0 < len(found) 

    def request_update(self, ch, form):
        """
        Inform self of a new chunk to be applied at a later time.
        
        Adds (ch, form) to an internal future update dict. Upon call to 
        self.resolve_update_requests(), the update dict will be passed as an 
        argument to self.update(). 
        
        Will overwrite existing chunks, if ch is already member of self. Does 
        not check for duplicate forms. Will throw an error if an update is 
        already registered for chunk ch.
        """

        if ch in self._promises:
            msg = "Chunk {} already registered for a promised update."
            raise ValueError(msg.format(ch))
        else:
            self._promises[ch] = form

    def resolve_update_requests(self):
        """
        Apply any promised updates (see Chunks.request_update()).
        
        Clears promised update dict upon completion.
        """

        self.update(self._promises)
        self._promises.clear()


class TopDown(Propagator):
    """Computes top-down activations."""

    _serves = ConstructType.flow_tb | ConstructType.flow_in

    def __init__(self, source: Symbol, chunks: Chunks):

        self.source = source
        self.chunks = chunks 

    @property
    def expected(self):

        return frozenset((self.source,))

    def call(self, inputs):
        """Execute a top-down activation cycle."""

        d = nd.NumDict()
        strengths = inputs[self.source]
        for ch, form in self.chunks.items():
            d |= form.top_down(strengths[ch])

        return d


class BottomUp(Propagator):
    """Computes bottom-up activations."""

    _serves = ConstructType.flow_bt | ConstructType.flow_in

    def __init__(self, source: Symbol, chunks: Chunks):

        self.source = source
        self.chunks = chunks 

    @property
    def expected(self):

        return frozenset((self.source,))

    def call(self, inputs): 
        """
        Execute a bottom-up activation cycle.

        :param inputs: Dictionary mapping input constructs to their pull 
            methods.
        """

        d = nd.NumDict()
        strengths = inputs[self.source]
        for ch, form in self.chunks.items():
            d[ch] = form.bottom_up(strengths)

        return d


class ChunkExtractor(Propagator):
    """
    Extracts chunks from the bottom level.
    
    Extracted chunks contain all features in the bottom level above a set 
    threshold. If the corresponding chunk form does not exist in client chunk 
    database, a request to update the chunk database is placed at call time. 
    Execution of the request is not enforced by the extractor.
    """

    _serves = ConstructType.terminus

    def __init__(
        self, 
        source: Symbol, 
        chunks: Chunks, 
        prefix: str,
        threshold: float = 0.85
    ) -> None:
        
        self.source = source
        self.chunks = chunks
        self.prefix = prefix
        self.threshold = threshold

        self._counter = count(start=1, step=1)
        self._to_add: Optional[Tuple[chunk, Chunk]] = None

    @property
    def expected(self):

        return frozenset((self.source,))

    def call(self, inputs):
        """Extract a chunk from bottom-level activations."""

        d = nd.NumDict()
        fs = nd.threshold(inputs[self.source], self.threshold)

        if len(fs) > 0:

            form = self.chunks.Chunk(fs)
            found = self.chunks.find_form(form)
            
            if len(found) == 0:
                name = "{}_{}".format(self.prefix, next(self._counter))
                ch = chunk(name)
                d[ch] = 1.0
                # The chunk will be added to the database at update time, 
                # assuming the self.chunks is updated at that time. By default, 
                # extractors do not initiate updates for their chunk databases 
                # that they serve.
                self.chunks.request_update(ch, form)
            elif len(found) == 1:
                d.extend(found, value=1.0)
            else:
                raise ValueError("Chunk database contains duplicate forms.")

        return d


class ControlledExtractor(ChunkExtractor):
    """Extract chunks from the bottom level conditional on a command."""

    @dataclass
    class Interface(FeatureInterface):
        """
        Control interface for ControlledExtractor.

        :param tag: Dimensional tag for control features.
        :param vals: Values for control features. The first entry represents 
            a standby command, the second value represents a firing command.
        """

        tag: Hashable
        vals: Tuple[Hashable, Hashable]

        def _validate_data(self):

            if len(set(vals)) != len(vals):
                raise ValueError("Values must be distinct.")
        
        def _set_interface_properties(self):

            self._cmds = frozenset(feature(self.tag, val) for val in self.vals)
            self._defaults = frozenset({feature(self.tag, self.vals[0])})
            self._params = frozenset()

    def __init__(
        self, 
        source: Symbol, 
        controller: Symbol,
        interface: Interface,
        chunks: Chunks, 
        prefix: str,
        threshold: float = 0.85
    ) -> None:

        super().__init__(source, chunks, prefix, threshold)
        
        self.controller = controller
        self.interface = interface

    @property
    def expected(self):

        return super().expected.union((self.controller[0],))

    def call(self, inputs):

        data = self.inputs[self.controller]
        cmds = self.interface.parse_commands(data)

        dim, = self.interface.cmd_dims # unpack the single control dimension
        if cmds[dim] == self.interface.vals[1]:
            d = super().call(inputs)
        else:
            d = nd.NumDict()

        return d
