"""Objects associated with defining, managing, and processing chunks."""


__all__ = ["Chunks", "TopDown", "BottomUp", "ChunkAdder", "ChunkExtractor"]


from ..base.symbols import (
    ConstructType, Symbol, 
    feature, chunk, terminus, subsystem
)
from ..base import numdicts as nd
from ..base.components import Propagator, UpdaterS 
from ..base.realizers import Construct

from .propagators import ThresholdSelector

from typing import (
    Mapping, Iterable, Union, Tuple, Set, Hashable, FrozenSet, Collection, List
)
from collections.abc import MutableMapping
from collections import namedtuple
from statistics import mean
from itertools import count
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
import operator


class Chunks(MutableMapping):
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
    """

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

            if isinstance(other, Chunks.Chunk):
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

            Implementation is based on p. 77-78 of Anatomy of the Mind.
            """

            d = nd.restrict(strengths, self.features)
            d = d.by(feature.dim.fget, max) # get maxima by dimensions
            weighted = d * self.weights / nd.val_sum(self.weights)
            strength = nd.val_sum(weighted)

            return strength

    def __init__(self, data: Mapping[chunk, "Chunks.Chunk"] = None) -> None:

        if data is None:
            data = dict()
        else:
            data = dict(data)

        self._data: MutableMapping[chunk, Chunks.Chunk] = data

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

        self._data[key] = val

    def link(self, ch, *features, weights=None):
        """Link chunk to features."""

        self[ch] = self.Chunk(features=features, weights=weights)

    def find_form(self, form):
        """Return the set of chunks matching the given form."""

        # This may need a faster implementation in the future. - Can
        chunks = set()
        for ch, form_ch in self.items():
            if form_ch == form:
                chunks.add(ch)

        return chunks

    def contains_form(self, form):
        """Return true if given chunk form matches at least one chunk."""

        found = self.find_form(form)

        return 0 < len(found) 


class TopDown(Propagator):
    """Computes top-down activations."""

    _serves = ConstructType.flow_tb | ConstructType.flow_in

    def __init__(self, source: Symbol, chunks: Chunks):

        self.source = source
        self.chunks = chunks 

    def expects(self, construct: Symbol):

        return construct == self.source

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

    def expects(self, construct: Symbol):

        return construct == self.source

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


class ChunkExtractor(ThresholdSelector):
    """
    Extracts chunks from the bottom level.
    
    Extracted chunks contain all features in the bottom level above a set 
    threshold. If the corresponding chunk form does not exist in client chunk 
    database, a new chunk is added to the database at call time. The strength 
    of the extracted chunk is written to the output.
    """

    def __init__(
        self, 
        source: Symbol, 
        chunks: Chunks, 
        prefix: str,
        threshold: float = 0.85
    ) -> None:

        super().__init__(source, threshold)
        
        self.chunks = chunks
        self.prefix = prefix
        self.counter = count(start=1, step=1)

        self._to_add: List[Tuple[chunk, Chunks.Chunk]] = []

    def call(self, inputs):
        """Extract a chunk from bottom-level activations."""

        fs = super().call(inputs)
        fs.squeeze()

        d = nd.NumDict()
        s_fs = nd.restrict(inputs[self.source], fs)

        if len(s_fs) > 0:

            form = self.chunks.Chunk(fs)
            found = self.chunks.find_form(form)
            
            if len(found) == 0:
                name = "{}_{}".format(self.prefix, next(self.counter))
                ch = chunk(name)
                d[ch] = 1.0
                # The new chunk is not added on the spot to prevent potential 
                # inconsistencies in construct behavior arising from call-time 
                # modifications to shared chunk data. Instead, it is kept in 
                # storage until update time.
                self._to_add.append((ch, form))
            elif len(found) == 1:
                d.extend(found, value=1.0)
            else:
                raise ValueError("Chunk database contains duplicate forms.")

        return d

    def update(self, inputs, output):
        """If a new chunk was extracted, add it to the database."""

        if len(self._to_add) == 0:
            pass
        elif len(self._to_add) == 1:
            ch, form = self._to_add.pop()
            self.chunks[ch] = form
        else:
            msg = "Too many chunks to add: got {}, limit is 1."
            raise ValueError(msg.format(len(self._to_add)))
