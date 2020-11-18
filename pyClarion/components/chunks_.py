"""Objects associated with defining, managing, and processing chunks."""


from ..base import (
    MatchSet, Construct, ConstructType, Symbol, chunk, UpdaterS, terminus, 
    subsystem, Propagator, feature
)
from ..base import numdicts as nd
from ..utils.str_funcs import pstr_iterable, pstr_iterable_cb
from typing import (
    Mapping, Iterable, Union, Tuple, Set, Hashable, FrozenSet, Collection
)
from collections.abc import MutableMapping
from collections import namedtuple
from statistics import mean
from itertools import count
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
import operator


__all__ = ["Chunks", "TopDown", "BottomUp", "ChunkAdder"]


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

    _format = {"indent": 4, "digits": 3}

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

    def __init__(self):

        self._data = {}

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

    def pstr(self):
        """Return a pretty string a representation of self."""

        body = pstr_iterable(
            iterable=self._data, 
            cb=pstr_iterable_cb, 
            cbargs={"digits": self._format["digits"]}, 
            indent=self._format["indent"], 
            level=1
        )
        size = len(self._data)
        head = " " * (size > 0) * self._format["indent"] + "data = "      
        content = head + body
        s = "{cls}({nl}{content}{nl})".format(
            cls=type(self).__name__, content=content, nl="\n" * bool(size)
        )
        return s

    def pprint(self):
        """Pretty print self."""

        print(self.pstr())


class TopDown(Propagator):
    """
    Computes a top-down activations in NACS.

    During a top-down cycle, chunk strengths are multiplied by dimensional 
    weights to get dimensional strengths. Dimensional strengths are then 
    distributed evenly among features of the corresponding dimension that 
    are linked to the source chunk.

    Implementation is based on p. 77-78 of Anatomy of the Mind.
    """

    _serves = ConstructType.flow_tb | ConstructType.flow_in

    def __init__(self, source: Symbol, chunks=None):

        self.source = source
        self.chunks: Chunks = chunks if chunks is not None else Chunks()

    def expects(self, construct: Symbol):

        return construct == self.source

    def call(self, inputs):
        """
        Execute a top-down activation cycle.

        :param construct: Construct symbol for client construct.
        :param inputs: Dictionary mapping input constructs to their pull 
            methods.
        """

        d = nd.NumDict()
        fd = nd.NumDict()
        strengths = inputs[self.source]
        for ch, form in self.chunks.items():
            weighted = strengths[ch] * form.weights
            fd.clear()
            fd.extend(form.features)
            fd.set_by(weighted, feature.dim.fget)
            d |= fd

        return d


class BottomUp(Propagator):
    """
    Computes a bottom-up activations in NACS.

    During a bottom-up cycle, chunk strengths are computed as a weighted sum 
    of the maximum activation of linked features within each dimension. The 
    weights are simply top-down weights normalized over dimensions. 

    Implementation is based on p. 77-78 of Anatomy of the Mind.
    """

    _serves = ConstructType.flow_bt | ConstructType.flow_in

    def __init__(self, source: Symbol, chunks=None):

        self.source = source
        self.chunks: Chunks = chunks if chunks is not None else Chunks()

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
            fd = nd.restrict(strengths, form.features)
            fd = fd.by(feature.dim.fget, max) # get maxima by dimensions
            weighted = fd * form.weights / sum(form.weights.values())
            d[ch] = sum(weighted.values())
        
        return d


class ChunkAdder(UpdaterS):
    """
    Adds new chunk nodes to client subsystem.
    
    Constructs and adds a new entry in the reference chunk database. 

    In the current implementation, dimensional weights are set to 1. This 
    limitations may be lifted in a future iteration.
    """

    # TODO: Override entrust() so that configuration matches served construct. 
    # - Can
    _serves = ConstructType.container_construct

    def __init__(
        self, 
        chunks: Chunks,
        terminus: terminus, 
        prefix: str,
        subsystem: subsystem = None
    ):
        """
        Initialize a new `ChunkAdder` instance.
        
        :param chunks: Client chunk database.
        :param terminus: Symbol for a terminus construct in client emmitting 
            new chunk recommendations. 
        :param prefix: Prefix added to created chunk names.
        :param subsystem: The client subsystem. If None, the parent realizer is 
            considered to be the client.
        :param constructor: A chunk constructor. If 'None', defaults to 
            ChunkConstructor(op="max").
        """

        self.chunks = chunks
        self.terminus = terminus
        self.prefix = prefix
        self.subsystem = subsystem
        self.count = count(start=1, step=1)

    def __call__(self, inputs, output, update_data):

        if self.subsystem is None:
            features = output[self.terminus]
        else:
            features = output[self.subsystem][self.terminus]
        
        if len(features) > 0:
            form = Chunks.Chunk(features=features)
            chunks = self.chunks.find_form(form)
            if len(chunks) == 0:
                ch = chunk("{}-{}".format(self.prefix, next(self.count)))
                self.chunks[ch] = form

    def expects(self, construct):

        if self.subsystem is not None:
            return construct == self.subsystem 
        else:
            return False
