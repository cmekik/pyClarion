"""Objects associated with defining, managing, and processing chunks."""


from ..base import (
    MatchSet, Construct, ConstructType, Symbol, chunk, UpdaterS, terminus, 
    subsystem, Propagator, feature
)
from ..base import numdicts as nd
from ..utils.str_funcs import pstr_iterable, pstr_iterable_cb
from typing import Mapping, Iterable, Union, Tuple, Set, Hashable, FrozenSet
from collections import namedtuple
from statistics import mean
from itertools import count
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType


__all__ = [
    "Chunks", "TopDown", "BottomUp", "ChunkConstructor", "ChunkAdder"
]


class Chunks(object):
    """
    A simple chunk database.

    This object provides methods for constructing, maintaining, and inspecting a 
    database of links between chunk nodes and corresponding feature nodes.

    It is important to distinguish chunks from chunk forms. Put simply, the 
    difference is that 'chunk' refers to a labeled chunk node along with its 
    links to feature nodes (these links may be empty). A chunk form, on the 
    other hand, refers to the pattern of connections between a labeled chunk 
    node and some feature nodes that define a chunk (excluding the labeled chunk 
    node itself).

    Chunks are stored in a dict of the form:
    _data = {
        chunk(1): {
            "dim1": {
                "op": "max",
                "values": {
                    feature("dim1", "val1"), 
                    feature("dim1, "val2"), 
                    ...
                },
                "weight": weight_1_1,
            },
            ... # other dimensions
        },
        ... # other chunks
    } 

    Under normal cricumstances, this dict should not be directly accessed and/or 
    modified. However, a dict of this form may be passed at initialization time 
    to set initial chunks.
    """

    _format = {"indent": 4, "digits": 3}

    # class Entry:
    #     """
    #     A chunk database entry.
        
    #     Specifies features and weights associated with a single chunk.
    #     """

    #     __slots__ = ("_features", "_weights", "_weights_proxy")

    #     def __init__(
    #         self, 
    #         features: Iterable[feature], 
    #         weights: Mapping[Tuple[Hashable, int], float] = None
    #     ):

    #         _weights = numdict(weights) if weights is not None else numdict()
    #         _weights.extend(features, value=1.0)

    #         self._features = set(features)
    #         self._weights = _weights

    #     def __repr__(self):

    #         template = "{}(features={}, weights={})"
    #         name = type(self).__name__
    #         frepr, wrepr = repr(self.features), repr(self.weights)

    #         return template.format(name, frepr, wrepr)

    #     def __eq__(self, other):

    #         if isinstance(other, Entry):
    #             fb = self.features == other.features
    #             wb = self.weights.isclose(other.weights)
    #             return fb and wb
    #         else:
    #             return NotImplemented

    #     @property
    #     def features(self):
    #         """Features associated with chunk defined by self."""
            
    #         return self._features
        
    #     @property
    #     def weights(self):
    #         """Dimensional weights for chunk defined by self."""

    #         return self._weights

    def __init__(self, data=None):

        self.validate_init_data(data)
        self._data = dict(data) if data is not None else dict()

    def __repr__(self):

        repr_ = "{}(data={})".format(type(self).__name__, repr(self._data))
        return repr_

    def __contains__(self, ch):
        """Return True if self contains given chunk."""

        return ch in self._data

    def __len__(self):

        return len(self._data)

    def get_form(self, ch, default=None):
        """Return the form of given chunk."""

        return self._data.get(ch, default)

    def find_form(self, form):
        """Return the set of chunks matching the given form."""

        # This may need a faster implementation in the future. - Can
        chunks = set()
        for ch, ch_form in self.items():
            if ch_form == form:
                chunks.add(ch)

        return chunks

    def chunks(self):
        """Return a view of chunks in self."""

        return self._data.keys()

    def forms(self):
        """Return a view of chunk forms in self."""

        return self._data.values()

    def items(self):
        """Return a view of chunk_node, chunk_form pairs in self."""

        return self._data.items()
    
    def link(self, ch, *features, op=None, weights=None):
        """Link chunk to features."""

        # If feature sequence contains duplicates, they will be ignored upon 
        # conversion to a set in update_form().
        d = self._data.setdefault(ch, dict())
        self.update_form(d, *features, op=op, weights=weights)

    def set_chunk(self, ch, form):
        """
        Set chunk to have given form to database.
        
        If the chunk is new, will simply add to database. Otherwise, will 
        overwrite exisiting data.
        """

        d = {ch: form}
        self.validate_init_data(d)
        self._data.update(d)

    def remove_chunk(self, ch):
        """Remove chunk from database."""

        del self._data[ch]
    
    def unlink_dim(self, ch, dim):
        """Unlink all features of a given dimension from chunk."""

        del self._data[ch][dim]

    def unlink_feature(self, ch, feat):
        """
        Unlink feature from chunk.

        If no features of the same dim are linked to chunk after operation, also 
        removes the dimension.
        """

        features = self._data[ch][feat.dim]["values"]
        features.remove(feat)
        if len(features) == 0:
            self.unlink_dim(ch, feat.dim)

    def set_weight(self, ch, dim, weight):
        """Set weight associated with a dimension of chunk."""

        self._data[ch][dim]["weight"] = weight

    def contains_form(self, *features, op=None, weights=None):
        """Return true if given chunk form matches at least one chunk."""

        test_form = self.update_form(dict(), *features, op=op, weights=weights)
        return any([form == test_form for form in self._data.values()]) 

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

    @staticmethod
    def update_form(form, *features, op=None, weights=None):
        """
        Update given chunk form.

        :param form: A chunk form (i.e., an unlabeled chunk).
        :param features: A sequence of feature construct symbols.
        :param op: Operation to compute dimensional activations.
        :param weights: A mapping from dimensions to corresponding weights.
        """
        
        for feat in features:
            op = op if op is not None else "max"
            w = weights[feat.dim] if weights is not None else 1.0
            dim_data = form.setdefault(
                feat.dim, {"op": op, "weight": w, "values": set()}
            )
            dim_data["values"].add(feat)
        
        return form

    @staticmethod
    def validate_init_data(data):
        """
        Check if initial data dict has valid form.

        Enforces
            - dict values must be dimension dicts
            - dimension dicts contain keys "weight", "values"
            - the "values" key returns an object of type set

        See class header for more information on expected data structure.
        """        

        if data is not None:
            for chunk_form in data.values():
                if not isinstance(chunk_form, dict):
                    raise TypeError("Chunk form must be of type dict.")
                for dim_form in chunk_form.values():
                    if "op" not in dim_form:
                        raise ValueError("Dimension data must specify op.")
                    if "weight" not in dim_form:
                        msg = "Dimension data must contain weight info."
                        raise ValueError(msg)
                    if "values" not in dim_form:
                        msg = "Dimension data must contain value info."
                        raise ValueError(msg)
                    if not isinstance(dim_form["values"], set):
                        raise TypeError("Value info must be of type set.")


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
        strengths = inputs[self.source]
        for ch, dim_dict in self.chunks.items():
            for _, data in dim_dict.items():
                s = data["weight"] * strengths[ch]
                fd = nd.NumDict()
                fd.extend(data["values"], value=s)
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
        for ch, ch_data in self.chunks.items():
            divisor = sum(data["weight"] for data in ch_data.values())
            for dim, data in ch_data.items():
                s = max(strengths[f] for f in data["values"])
                d[ch] += data["weight"] * s / divisor
        
        return d


class ChunkConstructor(object):
    """Constructs new chunks from feature strengths."""

    def __init__(self, op="max"):
        """
        Initialize a chunk constructor object.

        Collaborates with `Chunks` database.

        :param op: Default op for strength aggregation w/in dimensions.
        """

        self.op = op

    def __call__(self, features: Iterable[Symbol]) -> dict:
        """Create candidate chunk forms based on given strengths and filter."""

        for symbol in features:
            if not symbol.ctype in ConstructType.feature:
                raise TypeError(
                    "Cannot construct chunk containing {}.".format(symbol)
                )

        form = Chunks.update_form({}, *features, op=self.op) # weights?
        
        return form


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
        subsystem: subsystem = None,
        constructor: ChunkConstructor = None
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
        self.constructor = constructor or ChunkConstructor(op="max")
        self.prefix = prefix
        self.subsystem = subsystem
        self.count = count(start=1, step=1)

    def __call__(self, inputs, output, update_data):

        if self.subsystem is None:
            feature_set = output[self.terminus]
        else:
            feature_set = output[self.subsystem][self.terminus]
        
        if len(feature_set) > 0:
            form = self.constructor(features=feature_set)
            chunks = self.chunks.find_form(form)
            if len(chunks) == 0:
                ch = chunk("{}-{}".format(self.prefix, next(self.count)))
                self.chunks.set_chunk(ch, form)

    def expects(self, construct):

        if self.subsystem is not None:
            return construct == self.subsystem 
        else:
            return False
