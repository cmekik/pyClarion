"""
Objects associated with defining, managing, and processing chunks.

Provides:
    `Chunks`: A simple datastructure for storing chunk definitions.
    `TopDown`:
    `BottomUp`:
    `ChunkConstructor`:  
    `ChunkExtractor`: 
    `ChunkAdder`:
"""


from pyClarion.base import MatchSet, Construct, ConstructType, Symbol, chunk
from pyClarion.components.propagators import PropagatorA
from pyClarion.utils.str_funcs import pstr_iterable, pstr_iterable_cb
from typing import Mapping, Iterable
from collections import namedtuple
from statistics import mean
from itertools import count
from copy import copy


__all__ = [
    "Chunks", "TopDown", "BottomUp", "ChunkConstructor", "ChunkAdder"
]


class Chunks(object):
    """
    A simple chunk database.

    This object provides methods for constructing, maintaining, and inspecting a 
    database of links between chunk nodes and corresponding features.

    It is important to distinguish chunks from chunk nodes. Put simply, the 
    difference is that 'chunk' refers to a labeled chunk node along with its 
    links to feature nodes (these links may be empty). A chunk form, on the 
    other hand, refers to the pattern of connections between a labeled chunk 
    node and some feature nodes that define a chunk (excluding the labeled chunk 
    node itself).

    Chunks forms are stored in a dict of the form:
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
                        raise ValueError("Dimension data must contain weight info.")
                    if "values" not in dim_form:
                        raise ValueError("Dimension data must contain value info.")
                    if not isinstance(dim_form["values"], set):
                        raise TypeError("Value info must be of type set.")


class TopDown(PropagatorA):
    """
    Computes a top-down activations in NACS.

    During a top-down cycle, chunk strengths are multiplied by dimensional 
    weights to get dimensional strengths. Dimensional strengths are then 
    distributed evenly among features of the corresponding dimension that 
    are linked to the source chunk.

    Implementation is based on p. 77-78 of Anatomy of the Mind.
    """

    def __init__(self, chunks=None, op=None, default=0.0, matches=None):

        if matches is None: 
            matches = MatchSet(ctype=ConstructType.chunk)  
        super().__init__(matches=matches)
        
        self.chunks: Chunks = chunks if chunks is not None else Chunks()
        self.op = op if op is not None else max
        self.default = default

    def call(self, construct, inputs, **kwds):
        """
        Execute a top-down activation cycle.

        :param construct: Construct symbol for client construct.
        :param inputs: Dictionary mapping input constructs to their pull 
            methods.
        """

        if len(kwds) > 0:
            raise ValueError(
                (
                    "Unexpected keyword arguments passed to {}.call(): '{}'."
                ).format(self.__class__.__name__, next(iter(kwds.keys())))
            )

        d = {}
        for ch, dim_dict in self.chunks.items():
            for _, data in dim_dict.items():
                s = data["weight"] * inputs.get(ch, self.default)
                for feat in data["values"]:
                    l = d.setdefault(feat, [])
                    l.append(s)
        d = {f: self.op(l) for f, l in d.items()}

        return d


class BottomUp(PropagatorA):
    """
    Computes a bottom-up activations in NACS.

    During a bottom-up cycle, chunk strengths are computed as a weighted sum 
    of the maximum activation of linked features within each dimension. The 
    weights are simply top-down weights normalized over dimensions. 

    Implementation is based on p. 77-78 of Anatomy of the Mind.
    """

    default_ops = {"max": max, "min": min, "mean": mean}

    def __init__(self, chunks=None, ops=None, default=0.0, matches=None):

        if matches is None: 
            matches = MatchSet(ctype=ConstructType.feature)  
        super().__init__(matches=matches)
        
        self.chunks: Chunks = chunks if chunks is not None else Chunks()
        self.default = default
        self.ops = ops if ops is not None else self.default_ops.copy()

    def call(self, construct, inputs, **kwds): 
        """
        Execute a bottom-up activation cycle.

        :param construct: Construct symbol for client construct.
        :param inputs: Dictionary mapping input constructs to their pull 
            methods.
        """

        if len(kwds) > 0:
            raise ValueError(
                (
                    "Unexpected keyword arguments passed to {}.call(): '{}'."
                ).format(self.__class__.__name__, next(iter(kwds.keys())))
            )

        d = {}
        for ch, ch_data in self.chunks.items():
            divisor = sum(data["weight"] for data in ch_data.values())
            for dim, data in ch_data.items():
                op = self.ops[data["op"]]
                s = op(inputs.get(f, self.default) for f in data["values"])
                d[ch] = d.get(ch, self.default) + data["weight"] * s / divisor
        
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
                    "Cannot construct chunk containing {}".format(symbol)
                )

        form = Chunks.update_form({}, *features, op=self.op) # weights?
        
        return form


class ChunkAdder(object):
    """
    Adds new chunk nodes to client constructs.
    
    Constructs Node objects for new chunks from a given template and adds them 
    to client realizers.

    Does not allow adding updaters.

    Warning: This implementation relies on (shallow) copying. If propagators 
        have mutable attributes unexpected behavior may occur. To mitigate 
        this, propagators must define appropriate `__copy__()` methods.
    """

    def __init__(
        self, 
        emitter, 
        prefix,
        terminus, 
        subsystem=None, 
        clients=None, 
        op="max"
    ):
        """
        Initialize a new `ChunkAdder` instance.
        
        :param template: A ChunkAdder.Template object defining the form of 
            `Node` instances representing new chunks.
        :param terminus: Construct symbol for a terminus construct emmiting new 
            chunk recommendations. 
        :param subsystem: The subsystem that should be monitored. Used only if 
            the chunk adder is located at the `Agent` level.
        :param clients: Subsystem(s) to which new chunk nodes should be added. 
            If None, it will be assumed that the sole client is the realizer 
            housing this updater.
        :param prefix: Prefix added to created chunk names.
        :param op: Chunk op.
        """

        self.constructor = ChunkConstructor(op=op)
        self.emitter = emitter
        self.prefix = prefix
        self.count = count(start=1, step=1)

        self.terminus = terminus
        self.subsystem = subsystem
        self.clients = {subsystem} if clients is None else clients

    def __call__(self, realizer):

        if not isinstance(realizer.assets.chunks, Chunks):
            raise TypeError(
                "Realizer must have a `chunks` asset of type Chunks."
            )

        db: Chunks = realizer.assets.chunks
        subsystem = (
            realizer[self.subsystem] if self.subsystem is not None 
            else realizer
        )

        features = subsystem.output[self.terminus]
        form = self.constructor(features=features)
        chunks = db.find_form(form)
        added = set()
        if len(chunks) == 0:
            ch = chunk("{}-{}".format(self.prefix, next(self.count)))
            db.set_chunk(ch, form)
            added.add(ch)
        elif len(chunks) == 1:
            pass
        else:
            raise ValueError("Corrupt chunk database.")

        clients = self.clients if self.clients is not None else (None,)
        for construct in clients:
            client = realizer[construct] if construct is not None else realizer
            for ch in added: 
                client.add(
                    Construct(
                        name=ch,
                        emitter=copy(self.emitter)
                    )
                )
