"""Provides basic propagators for building pyClarion agents."""


__all__ = [
    "MaxNode", "AssociativeRules", "TopDown", "BottomUp", "BoltzmannSelector", 
    "ConstantBuffer", "Stimulus", "FilteredA", "FilteredR", "ChunkExtractor"
]


from pyClarion.base import ConstructSymbol, chunk 
from pyClarion.base.propagators import PropagatorA, PropagatorB, PropagatorR
from pyClarion.components.datastructures import Chunks, Rules 
from pyClarion.components.utils import ChunkConstructor
from pyClarion.utils.funcs import (
    max_strength, simple_junction, boltzmann_distribution, select, 
    multiplicative_filter, scale_strengths, linear_rule_strength
)
from statistics import mean
from itertools import count


##############################
### Activation Propagators ###
##############################


class MaxNode(PropagatorA):
    """Simple node returning maximum strength for given construct."""

    def __copy__(self):

        return type(self)()

    def call(self, construct, inputs, **kwds):

        packets = inputs.values()
        strength = max_strength(construct, packets)
        
        return strength


class AssociativeRules(PropagatorA):
    """
    Propagates activations among chunks through associative rules.
    
    The strength of the conclusion is calculated as a weighted sum of condition 
    strengths. In cases where there are multiple rules with the same conclusion, 
    the maximum is taken. 

    Implementation based on p. 73-74 of Anatomy of the Mind.
    """

    def __init__(self, rules: Rules, op=None, default=0.0):

        self.rules = rules
        self.op = op if op is not None else max
        self.default = default

    def call(self, construct, inputs, **kwds):

        if len(kwds) > 0:
            raise ValueError(
                (
                    "Unexpected keyword arguments passed to {}.call(): '{}'."
                ).format(self.__class__.__name__, next(iter(kwds.keys())))
            )
        
        d = {}
        packets = inputs.values()
        strengths = simple_junction(packets)
        for conc, cond_dict in self.rules:
            s = linear_rule_strength(cond_dict, strengths, self.default)
            l = d.setdefault(conc, [])
            l.append(s) 
        d = {c: self.op(l) for c, l in d.items()}
        
        return d


class TopDown(PropagatorA):
    """
    Computes a top-down activations in NACS.

    During a top-down cycle, chunk strengths are multiplied by dimensional 
    weights to get dimensional strengths. Dimensional strengths are then 
    distributed evenly among features of the corresponding dimension that 
    are linked to the source chunk.

    Implementation is based on p. 77-78 of Anatomy of the Mind.
    """

    def __init__(self, chunks=None, op=None, default=0.0):

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
        packets = inputs.values()
        strengths = simple_junction(packets)
        for ch, dim_dict in self.chunks.items():
            for _, data in dim_dict.items():
                s = data["weight"] * strengths.get(ch, self.default)
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

    def __init__(self, chunks=None, ops=None, default=0.0):

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
        packets = inputs.values()
        strengths = simple_junction(packets)
        for ch, ch_data in self.chunks.items():
            divisor = sum(data["weight"] for data in ch_data.values())
            for dim, data in ch_data.items():
                op = self.ops[data["op"]]
                s = op(strengths.get(f, self.default) for f in data["values"])
                d[ch] = d.get(ch, self.default) + data["weight"] * s / divisor
        
        return d


############################
### Response Propagators ###
############################


class BoltzmannSelector(PropagatorR):
    """Selects a chunk according to a Boltzmann distribution."""

    def __init__(self, temperature, k=1):
        """
        Initialize a ``BoltzmannSelector`` instance.

        :param temperature: Temperature of the Boltzmann distribution.
        """

        self.temperature = temperature
        self.k = k

    def call(self, construct, inputs, **kwds):
        """Select actionable chunks for execution. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.

        :param strengths: Mapping of node strengths.
        """

        packets = inputs.values()
        strengths = simple_junction(packets)
        probabilities = boltzmann_distribution(strengths, self.temperature)
        selection = select(probabilities, self.k)

        return probabilities, selection


class ChunkExtractor(PropagatorR):

    def __init__(self, items, threshold, op="max"):

        self.constructor = ChunkConstructor(threshold=threshold, op=op)
        self.items = items
        
    def call(self, construct, inputs, **kwds):

        packets = inputs.values()
        strengths = simple_junction(packets)
        names = [
            chunk("{}-{}".format(item.name, next(item.counter)))
            for item in self.items
        ]
        forms = self.constructor(
            strengths=strengths, 
            filters=[item.filter for item in self.items]
        )
        extracts = {n: f for n, f in zip(names, forms)}
        selection = set()

        return extracts, selection 

    class Item(object):
        """Configuration data for a chunk extractor."""

        def __init__(self, name, filter, flag=None):

            self.name = name
            self.flag = flag
            self.filter = filter
            self.counter = count(start=1, step=1)


##########################
### Buffer Propagators ###
##########################


class ConstantBuffer(PropagatorB):
    """Outputs a stored activation packet."""

    def __init__(self, strengths = None) -> None:

        self.strengths = strengths or dict()

    def call(self, construct, inputs, **kwds):
        """Return stored strengths."""

        return self.strengths

    def update(self, strengths):
        """Update self with contents of dict-like strengths."""

        self.strengths = self.strengths.copy()
        self.strengths.update(strengths)

    def clear(self) -> None:
        """Clear stored node strengths."""

        self.strengths = {}


class Stimulus(PropagatorB):
    """Propagates externally provided stimulus."""

    def call(self, construct, inputs, stimulus=None, **kwds):

        return stimulus or {}


##########################
### Filtering Wrappers ###
##########################


class FilteredA(PropagatorA):
    """Filters input and output activations of an activation propagator."""
    
    def __init__(
        self, 
        base: PropagatorA, 
        source_filter: ConstructSymbol = None, 
        input_filter: ConstructSymbol = None, 
        output_filter: ConstructSymbol = None, 
        fdefault=0.0
    ):

        self.base = base
        # Expected types for source_filter, input_filter and output_filter 
        # should be construct symbols.
        self.source_filter = source_filter
        self.input_filter = input_filter
        self.output_filter = output_filter
        self.fdefault = fdefault

    def __copy__(self):

        return type(self)(
            base=copy(self.base),
            source_filter=copy(self.source_filter),
            input_filter=copy(self.input_filter),
            output_filter=copy(self.output_filter),
            fdefault=copy(self.fdefault)
        )

    def call(self, construct, inputs, **kwds):

        # Get filter settings and remove filter info from inputs dict so they 
        # are not processed by self.base.
        if self.source_filter is not None:
            source_weights = inputs.pop(self.source_filter)
        if self.input_filter is not None:
            input_weights = inputs.pop(self.input_filter)
        if self.output_filter is not None:
            output_weights = inputs.pop(self.output_filter)

        # Apply source filtering
        if self.source_filter is not None:
            inputs = {
                source: scale_strengths(
                    weight=source_weights.get(source, 1 - self.fdefault), 
                    strengths=packet, 
                ) 
                for source, packet in inputs.items()
            }

        # Filter inputs to base
        if self.input_filter is not None:
            inputs = {
                source: multiplicative_filter(
                    filter_weights=input_weights, 
                    strengths=packet, 
                    fdefault=self.fdefault
                )
                for source, packet in inputs.items()
            }
        
        # Call base on (potentially) filtered inputs. Note that call is to 
        # `base.call()` instead of `base.__call__()`. This is because we rely 
        # on `self.__call__()` instead.
        output = self.base.call(construct, inputs, **kwds)

        # Filter outputs of base
        if self.output_filter is not None:
            output = multiplicative_filter(
                filter_weights=output_weights, 
                strengths=output, 
                fdefault=self.fdefault
            )

        return output


class FilteredR(PropagatorR):
    """Filters input and output activations of a decision propagator."""
    
    def __init__(
        self, 
        base: PropagatorR, 
        input_filter: ConstructSymbol = None, 
        fdefault=0.0
    ):

        self.base = base
        self.input_filter = input_filter
        self.fdefault = fdefault

    def call(self, construct, inputs, **kwds):

        # Get filter settings and remove filter info from inputs dict so they 
        # are not processed by self.base
        if self.input_filter is not None:
            input_weights = inputs.pop(self.input_filter)

        # Filter inputs to base
        if self.input_filter is not None:
            inputs = {
                source: multiplicative_filter(
                    filter_weights=input_weights, 
                    strengths=packet, 
                    fdefault=self.fdefault
                )
                for source, packet in inputs.items()
            }
        
        # Call base on (potentially) filtered inputs. Note that call is to 
        # `base.call()` instead of `base.__call__()`. This is because we rely 
        # on `self.__call__()` instead.
        output = self.base.call(construct, inputs, **kwds)

        return output
