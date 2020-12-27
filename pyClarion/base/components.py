"""Basic definitions for constructing components."""


__all__ = [
    "Process", "CompositeProcess", "WrappedProcess", "Assets", "FeatureDomain", 
    "FeatureInterface", "SimpleDomain", "SimpleInterface"
]


from .symbols import (
    ConstructType, Symbol, SymbolTrie, SymbolicAddress, feature, group_by_dims
)
from .. import numdicts as nd

from typing import (
    TypeVar, Union, Tuple, Dict, Callable, Hashable, Generic, Any, Optional, 
    Text, Iterator, Iterable, Mapping, ClassVar, List, FrozenSet, Sequence, 
    Collection, Set, cast, no_type_check
)
from abc import abstractmethod
from types import SimpleNamespace, MappingProxyType
from dataclasses import dataclass
import logging


Pt = TypeVar("Pt", bound="Process")



class Process(object):
    """
    Defines how constructs connect, compute outputs, perform updates etc.
    
    Process instances are responsible for implementing the function or process 
    associated with some client construct.

    The attributes `interface` and `domain` are reserved for FeatureDomain and
    FeatureInterface instances, respectively, each defining the domain or 
    interface of a Process instance. These may be left undefined.
    
    If a domain is defined, it should be assumed that the Process instance will 
    only operate on features of that domain. If an interface is defined, it 
    should be assumed that the interface exposes features that can be used to 
    control the behavior of the Process. 
    
    In some cases, Processes may serve various domains or interfaces that are 
    not their own in the sense defined above. In such cases, the client domains 
    or interfaces should NOT be stored in the `domain` or `interface` attribute.
    """

    _serves: ClassVar[ConstructType] = ConstructType.null_construct

    _client: Symbol
    _expected: Tuple[SymbolicAddress, ...]

    def __init__(self, expected: Sequence[SymbolicAddress] = None):

        self._expected = tuple(expected or ()) 

    def __call__(self, inputs: SymbolTrie[nd.NumDict]) -> nd.NumDict:
        """
        Execute construct's forward propagation cycle.

        Pulls expected data from inputs constructs, delegates processing to 
        self.call(), and passes result to self.emit().
        """

        return self.emit(self.call(inputs))

    @property
    def client(self) -> Symbol:
        """Client construct entrusted to self."""

        return self._client

    @property
    def expected(self) -> Tuple[SymbolicAddress, ...]:
        """Constructs from which self expects to receive activations."""

        return self._expected

    def expects(self, construct: Symbol) -> bool:
        """Return True iff self pulls input from construct."""

        return construct in self._expects()

    def _expects(self) -> Set[Symbol]:

        s = set()
        for x in self.expected:
            if isinstance(x, Symbol):
                s.add(x)
            else:
                assert isinstance(x, tuple)
                if len(x) == 0:
                    msg = "Empty tuple encountered in symbolic address."
                    raise RuntimeError(msg)
                else:
                    s.add(x[0])

        return s

    def check_inputs(self, inputs: SymbolTrie[nd.NumDict]) -> None:
        """Raise an error iff inputs to self are NOT as expected."""

        # TODO: Make errors more precise & print more informative msgs. - Can

        for source in self.expected:
            if isinstance(source, Symbol):
                source = (source,)
            val = inputs
            for i, x in enumerate(source):
                try:
                    _val = val[x]
                except KeyError:
                    msg = "Missing input data: {}."
                    raise RuntimeError(msg.format(source)) from None
                if i < len(source) - 1:
                    if isinstance(_val, nd.NumDict):
                        msg = "Expected SymbolTrie but got NumDict at: {}."
                        raise RuntimeError(msg.format(source))
                    val = _val
                else:
                    assert i == len(source) - 1
                    if not isinstance(_val, nd.NumDict):
                        msg = "Expected NumDict but got SymbolTrie at: {}"
                        raise RuntimeError(msg.format(source))

    def extract_inputs(
        self, inputs: SymbolTrie[nd.NumDict]
    ) -> Tuple[nd.NumDict, ...]:
        """
        Extract expected inputs from given symbol trie.

        Returns inputs in the order set by self.expected.
        """

        self.check_inputs(inputs) # Make sure inputs have correct structure.

        # NOTE: Casts below should be safe assuming check_inputs is correct.
        extracted = []
        for source in self.expected:
            if isinstance(source, Symbol):
                source = (source,)
            _extracted: Any = inputs
            for x in source:
                _extracted = _extracted[x]
            extracted.append(cast(nd.NumDict, _extracted))

        return tuple(extracted)

    def entrust(self, construct: Symbol) -> None:
        """Entrust handling of construct to self."""

        if construct.ctype in type(self)._serves:
            self._client = construct
        else:
            msg = "{} cannot serve constructs of type {}."
            name, ctype = type(self).__name__, repr(construct.ctype) 
            raise ValueError(msg.format(name, ctype))

    def call(self, inputs: SymbolTrie[nd.NumDict]) -> nd.NumDict:
        """
        Compute construct's output.

        :param inputs: Pairs the names of input constructs with their outputs. 
        """

        return nd.NumDict(default=0)

    def emit(self, data: nd.D = None) -> nd.NumDict:
        """
        Emit output.

        If data is None, emits an empty numdict. Otherwise, emits a frozen 
        version of data. 
        
        Raises ValueError if default of data is not 0.
        """
        
        if data is None:
            return nd.NumDict(default=0.0)
        elif data.default != 0:
            msg = "Unexpected default in passed to {}."
            raise ValueError(msg.format(type(self).__name__))
        elif isinstance(data, nd.NumDict):
            d = nd.squeeze(data)
            return d
        else:
            msg = "Expected NumDict instance, got {}"
            raise TypeError(msg.format(type(data).__name__))


class CompositeProcess(Process, Generic[Pt]):
    """A process built on top of an existing process."""
    
    base: Pt

    def __init__(
        self, base: Pt, expected: Sequence[SymbolicAddress] = None
    ) -> None:

        _expected = tuple(expected or ())

        super().__init__(expected=_expected + base.expected)

        self._expected_top = _expected
        self.base = base

    @property
    def client(self) -> Symbol:
        """Client construct entrusted to self."""

        return self.base.client

    @property
    def expected(self) -> Tuple[SymbolicAddress, ...]:
        """
        Expected input constructs.

        Wrapper's own expected constructs are listed first, followed by those of
        self.base.
        """

        return super().expected

    @property
    def expected_top(self) -> Tuple[SymbolicAddress, ...]:
        """Input constructs expected exclusively by the top of the composite."""

        return self._expected_top

    def entrust(self, construct: Symbol) -> None:
        """Entrust handling of construct to self."""

        self.base.entrust(construct)


class WrappedProcess(CompositeProcess[Pt]):
    """
    A Process wrapped by a pre- and/or post- processor.

    The attribute `base` is reserved for the wrapped Processor instance.
    """

    def call(self, inputs):
        """
        Compute base construct's output.

        Feeds base construct preprocessed input and postprocesses its output.
        """

        preprocessed = self.preprocess(inputs)
        output = self.base.call(preprocessed)
        postprocessed = self.postprocess(inputs, output)
    
        return postprocessed

    def preprocess(
        self, inputs: SymbolTrie[nd.NumDict]
    ) -> SymbolTrie[nd.NumDict]:
        """Preprocess inputs to base construct."""

        return inputs

    def postprocess(
        self, inputs: SymbolTrie[nd.NumDict], output: nd.NumDict
    ) -> nd.NumDict:
        """Postprocess output of base construct."""

        return output


# @no_type_check disables type_checking for Assets (but not subclasses). 
# Included b/c dynamic usage of Assets causes mypy to complain.
# @no_type_check is not supported on mypy as of 2020-06-10. 'type: ignore' will 
# do for now. - Can
@no_type_check
class Assets(SimpleNamespace): # type: ignore
    """
    Dynamic namespace for construct assets.
    
    Provides handles for various datastructures such as chunk databases, rule 
    databases, bla information, etc. In general, all resources shared among 
    different components of a container construct are considered assets. 
    """
    pass


class FeatureDomain(object):
    """
    Formally defines a feature domain.

    Represents a collection of features defined for the purposes of a 
    simulation with methods for parsing out dimensions, tags, etc.
    
    Intended to be used as a dataclass.
    """

    _features: FrozenSet[feature]
    _tags: FrozenSet[Hashable]
    _dims: FrozenSet[Tuple[Hashable, int]]

    def __post_init__(self) -> None:

        self.compute_properties()

    @property
    def features(self) -> FrozenSet[feature]:
        """The set of features defined by self."""
        
        return self._features

    @property
    def tags(self) -> FrozenSet[Hashable]:
        """The set of dimensional labels defined by self."""
        
        return self._tags

    @property
    def dims(self) -> FrozenSet[Tuple[Hashable, int]]:
        """The set of feature dimensions (w/ lags) defined by self."""
        
        return self._dims

    @property
    def features_by_dims(
        self
    ) -> Mapping[Tuple[Hashable, int], Tuple[feature, ...]]:
        """Features grouped by dims."""

        # type should be more precise, but cannot b/c group_by does not support 
        # it.

        return self._features_by_dims

    def compute_properties(self) -> None:
        """
        Compute domain properties.
        
        When domains are deployed as dataclasses, this method is called 
        after initialization to populate domain properties. 
        
        Must be called again after modifications to domain configuration to 
        ensure that domain properties reflect desired changes.
        """

        self._validate_data()
        self._set_interface_properties()
        self._set_derivative_properties()
        self._validate_interface_properties()

    def _validate_data(self) -> None:

        # Should throw errors if dataclass members violate assumptions or 
        # requirements

        raise NotImplementedError()

    def _set_interface_properties(self) -> None:
        """
        Set basic interface properties.
        
        Should minimally set self._features. Other data may be set as necessary.
        """

        # Should minimally set self._features. Other properties may be set as 
        # necessary.

        raise NotImplementedError()

    def _set_derivative_properties(self) -> None:

        self._tags = frozenset(f.tag for f in self.features)
        self._dims = frozenset(f.dim for f in self.features)
        self._features_by_dims = MappingProxyType(group_by_dims(self.features))

    def _validate_interface_properties(self) -> None:

        _features_dims = set(f.dim for f in self.features)
        _features_tags = set(f.tag for f in self.features)

        # TODO: Use a more suitable exception class. - Can

        if not (self.tags == _features_tags):
            raise ValueError("self.tag conflicts with self.features.")
        if not (self.dims == _features_dims):
            raise ValueError("self.dims conflicts with self.features.")


class FeatureInterface(FeatureDomain):
    """
    Control interface for a component.
    
    Defines control features and default values. Provides parsing utilities.
    Each defined feature dimension is interpreted as defining a specific set of 
    alternate actions. A default value must be defined for each dimension, 
    representing the 'do nothing' action.

    Intended to be used as a dataclass.
    """

    _cmds: FrozenSet[feature]
    _defaults: FrozenSet[feature]
    _flags: FrozenSet[feature]
    _params: FrozenSet[feature]

    @property
    def cmds(self) -> FrozenSet[feature]:
        """Features representing (discrete) actions."""

        return self._cmds

    @property
    def defaults(self) -> FrozenSet[feature]:
        """Features representing default actions."""
        
        return self._defaults

    @property
    def flags(self) -> FrozenSet[feature]:
        """Features representing the state of the client process."""

        return self._flags

    @property
    def params(self) -> FrozenSet[feature]:
        """Features representing action parameters."""

        return self._params

    @property
    def cmd_tags(self) -> FrozenSet[Hashable]:
        """The set of command dimensional labels defined by self."""
        
        return self._cmd_tags

    @property
    def cmd_dims(self) -> FrozenSet[Tuple[Hashable, int]]:
        """The set of command feature dimensions (w/ lags) defined by self."""
        
        return self._cmd_dims

    @property
    def flag_tags(self) -> FrozenSet[Hashable]:
        """The set of flag dimensional labels defined by self."""
        
        return self._flag_tags

    @property
    def flag_dims(self) -> FrozenSet[Tuple[Hashable, int]]:
        """The set of flag feature dimensions (w/ lags) defined by self."""
        
        return self._flag_dims

    @property
    def param_tags(self) -> FrozenSet[Hashable]:
        """The set of param dimensional labels defined by self."""
        
        return self._param_tags

    @property
    def param_dims(self) -> FrozenSet[Tuple[Hashable, int]]:
        """The set of param feature dimensions (w/ lags) defined by self."""
        
        return self._param_dims

    @property
    def cmds_by_dims(
        self
    ) -> Mapping[Tuple[Hashable, int], Tuple[feature, ...]]:
        """Command features grouped by dims."""

        return self._cmds_by_dims

    @property
    def flags_by_dims(
        self
    ) -> Mapping[Tuple[Hashable, int], Tuple[feature, ...]]:
        """Flag features grouped by dims."""

        return self._flags_by_dims

    @property
    def params_by_dims(
        self
    ) -> Mapping[Tuple[Hashable, int], Tuple[feature, ...]]:
        """Param features grouped by dims."""

        return self._params_by_dims

    def parse_commands(
        self, data: nd.NumDict
    ) -> Dict[Tuple[Hashable, int], Hashable]:
        """
        Determine the value associated with each control dimension.
        
        :param data: A set of features.
        """

        _cmds = set(f for f in self.cmds if f in data)

        cmds, groups = {}, group_by_dims(features=_cmds)
        for k, g in groups.items():
            if len(g) > 1:
                msg = "Received multiple commands for dim '{}'."
                raise ValueError(msg.format(k))
            cmds[k] = g[0].val
        
        for f in self.defaults:
            if f.dim not in cmds:
                cmds[f.dim] = f.val

        return cmds

    def _set_interface_properties(self) -> None:
        """
        Set basic interface properties.
        
        Should minimally set self._cmds, self._defaults, self.flags, and 
        self._params. Other data may be set as necessary.
        """

        raise NotImplementedError()

    def _set_derivative_properties(self) -> None:

        self._cmd_tags = frozenset(f.tag for f in self.cmds)
        self._cmd_dims = frozenset(f.dim for f in self.cmds)

        self._flag_tags = frozenset(f.tag for f in self.flags)
        self._flag_dims = frozenset(f.dim for f in self.flags)

        self._param_tags = frozenset(f.tag for f in self.params)
        self._param_dims = frozenset(f.dim for f in self.params)
       
        self._features = self.cmds | self.params | self.flags

        self._cmds_by_dims = MappingProxyType(group_by_dims(self.cmds))
        self._params_by_dims = MappingProxyType(group_by_dims(self.params))
        self._flags_by_dims = MappingProxyType(group_by_dims(self.flags))

        super()._set_derivative_properties()

    def _validate_interface_properties(self) -> None:

        super()._validate_interface_properties()

        # TODO: Use a more suitable exception class. - Can

        if not self.defaults.issubset(self.cmds):
            raise ValueError("self.defaults not a subset of self.features.")
        
        _defaults_dims = set(f.dim for f in self.defaults)
        
        if not self.cmd_dims.issubset(_defaults_dims):
            raise ValueError("self.defaults conflicts with self.dims.")
        if len(self.cmd_dims) != len(_defaults_dims):
            raise ValueError("multiple defaults assigned to a single dim.")


@dataclass(init=False)
class SimpleDomain(FeatureDomain):
    """
    A simple feature domain, specified through enumeration.
    
    :param features: A collection of features defining the domain.
    """

    def __init__(self, features: Collection[feature]) -> None:

        self._features = frozenset(features)

        self.__post_init__()

    def _validate_data(self) -> None:
        pass

    def _set_interface_properties(self) ->None:
        pass


@dataclass(init=False)
class SimpleInterface(FeatureInterface):
    """A simple feature interface, specified through enumeration."""

    def __init__(
        self, 
        cmds: Collection[feature], 
        defaults: Collection[feature], 
        flags: Collection[feature] = None, 
        params: Collection[feature] = None
    ) -> None:

        self._cmds = frozenset(cmds)
        self._defaults = frozenset(defaults)
        self._flags = frozenset(flags or set())
        self._params = frozenset(params or set())

        self.__post_init__()

    def _validate_data(self) -> None:
        pass

    def _set_interface_properties(self) -> None:
        pass
