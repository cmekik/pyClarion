"""Basic definitions for constructing components."""


__all__ = [
    "Emitter", "Propagator", "Cycle", "Updater", "UpdaterC", "UpdaterS", 
    "Assets", "FeatureDomain", "FeatureInterface", "SimpleDomain", 
    "SimpleInterface"
]


from .symbols import ConstructType, Symbol, SymbolTrie, feature, group_by_dims
from .. import numdicts as nd

from typing import (
    TypeVar, Union, Tuple, Dict, Callable, Hashable, Generic, Any, Optional, 
    Text, Iterator, Iterable, Mapping, ClassVar, List, FrozenSet, Collection, 
    FrozenSet, AbstractSet, ValuesView, cast, no_type_check,
)
from abc import abstractmethod
from types import SimpleNamespace, MappingProxyType
from dataclasses import dataclass
import warnings
import logging


Ft = TypeVar("Ft", bound="FeatureInterface")
Dt = TypeVar("Dt", bound="FeatureDomain")
Pt = TypeVar("Pt", bound="Propagator")


class Component(object):
    """
    Defines how constructs connect, compute outputs, perform updates etc.
    
    Each pyClarion construct is entrusted to one or more components, each of 
    which is responsible for implementing some function or process associated 
    with the client construct.
    """

    _serves: ClassVar[ConstructType]
    _client: Symbol

    @property
    def client(self) -> Symbol:
        """Client construct entrusted to self."""

        return self._client

    @property
    def expected(self) -> FrozenSet[Symbol]:
        """Constructs from which self expects to receive activations."""

        raise NotImplementedError()

    def entrust(self, construct: Symbol) -> None:
        """Entrust handling of construct to self."""

        if construct.ctype in type(self)._serves:
            self._client = construct
        else:
            msg = "{} cannot serve constructs of type {}."
            name, ctype = type(self).__name__, repr(construct.ctype) 
            raise ValueError(msg.format(name, ctype))

    def expects(self, construct: Symbol) -> bool:
        """Return True iff self expects input from construct."""

        return construct in self.expected

    def check_links(self, linked: Collection[Symbol]) -> bool:
        """Return True iff expected constructs are a subset of linked."""

        return self.expected.issubset(linked)


class Emitter(Component):
    """
    Base class for propagating strengths, decisions, etc.
    
    Emitters are responsible primarily for defining the process(es) by which an 
    entrusted construct computes its output. 
    """

    @abstractmethod
    def emit(self, data: Any = None) -> Any:
        """
        Emit output.

        If no data is passed in, emits a default or null value of the expected
        output type. Otherwise, ensures output is of the expected type and 
        (preferably) immutable before returning the result. 
        """

        raise NotImplementedError()


class Propagator(Emitter, Generic[Ft, Dt]):
    """
    Emitter for basic constructs.
    
    Propagates outputs and performs basic (i.e., essential) updates. The 
    attributes `interface` and `domain` are reserved for feature interfaces and 
    feature domains, respectively.
    """

    interface: Ft
    domain: Dt

    def __call__(self, inputs: SymbolTrie[nd.NumDict]) -> nd.NumDict:
        """
        Execute construct's forward propagation cycle.

        Pulls expected data from inputs constructs, delegates processing to 
        self.call(), and passes result to self.emit().
        """

        return self.emit(self.call(inputs))

    @abstractmethod
    def call(self, inputs: SymbolTrie[nd.NumDict]) -> nd.NumDict:
        """
        Compute construct's output.

        :param inputs: Pairs the names of input constructs with their outputs. 
        """

        raise NotImplementedError()

    def update(
        self, inputs: SymbolTrie[nd.NumDict], output: nd.NumDict
    ) -> None:
        """
        Apply essential updates to self.
        
        Some components require state updates as part of their basic semantics 
        (e.g. memory components). This method is a hook for such essential 
        update routines. 
        """

        pass

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
            d = nd.squeeze(data) # squeezing should not be done here...
            assert type(d) == nd.NumDict, "NumDict subtypes not allowed."
            return d
        else:
            msg = "Expected NumDict instance, got {}"
            raise TypeError(msg.format(type(data).__name__))


class Cycle(Emitter):
    """
    Emitter for composite constructs (i.e., structures).
    
    Defines activation cycles.
    """

    # Specifies data required to construct the output packet
    output: ClassVar[ConstructType] = ConstructType.null_construct
    sequence: Iterable[ConstructType]

    @property
    def expected(self) -> FrozenSet[Symbol]:

        return frozenset()

    @abstractmethod
    def emit(
        self, data: SymbolTrie[nd.NumDict] = None
    ) -> SymbolTrie[nd.NumDict]:
        """
        Emit output.

        If no data is passed in, emits a default or null value of the expected
        output type. Otherwise, ensures output is of the expected type and 
        (preferably) immutable before returning the result. 
        """

        raise NotImplementedError()


class Updater(Component, Generic[Ft]):
    """
    Defines update processes associated with an entrusted construct.

    Updaters implement learning algorithms (e.g., rule extraction) or maintain 
    parameters (e.g., BLA updates) associated with a client construct.
    """

    interface: Ft


class UpdaterC(Updater, Generic[Pt]):
    """
    Updater for basic constructs.

    Performs updates on the propagator associated with the client construct.
    """

    @abstractmethod
    def __call__(
        self, 
        propagator: Pt, 
        inputs: SymbolTrie[nd.NumDict], 
        output: nd.NumDict, 
        update_data: SymbolTrie[nd.NumDict]
    ) -> None:
        """
        Apply updates to propagator. 
        
        Assumes that propagator is associated with client construct.

        :param propagator: The client construct's propagator.
        :param inputs: Inputs seen by client construct on current activation 
            cycle.
        :param output: The current output of the client construct as computed 
            by the client's propagator.
        :param update_data: Current outputs of relevant constructs (as defined 
            by self.expects()). 
        """
                
        raise NotImplementedError()


class UpdaterS(Updater):
    """
    Updater for composite constructs (i.e., structures).

    Performs updates on shared assets associated with the client construct.
    """

    # TODO: Improve type annotations. - Can

    @abstractmethod
    def __call__(
        self, 
        inputs: SymbolTrie[nd.NumDict], 
        output: SymbolTrie[nd.NumDict], 
        update_data: SymbolTrie[nd.NumDict]
    ) -> None:
        """
        Apply updates to relevant asset(s) of client construct. 
        
        Assumes that self is initialized with references to the relevant 
        assets.

        :param inputs: Inputs seen by client construct on current activation 
            cycle.
        :param output: The current output of the client construct as computed 
            by the client's propagator.
        :param update_data: Current outputs of relevant constructs (as defined 
            by self.expects()). 
        """

        raise NotImplementedError()


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
    """A simple feature domain, specified through enumeration."""

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
