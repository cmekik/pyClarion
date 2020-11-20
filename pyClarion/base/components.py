"""Basic definitions for constructing components."""


__all__ = [
    "Emitter", "Propagator", "Cycle", "Updater", "UpdaterC", "UpdaterS", 
    "Assets", "FeatureDomain", "FeatureInterface", "SimpleDomain", 
    "SimpleInterface"
]


from .symbols import ConstructType, Symbol, feature, group_by_dims
from .numdicts import NumDict, FrozenNumDict

from typing import (
    TypeVar, Union, Tuple, Dict, Callable, Hashable, Generic, Any, Optional, 
    Text, Iterator, Iterable, Mapping, ClassVar, List, FrozenSet, Collection, 
    cast, no_type_check
)
from abc import abstractmethod
from types import SimpleNamespace, MappingProxyType
from dataclasses import dataclass
import logging


Inputs = Mapping[Symbol, Any]
Ft = TypeVar("Ft", bound="FeatureInterface")
Pt = TypeVar("Pt", bound="Propagator")


class Component(object):
    """
    Defines how constructs connect, compute outputs, perform updates etc.
    
    Each construct pyClarion construct is entrusted to one or more components, 
    each of which is responsible for implementing some function or process 
    associated with the client construct.
    """

    _serves: ClassVar[ConstructType]
    _client: Symbol

    @property
    def client(self) -> Symbol:
        """Client construct entrusted to self."""

        return self._client

    def entrust(self, construct: Symbol) -> None:
        """Entrust handling of construct to self."""

        if construct.ctype in type(self)._serves:
            self._client = construct
        else:
            msg = "{} cannot serve constructs of type {}."
            name, ctype = type(self).__name__, repr(construct.ctype) 
            raise ValueError(msg.format(name, ctype))

    @abstractmethod
    def expects(self, construct: Symbol) -> bool:
        """Return True iff self expects input from construct."""

        raise NotImplementedError()


class Emitter(Component):
    """
    Base class for propagating strengths, decisions, etc.
    
    Emitters are responsible primarily for defining the process(es) by which an 
    entrusted construct computes its output. 
    """

    @staticmethod
    @abstractmethod
    def emit(data: Any = None) -> Any:
        """
        Emit output.

        If no data is passed in, emits a default or null value of the expected
        output type. Otherwise, ensures output is of the expected type and 
        (preferably) immutable before returning the result. 
        """

        raise NotImplementedError()


class Propagator(Emitter, Generic[Ft]):
    """
    Emitter for basic constructs.
    
    Propagates outputs and performs basic (i.e., essential) updates.
    """

    interface: Ft

    def __call__(self, inputs: Inputs) -> Mapping:
        """
        Execute construct's forward propagation cycle.

        Pulls expected data from inputs constructs, delegates processing to 
        self.call(), and passes result to self.emit().
        """

        return self.emit(self.call(inputs))

    @abstractmethod
    def call(self, inputs: Inputs) -> NumDict:
        """
        Compute construct's output.

        :param construct: Name of the client construct. 
        :param inputs: Pairs the names of input constructs with their outputs. 
        :param kwds: Optional parameters. Propagator instances are recommended 
            to throw errors upon receipt of unexpected keywords.
        """

        raise NotImplementedError()

    def update(self, inputs: Inputs, output: Any) -> None:
        """
        Apply essential updates to self.
        
        Some components require state updates as part of their basic semantics 
        (e.g. memory components). This method is a hook for such essential 
        update routines. 
        """

        pass

    @staticmethod
    def emit(data: NumDict = None) -> FrozenNumDict:
        """
        Emit output.

        If data is None, emits an empty frozen numdict. Otherwise, emits a 
        frozen version of data.
        """

        d = data if data is not None else NumDict()
        d.squeeze()

        return FrozenNumDict(d)


class Cycle(Emitter):
    """
    Emitter for composite constructs (i.e., structures).
    
    Defines activation cycles.
    """

    # Specifies data required to construct the output packet
    output: ClassVar[ConstructType] = ConstructType.null_construct
    sequence: Iterable[ConstructType]


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
        inputs: Inputs, 
        output: Any, 
        update_data: Inputs
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
        inputs: Inputs, 
        output: Any, 
        update_data: Inputs
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
    simulation with hooks for parsing out dimensions, tags, etc.
    
    Intended to be used as a dataclass.
    """

    _features: FrozenSet[feature]
    _tags: FrozenSet[Hashable]
    _dims: FrozenSet[Tuple[Hashable, int]]

    def __post_init__(self):

        self.compute_properties()

    @property
    def features(self):
        """The set of features defined by self."""
        
        return self._features

    @property
    def tags(self):
        """The set of dimensional labels defined by self."""
        
        return self._tags

    @property
    def dims(self):
        """The set of feature dimensions (w/ lags) defined by self."""
        
        return self._dims

    @property
    def features_by_dims(self):
        """Features grouped by dims."""

        return self._features_by_dims

    def compute_properties(self):
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

    def _validate_data(self):

        # Should throw errors if dataclass members violate assumptions or 
        # requirements

        raise NotImplementedError()

    def _set_interface_properties(self):

        # Should minimally set self._features. Other properties may be set as 
        # necessary.

        raise NotImplementedError()

    def _set_derivative_properties(self):

        self._tags = frozenset(f.tag for f in self.features)
        self._dims = frozenset(f.dim for f in self.features)
        self._features_by_dims = MappingProxyType(group_by_dims(self.features))

    def _validate_interface_properties(self):

        _features_dims = set(f.dim for f in self.features)
        _features_tags = set(f.tag for f in self.features)

        # TODO: Use a more suitable exception class. - Can

        if self.tags != _features_tags:
            raise ValueError("self.tag conflicts with self.features.")
        if self.dims != _features_dims:
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
    _params: FrozenSet[feature]

    @property
    def cmds(self):
        """Features, defined by self, indicating actions."""

        return self._cmds

    @property
    def defaults(self):
        """Features, defined by self, indicating default action values."""
        
        return self._defaults

    @property
    def params(self):
        """Features, defined by self, indicating action parameters."""

        return self._params

    @property
    def cmd_tags(self):
        """The set of command dimensional labels defined by self."""
        
        return self._cmd_tags

    @property
    def cmd_dims(self):
        """The set of command feature dimensions (w/ lags) defined by self."""
        
        return self._cmd_dims

    @property
    def param_tags(self):
        """The set of param dimensional labels defined by self."""
        
        return self._param_tags

    @property
    def param_dims(self):
        """The set of param feature dimensions (w/ lags) defined by self."""
        
        return self._param_dims

    @property
    def cmds_by_dims(self):

        return self._cmds_by_dims

    @property
    def params_by_dims(self):

        return self._params_by_dims

    def parse_commands(self, data):
        """
        Determine the value associated with each control dimension.
        
        :param data: A set of features.
        """

        _cmds = set(f for f in data if f in self.cmds)

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

    def _set_interface_properties(self):

        # Should minimally set self._cmds, self._defaults, and self._params.
        # Other properties may be set as necessary.

        raise NotImplementedError()

    def _set_derivative_properties(self):

        self._cmd_tags = frozenset(f.tag for f in self.cmds)
        self._cmd_dims = frozenset(f.dim for f in self.cmds)

        self._param_tags = frozenset(f.tag for f in self.params)
        self._param_dims = frozenset(f.dim for f in self.params)
       
        self._features = self.cmds | self.params

        self._cmds_by_dims = MappingProxyType(group_by_dims(self.cmds))
        self._params_by_dims = MappingProxyType(group_by_dims(self.params))

        super()._set_derivative_properties()

    def _validate_interface_properties(self):

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

    def __init__(self, features):

        self._features = frozenset(features)

        self.__post_init__()

    def _validate_data(self):

        pass

    def _set_interface_properties(self):

        pass


@dataclass(init=False)
class SimpleInterface(FeatureInterface):
    """A simple feature interface, specified through enumeration."""

    def __init__(self, cmds, defaults, params):

        self._cmds = frozenset(cmds)
        self._defaults = frozenset(defaults)
        self._params = frozenset(params)

        self.__post_init__()

    def _validate_data(self):

        pass

    def _set_interface_properties(self):

        pass
