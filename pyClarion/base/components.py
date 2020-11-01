"""Basic definitions for constructing components."""


__all__ = [
    "Emitter", "Propagator", "Cycle", "Updater", "UpdaterC", "UpdaterS", 
    "Assets", "FeatureInterface"
]


from .symbols import ConstructType, Symbol, feature
from ..utils.funcs import group_by_dims
from abc import abstractmethod
from types import SimpleNamespace, MappingProxyType
from typing import (
    TypeVar, Union, Tuple, Dict, Callable, Hashable, Generic, Any, Optional, 
    Text, Iterator, Iterable, Mapping, ClassVar, List, FrozenSet, cast, 
    no_type_check
)


Inputs = Mapping[Symbol, Any]
Ft = TypeVar("Ft", bound="FeatureInterface")
Pt = TypeVar("Pt", bound="Propagator")


class Component(object):

    _serves: ClassVar[ConstructType]
    _client: Symbol

    @property
    def client(self) -> Symbol:
        """Client construct entrusted to self."""

        return self._client

    def entrust(self, construct: Symbol):
        """Entrust handling of construct to self."""

        if construct.ctype in type(self)._serves:
            self._client = construct
        else:
            msg = "{} cannot serve constructs of type {}."
            name, ctype = type(self).__name__, repr(construct.ctype) 
            raise ValueError(msg.format(name, ctype))

    @abstractmethod
    def expects(self, construct: Symbol):
        """Return True iff self expects input from construct."""

        raise NotImplementedError()


class Emitter(Component):
    """
    Base class for propagating strengths, decisions, etc.

    Emitters define how constructs connect, process inputs, and set outputs.
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
    """Emitter for basic constructs."""

    interface: Ft

    def __call__(self, inputs: Inputs) -> Any:
        """
        Execute construct's forward propagation cycle.

        Pulls expected data from inputs constructs, delegates processing to 
        self.call(), and passes result to self.emit().
        """

        return self.emit(self.call(inputs))

    @abstractmethod
    def call(self, inputs: Inputs) -> Any:
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


class Cycle(Emitter):
    """Emitter for composite constructs."""

    # Specifies data required to construct the output packet
    output: ClassVar[ConstructType] = ConstructType.null_construct
    sequence: Iterable[ConstructType]


class Updater(Component, Generic[Ft]):

    interface: Ft


class UpdaterC(Updater, Generic[Pt]):

    @abstractmethod
    def __call__(
        self, 
        propagator: Pt, 
        inputs: Inputs, 
        output: Any, 
        update_data: Inputs
    ) -> None:
                
        raise NotImplementedError()


class UpdaterS(Updater):

    # TODO: Improve type annotations. - Can

    @abstractmethod
    def __call__(
        self, 
        inputs: Inputs, 
        output: Any, 
        update_data: Inputs
    ) -> None:
        
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


class FeatureInterface(object):
    """
    Control interface for a component.
    
    Defines control features and default values. Provides parsing utilities.
    Each defined feature dimension is interpreted as defining a specific set of 
    alternate actions. A default value must be defined for each dimension, 
    representing the 'do nothing' action.

    Intended to be used as a dataclass.
    """

    has_defaults: ClassVar[bool] = True

    _features: FrozenSet[feature]
    _defaults: FrozenSet[feature]
    _tags: FrozenSet[Hashable]
    _dims: FrozenSet[Tuple[Hashable, int]]

    def __post_init__(self):

        self.compute_properties()

    @property
    def features(self):
        """The set of features defined by self."""
        
        return self._features

    @property
    def defaults(self):
        """Feature, defined by self, indicating default values, if any."""
        
        return self._defaults

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

    def parse_commands(self, data):
        """
        Determine the value associated with each control dimension.
        
        :param data: A set of features.
        """

        _cmds = set(f for f in data if f in self.features)

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

    def compute_properties(self):
        """
        Compute interface properties.
        
        When interfaces are deployed as dataclasses, this method is called 
        after initialization to populate interface properties. 
        
        Must be called again after modifications to interface configuration to 
        ensure that interface properties reflect desired changes..
        """

        self._validate_data()
        self._set_interface_properties()
        self._set_derivative_properties()
        self._validate_interface_properties()

    def _validate_data(self):

        raise NotImplementedError()

    def _set_interface_properties(self):

        raise NotImplementedError()

    def _set_derivative_properties(self):

        self._tags = frozenset(f.tag for f in self.features)
        self._dims = frozenset(f.dim for f in self.features)
        self._features_by_dims = MappingProxyType(group_by_dims(self.features))

    def _validate_interface_properties(self):

        _features_dims = set(f.dim for f in self.features)
        _features_tags = set(f.tag for f in self.features)
        _defaults_dims = set(f.dim for f in self.defaults)

        # TODO: Use a more suitable exception class. - Can

        if self.tags != _features_tags:
            raise ValueError("self.tag conflicts with self.features.")
        if self.dims != _features_dims:
            raise ValueError("self.dims conflicts with self.features.")
        if not self.defaults.issubset(self.features):
            raise ValueError("self.defaults not a subset of self.features.")
        if self.has_defaults and not self.dims.issubset(_defaults_dims):
            raise ValueError("self.defaults conflicts with self.dims.")
        if self.has_defaults and len(self.dims) != len(_defaults_dims):
            raise ValueError("multiple defaults assigned to a single dim.")
        if not self.has_defaults and len(self.defaults) != 0:
            raise ValueError("interface should not expose any defaults.")
