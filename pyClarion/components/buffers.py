"""Definitions for memory constructs, most notably working memory."""


__all__ = ["Bus", "Register", "WorkingMemory"]


from pyClarion.base.symbols import Symbol, MatchSet, ConstructType, feature
from pyClarion.base.realizers import FeatureInterface
from pyClarion.components.propagators import PropagatorB
from pyClarion.components.chunks_ import Chunks, ChunkAdder, ChunkConstructor
from pyClarion.utils import simple_junction, group_by_dims

from dataclasses import dataclass
from itertools import chain, product, groupby
from typing import Callable, Hashable, Tuple, NamedTuple, List, Mapping
from types import MappingProxyType


class Bus(PropagatorB):
    """
    Bus for commands & parameters.
    
    Relays commands issued at some terminus of a controller subsystem to any 
    listeners.
    """

    def __init__(
        self,
        source: Tuple[Symbol, Symbol],
        filter: MatchSet = None,
        level: float = 1.0
    ) -> None:
        
        super().__init__()
        
        self.source = source
        self.filter = filter
        self.level = level

    def expects(self, construct):

        return construct == self.source[0]

    def call(self, construct, inputs):

        subsystem, terminus = self.source
        data = inputs[subsystem][terminus]
        d = {node: self.level for node in data if node in self.filter}

        return d


class Register(PropagatorB):
    """
    Dynamically stores and activates nodes.
    
    Consists of a node store plus a flag buffer. Stored nodes are persistent, 
    flags are cleared at update time.
    """

    @dataclass
    class Interface(FeatureInterface):
        """
        Control interface for Register instances.
        
        :param tag: Dimension label for controlling write ops to register.
        :param standby: Value corresponding to standby operation.
        :param clear: Value corresponding to clear operation.
        :param channel_map: Tuple pairing values to terminuses for write 
            operation.
        """

        mapping: Mapping[Hashable, Symbol]
        tag: Hashable
        standby: Hashable
        clear: Hashable

        def __post_init__(self):

            self._validate_data()
            self._set_interface_properties()

        def _set_interface_properties(self) -> None:
            
            vals = chain((self.standby, self.clear), self.mapping)
            default = feature(self.tag, self.standby)
            
            self._features = frozenset(feature(self.tag, val) for val in vals)
            self._defaults = MappingProxyType({default.dim: default})
            self._tags = frozenset(f.tag for f in self._features)
            self._dims = frozenset(f.dim for f in self._features)

        def _validate_data(self):
            
            value_set = set(chain((self.standby, self.clear), self.mapping))
            if len(value_set) < len(self.mapping) + 2:
                raise ValueError("Value set may not contain duplicates.") 

    def __init__(
        self, 
        controller: Tuple[Symbol, Symbol], 
        source: Symbol,
        interface: Interface,
        filter: MatchSet = None,
        level: float = 1.0
    ) -> None:
        """
        Initialize a Register instance.

        :param controller: Reference for construct issuing commands to self.
        :param source: Reference for construct from which to pull data.
        :param interface: Defines features for controlling updates to self.
        :param filter: Optional filter for state updates.
        :param level: Output activation level for stored features.
        """

        self._validate_controller(controller)
        self._validate_source(source)

        super().__init__()
        self.store: list = list() # TODO: Improve type annotation. - Can
        self.flags: list = list()

        self.controller = controller
        self.source = source
        self.interface = interface
        self.filter = filter
        self.level = level

    @property
    def is_empty(self):
        """True if no nodes are stored in self."""

        return len(self.store) == 0

    def expects(self, construct):
        
        ctl_subsystem, src_subsystem = self.controller[0], self.source

        return construct == ctl_subsystem or construct == src_subsystem

    def call(self, construct, inputs):
        """Activate stored nodes."""

        return {node: self.level for node in chain(self.store, self.flags)}

    def update(self, construct, inputs):

        # Flags get cleared on each update. New flags may then be added for the 
        # next cycle.
        self.clear_flags()

        cmds = self._parse_commands(inputs)
        cmd = cmds.pop()
        if cmd.val == self.interface.standby:
            pass
        elif cmd.val == self.interface.clear:
            self.clear()
        else: # cmd.val in self.interface.mapping
            channel = self.interface.mapping[cmd.val]
            nodes = (
                node for node in inputs[self.source][channel] if
                self.filter is None or node in self.filter
            )
            self.write(nodes)

    def _parse_commands(self, inputs):

        ctl_subsystem, ctl_terminus = self.controller
        _cmds = inputs[ctl_subsystem][ctl_terminus]
        cmds = {cmd for cmd in _cmds if cmd in self.interface.features}

        if len(cmds) > 1:
            raise ValueError("Multiple commands received.")
        elif len(cmds) == 0:
            cmds = set(self.interface.defaults.values())
        else:
            pass

        return cmds

    def write(self, nodes):
        """Write nodes to self.store."""

        self.store.clear()
        self.store.extend(nodes)

    def clear(self):
        """Clear any nodes stored in self."""

        self.store.clear()

    def write_flags(self, *flags):

        self.flags.extend(flags)

    def clear_flags(self):

        self.flags.clear()

    @staticmethod
    def _validate_controller(controller):

        subsystem, terminus = controller
        if subsystem.ctype not in ConstructType.subsystem:
            raise ValueError(
                "Arg `controller` must name a subsystem at index 0."
            )
        if terminus.ctype not in ConstructType.terminus:
            raise ValueError(
                "Arg `controller` must name a terminus at index 1."
            )

    @staticmethod
    def _validate_source(source):

        if source.ctype not in ConstructType.subsystem:
            raise ValueError("Arg `source` must name a subsystem.")


class WorkingMemory(PropagatorB):
    """
    A simple working memory mechanism.

    The mechanism follows a slot-based storage and control architecture. It 
    supports writing data to slots, clearing slots, excluding slots from the 
    output and resetting the memory state. 

    This class defines the basic datastructure and memory update method. For 
    minimality, it does not report mechanism states (e.g., which slots are 
    filled).
    """

    # TODO: In the future, WorkingMemory should return a special flag feature 
    # if an empty slot is opened to signify retrieval failure from that slot. 
    # This requires extensions to the interface. - Can

    @dataclass
    class Interface(FeatureInterface):
        """
        Control interface for WorkingMemory propagator.

        :param slots: Number of working memory slots.
        :param prefix: Marker for identifying this particular set of control 
            features.
        :param write_marker: Marker for controlling WM slot write operations.
        :param read_marker: Marker for controlling WM slot read operations.
        :param reset_marker: Marker for controlling global WM state resets.
        :param standby: Value for standby action on writing operations.
        :param clear: Value for clear action on writing operations.
        :param mapping: Mapping pairing a write operation value with a 
            terminus from the source subsystem. Signals WM to write contents of 
            terminus to a slot.
        :param reset_vals: Global reset control values. First value corresponds 
            to standby. Second value corresponds to reset initiation.
        :param read_vals: Read operation control values. First value 
            corresponds to standby (i.e., no read), second value to read action. 
        """

        slots: int
        prefix: Hashable
        write_marker: Hashable
        read_marker: Hashable
        reset_marker: Hashable
        standby: Hashable
        clear: Hashable
        mapping: Mapping[Hashable, Symbol]
        reset_vals: Tuple[Hashable, Hashable]
        read_vals: Tuple[Hashable, Hashable]

        def __post_init__(self):

            self._validate_data()
            self._set_interface_properties()

        @property
        def write_tags(self):

            return self._write_tags

        @property
        def read_tags(self):

            return self._read_tags

        @property
        def reset_tag(self):

            return self._reset_tag

        @property
        def write_dims(self):

            return self._write_dims

        @property
        def read_dims(self):

            return self._read_dims

        @property
        def reset_dim(self):

            return self._reset_dim

        def _set_interface_properties(self) -> None:
            
            slots, pre = self.slots, self.prefix
            w, r, re = self.write_marker, self.read_marker, self.reset_marker

            _w_tags = tuple((pre, w, i) for i in range(slots))
            _r_tags = tuple((pre, r, i) for i in range(slots))
            _re_tag = (pre, re)

            _w_vals = set(chain((self.standby, self.clear), self.mapping))
            _r_vals = self.read_vals
            _re_vals = self.reset_vals

            _w_d_val = self.standby
            _r_d_val = _r_vals[0]
            _re_d_val = _re_vals[0]

            _w_gen = ((tag, val) for tag, val in product(_w_tags, _w_vals))
            _r_gen = ((tag, val) for tag, val in product(_r_tags, _r_vals))
            _re_gen = ((_re_tag, val) for val in _re_vals)

            _w_dgen = ((tag, val) for tag, val in product(_w_tags, _w_vals))
            _r_dgen = ((tag, val) for tag, val in product(_r_tags, _r_vals))
            _re_dgen = ((_re_tag, val) for val in _re_vals)

            _w_features = frozenset(feature(tag, val) for tag, val in _w_gen)
            _r_features = frozenset(feature(tag, val) for tag, val in _r_gen)
            _re_features = frozenset(feature(tag, val) for tag, val in _re_gen)

            _w_defaults = set(feature(tag, _w_d_val) for tag in _w_tags)
            _r_defaults = set(feature(tag, _r_d_val) for tag in _r_tags)
            _re_defaults = {feature(_re_tag, _re_d_val),}
            _defaults = _w_defaults | _r_defaults | _re_defaults

            self._write_tags = _w_tags
            self._read_tags = _r_tags
            self._reset_tag = _re_tag

            self._write_dims = tuple(sorted(f.dim for f in _w_features))
            self._read_dims = tuple(sorted(f.dim for f in _r_features))
            self._reset_dim = (_re_tag, 0)

            self._features = _w_features | _r_features | _re_features
            self._defaults = MappingProxyType({f.dim: f for f in _defaults})
            self._tags = frozenset(f.tag for f in self._features)
            self._dims = frozenset(f.dim for f in self._features)

        def _validate_data(self):
            
            markers = (self.write_marker, self.read_marker, self.reset_marker)
            w_vals = set(chain((self.standby, self.clear), self.mapping))
            
            if len(set(markers)) < 3:
                raise ValueError("Marker arguments must be mutually distinct.")
            if len(set(w_vals)) < len(self.mapping) + 2:
                raise ValueError("Write vals may not contain duplicates.")
            if len(set(self.read_vals)) < len(self.read_vals):
                raise ValueError("Read vals may not contain duplicates.")
            if len(set(self.reset_vals)) < len(self.reset_vals):
                raise ValueError("Reset vals may not contain duplicates.")

    def __init__(
        self,
        controller: Tuple[Symbol, Symbol],
        source: Symbol,
        interface: Interface,
        level: float = 1.0,
        filter: MatchSet = None
    ) -> None:
        """
        Initialize a new WorkingMemory instance.

        :param controller: Reference for construct issuing commands to self.
        :param source: Reference for construct from which to pull data.
        :param interface: Defines features for controlling updates to self.
        :param filter: Optional filter for state updates.
        :param level: Output activation level for stored features.
        """

        self._validate_controller(controller)
        self._validate_source(source)

        self.controller = controller
        self.source = source
        self.interface = interface
        self.level = level

        self.flags: List[Symbol] = []
        self.cells = tuple(
            Register(
                controller=controller,
                source=source,
                interface=Register.Interface(
                    mapping=interface.mapping,
                    tag=tag,
                    standby=interface.standby,
                    clear=interface.clear
                ),
                filter=filter,
                level=level
            )
            for tag in interface.write_tags
        )

    def call(self, construct, inputs):
        """
        Activate stored nodes.

        Only activates stored nodes in opened slots.
        """

        # Could possibly also use collections.ChainMap. Need to check how it 
        # will interact w/ downstream processing. - Can
        
        cmds = self._parse_commands(inputs)

        # toggle switches
        switches = []
        for slot, dim in enumerate(self.interface.read_dims):
            if dim in cmds:
                val = cmds[dim]
                switch = (val == self.interface.read_vals[1])
                switches.append(switch)
                    
        d = {f: self.level for f in self.flags}
        for switch, cell in zip(switches, self.cells):
            if switch is True:
                d_cell = cell.call(construct, inputs.copy())
                d.update(d_cell)
        
        return d

    def expects(self, construct):
        
        ctl_subsystem, src_subsystem = self.controller[0], self.source

        return construct == ctl_subsystem or construct == src_subsystem

    def reset(self):
        """
        Reset memory state.
        
        Clears all memory slots and closes all switches.
        """

        for cell in self.cells:
            cell.clear()

    def write_flags(self, *flags):

        self.flags.extend(flags)

    def clear_flags(self):

        self.flags.clear()

    def update(self, construct, inputs):
        """
        Update the memory state.

        Updates are controlled by matching features emitted in the output of 
        self.controller to those defined in self.interface. If no commands are 
        encountered, default/standby behavior will be executed. The default 
        behavior is to maintain the current memory state.
        
        The update cycle processes global resets first, slot contents are 
        updated next. As a result, it is possible to clear the memory globally 
        and populate it with new information (e.g., in the service of a new 
        goal) in one single update. 
        """

        cmds = self._parse_commands(inputs)

        # Flags get cleared on each update. New flags may then be added for the 
        # next cycle.
        self.clear_flags()

        # global wm reset
        if self.interface.reset_dim in cmds:
            val = cmds[self.interface.reset_dim]
            if self.interface.reset_vals[1] == val:
                self.reset()

        # cell/slot updates
        for slot, cell in enumerate(self.cells):
            # The copy here is for safety... better option may be to make 
            # inputs immutable. - Cans
            cell.update(construct, inputs.copy())
            # Clearing a slot automatically sets corresponding switch to False.

    def _parse_commands(self, inputs):

        subsystem, terminus = self.controller
        try:
            raw_cmds = inputs[subsystem][terminus]
        except KeyError:
            # if command interface cannot be found, assume default commands.
            raw_cmds = set(self.interface.defaults.values())
            # TODO: This should publish a warning in a log; remember to do this 
            # when setting up logging. - Can

        # Filter irrelevant feature symbols
        cmd_set = set(
            f for f in raw_cmds if f in self.interface.features and (
                f.dim == self.interface.reset_dim or
                f.dim in self.interface.read_dims
            )
        )

        groups = group_by_dims(features=cmd_set)
        cmds = {}
        for k, g in groups.items():
            if len(g) > 1:
                msg = "Multiple commands for dim '{}' in WorkingMemory."
                raise ValueError(msg.format(k))
            cmds[k] = g[0].val

        return cmds

    @staticmethod
    def _validate_controller(controller):

        subsystem, terminus = controller
        if subsystem.ctype not in ConstructType.subsystem:
            msg = "Arg `controller` must name a subsystem at index 0."
            raise ValueError(msg)
        if terminus.ctype not in ConstructType.terminus:
            msg = "Arg `controller` must name a terminus at index 1."
            raise ValueError(msg)

    @staticmethod
    def _validate_source(source):

        if source.ctype not in ConstructType.subsystem:
            raise ValueError("Arg `source` must name a subsystem.")
