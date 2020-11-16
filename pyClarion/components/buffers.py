"""Definitions for memory constructs, most notably working memory."""


__all__ = ["Register", "WorkingMemory"]


from ..base.symbols import (
    Symbol, MatchSet, ConstructType, feature, subsystem, terminus
)
from ..base.components import FeatureInterface, Propagator
from .chunks_ import Chunks, ChunkAdder, ChunkConstructor
from ..utils import simple_junction, group_by_dims, collect_cmd_data

from dataclasses import dataclass
from itertools import chain, product, groupby
from typing import Callable, Hashable, Tuple, NamedTuple, List, Mapping
from types import MappingProxyType
import logging


class Register(Propagator):
    """
    Dynamically stores and activates nodes.
    
    Consists of a node store plus a flag buffer. Stored nodes are persistent, 
    flags are cleared at update time.
    """

    _serves = ConstructType.buffer

    @dataclass
    class Interface(FeatureInterface):
        """
        Control interface for Register instances.
        
        :param tag: Dimension label for controlling write ops to register.
        :param standby: Value corresponding to standby operation.
        :param clear: Value corresponding to clear operation.
        :param channel_map: Tuple pairing values to termini for write 
            operation.
        """

        mapping: Mapping[Hashable, Symbol]
        tag: Hashable
        standby: Hashable
        clear: Hashable

        def _set_interface_properties(self) -> None:
            
            vals = chain((self.standby, self.clear), self.mapping)
            default = feature(self.tag, self.standby)
            
            self._cmds = frozenset(feature(self.tag, val) for val in vals)
            self._defaults = frozenset({default})
            self._params = frozenset()

        def _validate_data(self):
            
            value_set = set(chain((self.standby, self.clear), self.mapping))
            if len(value_set) < len(self.mapping) + 2:
                raise ValueError("Value set may not contain duplicates.") 

    def __init__(
        self, 
        controller: Tuple[subsystem, terminus], 
        source: subsystem,
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

        self.store: list = list() # TODO: Improve type annotation. - Can
        self.flags: list = list()

        self.controller = controller
        self.source = source
        self.interface = interface
        self.filter = filter
        self.level = level

    @property
    def is_empty(self):
        """True iff no nodes are stored in self."""

        return len(self.store) == 0

    def expects(self, construct):
        
        ctl_subsystem, src_subsystem = self.controller[0], self.source

        return construct == ctl_subsystem or construct == src_subsystem

    def call(self, inputs):
        """Activate stored nodes."""

        return {node: self.level for node in chain(self.store, self.flags)}

    def update(self, inputs, output):
        """
        Update the register state.

        Updates are controlled by matching features emitted in the output of 
        self.controller to those defined in self.interface. If no commands are 
        encountered, default/standby behavior will be executed. The default 
        behavior is to maintain the current memory state.
        """

        data = collect_cmd_data(self.client, inputs, self.controller)
        cmds = self.interface.parse_commands(data)

        # Flags get cleared on each update. New flags may then be added for the 
        # next cycle.
        self.clear_flags()

        # There should be exactly one command, but given the structure of the 
        # parse dict, a for loop is used.
        for dim, val in cmds.items():
            if val == self.interface.standby:
                pass
            elif val == self.interface.clear:
                self.clear()
            elif val in self.interface.mapping: 
                channel = self.interface.mapping[val]
                nodes = (
                    node for node in inputs[self.source][channel] if
                    self.filter is None or node in self.filter
                )
                self.write(nodes)

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


class WorkingMemory(Propagator):
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

    _serves = ConstructType.buffer

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

            _w_cmds = frozenset(feature(tag, val) for tag, val in _w_gen)
            _r_cmds = frozenset(feature(tag, val) for tag, val in _r_gen)
            _re_cmds = frozenset(feature(tag, val) for tag, val in _re_gen)

            _w_defaults = frozenset(feature(tag, _w_d_val) for tag in _w_tags)
            _r_defaults = frozenset(feature(tag, _r_d_val) for tag in _r_tags)
            _re_defaults = frozenset({feature(_re_tag, _re_d_val)})
            _defaults = _w_defaults | _r_defaults | _re_defaults

            self._write_tags = _w_tags
            self._read_tags = _r_tags
            self._reset_tag = _re_tag

            self._write_dims = tuple(sorted(set(f.dim for f in _w_cmds)))
            self._read_dims = tuple(sorted(set(f.dim for f in _r_cmds)))
            self._reset_dim = (_re_tag, 0)

            self._cmds = _w_cmds | _r_cmds | _re_cmds
            self._defaults = _defaults
            self._params = frozenset()

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
        controller: Tuple[subsystem, terminus],
        source: subsystem,
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

    def entrust(self, construct):

        for cell in self.cells:
            cell.entrust(construct)
        super().entrust(construct)

    def expects(self, construct):
        
        ctl_subsystem, src_subsystem = self.controller[0], self.source

        return construct == ctl_subsystem or construct == src_subsystem

    def call(self, inputs):
        """
        Activate stored nodes.

        Only activates stored nodes in opened slots.
        """

        data = collect_cmd_data(self.client, inputs, self.controller)
        cmds = self.interface.parse_commands(data)

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
                d_cell = cell.call(inputs)
                d.update(d_cell)
        
        return d

    def update(self, inputs, output):
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

        data = collect_cmd_data(self.client, inputs, self.controller)
        cmds = self.interface.parse_commands(data)

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
            # inputs immutable. - Can
            cell.update(inputs, output)
            # Clearing a slot automatically sets corresponding switch to False.

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
