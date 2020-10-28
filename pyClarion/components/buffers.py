"""Definitions for memory constructs, most notably working memory."""


__all__ = ["Bus", "Register", "WorkingMemory"]


from pyClarion.base.symbols import Symbol, MatchSet, ConstructType, feature
from pyClarion.components.propagators import PropagatorB
from pyClarion.components.chunks_ import Chunks, ChunkAdder, ChunkConstructor
from pyClarion.utils import simple_junction, group_by_dims

from itertools import chain, product, groupby
from typing import Callable, Hashable, Tuple, NamedTuple, List


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

    class Interface(NamedTuple):
        """
        Control interface for Register instances.
        
        :param dlb: Dimension label for controlling write ops to register.
        :param standby: Value corresponding to standby operation.
        :param clear: Value corresponding to clear operation.
        :param channel_map: Tuple pairing values to terminuses for write 
            operation.
        """

        dlb: Hashable
        standby: Hashable
        clear: Hashable
        channel_map: Tuple[Tuple[Hashable, Symbol], ...]

        @property
        def channels(self):

            return tuple(channel for channel, _ in self.channel_map)

        @property
        def features(self):
            """
            Tuple listing all control features associated with self.
            
            Features are listed in the following order:
                standby, clear, channel_1, ..., channel_n
            """

            dlb = self.dlb
            vals = chain((self.standby,), (self.clear,), self.channels)

            return tuple(feature(dlb, val) for val in vals)

        # @property
        # def dim(self):

        #     return (dlb, 0)

        @property
        def defaults(self):

            return (feature(self.dlb, self.standby),)

    class InterfaceError(Exception):
        """Raised when a passed a malformed interface."""
        pass

    @classmethod
    def _validate_interface(cls, interface):

        channel_vals = [v for v, s in interface.channel_map]
        if len(set(channel_vals)) != len(channel_vals):
            raise cls.InterfaceError(
                "Arg `channel_map` may not contain duplicate values."
            )

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
        self._validate_interface(interface)

        super().__init__()
        self.store: list = list() # TODO: Improve type annotation. - Can
        self.flags: list = list()

        self.controller = controller
        self.source = source
        self.interface = interface
        self.filter = filter
        self.level = level

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
        else: # cmd.val in self.interface.channels
            channel = self._channel_dict[cmd.val]
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
            cmds = set(self.interface.defaults)
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

    @property
    def is_empty(self):
        """True if no nodes are stored in self."""

        return len(self.store) == 0

    @property
    def interface(self):

        return self._interface

    @interface.setter
    def interface(self, obj: Interface):

        self._interface = obj
        self._channel_dict = dict(obj.channel_map)


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

    class Interface(NamedTuple):
        """
        Control interface for WorkingMemory propagator.

        :param dims: Dimensions for controlling WM slot write operations.
        :param standby: Value for standby action on writing operations.
        :param clear: Value for clear action on writing operations.
        :param channel_map: Tuple pairing a write operation value with a 
            terminus from the source subsystem. Signals WM to write contents of 
            terminus to a slot.
        :param reset_dim: Dimension for controlling global WM state resets.
        :param reset_vals: Global reset control values. First value corresponds 
            to standby. Second value corresponds to reset initiation.
        :param switch_dims: Dimension for controlling WM slot read operations.
        :param switch_vals: Read operation control values. First value 
            corresponds to standby (i.e., no read), second value to read action. 
        """

        write_dlbs: Tuple[Hashable, ...]
        standby: Hashable
        clear: Hashable
        channel_map: Tuple[Tuple[Hashable, Symbol], ...]

        reset_dlb: Hashable
        reset_vals: Tuple[Hashable, Hashable]

        read_dlbs: Tuple[Hashable, ...]
        read_vals: Tuple[Hashable, Hashable]

        @property
        def channels(self):

            return tuple(channel for channel, _ in self.channel_map)

        # TODO: Cache output for better efficiency? - Can
        @property
        def features(self):
            """Tuple listing all interface features."""

            w_dlbs = self.write_dlbs
            w_vals = chain((self.standby,), (self.clear,), self.channels)

            r_dlbs = self.read_dlbs
            r_vals = self.read_vals

            w = tuple(feature(dlb, val) for dlb, val in product(w_dlbs, w_vals))
            r = tuple(feature(self.reset_dlb, val) for val in self.reset_vals)
            s = tuple(feature(dlb, val) for dlb, val in product(r_dlbs, r_vals))

            return w + r + s

        @property
        def write_dims(self):

            return tuple((dlb, 0) for dlb in self.write_dlbs)

        @property
        def reset_dim(self):

            return (self.reset_dlb, 0)

        @property
        def read_dims(self):

            return tuple((dlb, 0) for dlb in self.read_dlbs)

        @property
        def dims(self):

            return self.write_dims + (self.reset_dim,) + self.read_dims

        @property
        def defaults(self):
            """Tuple listing default action features."""

            # 'w' for 'write', 'r' for 'reset', 's' for 'switch' 
            
            stby_w = self.standby
            stby_r = self.reset_vals[0]
            stby_s = self.read_vals[0]

            w_defaults = tuple(feature(dim, stby_w) for dim in self.write_dlbs)
            r_defaults = (feature(self.reset_dlb, stby_r),)
            s_defaults = tuple(feature(dim, stby_s) for dim in self.read_dlbs)

            return w_defaults + r_defaults + s_defaults

    # TODO: Add validation checks for interface. - Can

    class InterfaceError(Exception):
        """Raised when a passed a malformed interface."""
        pass

    @classmethod
    def _validate_interface(cls, interface: Interface) -> None:

        if len(interface.write_dlbs) != len(interface.read_dlbs):
            msg = "Len of write_dlbs and read_dlbs must match."
            raise cls.InterfaceError(msg) 

        if len(set(interface.write_dlbs)) != len(interface.write_dlbs):
            raise cls.InterfaceError("dims may not contain duplicates.")

        channel_vals = [v for v, s in interface.channel_map]
        if len(set(channel_vals)) != len(channel_vals):
            raise cls.InterfaceError(
                "Arg `channel_map` may not contain duplicate values."
            )

        if len(set(interface.read_dlbs)) != len(interface.read_dlbs):
            raise cls.InterfaceError("switch_dims may not contain duplicates.")

        if len(set(interface.reset_vals)) != 2:
            raise cls.InterfaceError("Must provide two distinct reset_vals")

        if len(set(interface.reset_vals)) != 2:
            raise cls.InterfaceError("Must provide two distinct switch_vals")

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
        self._validate_interface(interface)

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
                    dlb=dlb,
                    standby=interface.standby,
                    clear=interface.clear,
                    channel_map=interface.channel_map
                ),
                filter=filter,
                level=level
            )
            for dlb in interface.write_dlbs
        )

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
            raw_cmds = set(self.interface.defaults)
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
                raise ValueError(
                "Multiple commands for dim '{}' in WorkingMemory.".format(k)
            )
            cmds[k] = g[0].val

        return cmds
