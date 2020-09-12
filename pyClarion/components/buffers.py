"""Definitions for memory constructs, most notably working memory."""


__all__ = ["Bus", "Register", "WorkingMemory"]


from pyClarion.base.symbols import Symbol, MatchSet, ConstructType, feature
from pyClarion.components.propagators import PropagatorB
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

    def call(self, construct, inputs, **kwds):

        subsystem, terminus = self.source
        data = inputs[subsystem][terminus]
        d = {node: self.level for node in data if node in self.filter}

        return d


class Register(PropagatorB):
    """Dynamically stores and activates nodes."""

    class Interface(NamedTuple):
        """Control interface for Register instances."""

        dim: Hashable
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

            dim = self.dim
            vals = chain((self.standby,), (self.clear,), self.channels)

            return tuple(feature(dim, val) for val in vals)

        @property
        def defaults(self):

            return (feature(self.dim, self.standby),)

    def __init__(
        self, 
        controller: Tuple[Symbol, Symbol], 
        source: Symbol,
        interface: Interface,
        filter: MatchSet = None,
        level: float = 1.0
    ) -> None:
        """
        Initialize a SimpleMemory instance.

        :param controller: Reference for construct issuing commands to self.
        :param source: Reference for construct from which to pull data.
        :param interface: Defines features for controlling updates to self.
        :param filter: Optional filter for state updates.
        :param level: Output activation level for stored features.
        """

        self._validate_init_args(controller, source, interface, filter, level)

        super().__init__()
        self.store: list = list() # TODO: Improve type annotation. - Can

        self.controller = controller
        self.source = source
        self.interface = interface
        self.filter = filter
        self.level = level

    @staticmethod
    def _validate_init_args(controller, source, interface, filter, level):

        # TODO: Add more comprehensive validation? - Can

        ctl_subsystem, ctl_terminus = controller
        if ctl_subsystem.ctype not in ConstructType.subsystem:
            raise ValueError(
                "Arg `controller` does not name a subsystem at index 0."
            )
        if ctl_terminus.ctype not in ConstructType.terminus:
            raise ValueError(
                "Arg `controller` must name a terminus at index 1."
            )
        if source.ctype not in ConstructType.subsystem:
            raise ValueError("Arg `source` must name a subsystem.")

    def expects(self, construct):
        
        ctl_subsystem, src_subsystem = self.controller[0], self.source

        return construct == ctl_subsystem or construct == src_subsystem

    def call(self, construct, inputs, **kwds):
        """Activate stored nodes."""

        return {node: self.level for node in self.store}

    def update(self, construct, inputs):

        cmds = self._parse_commands(inputs)
        cmd = cmds.pop()
        if cmd.val == self.interface.standby:
            pass
        elif cmd.val == self.interface.clear:
            self.store.clear()
        else: # cmd.val in self.interface.channels
            channel = self._channel_dict[cmd.val]
            nodes = (
                node for node in inputs[self.source][channel] if
                self.filter is None or node in self.filter
            )
            self.store.clear()
            self.store.extend(nodes)

    def _parse_commands(self, inputs):

        ctl_subsystem, ctl_terminus = self.controller
        _cmds = inputs[ctl_subsystem][ctl_terminus]
        cmds = {cmd for cmd in _cmds if cmd in self.interface.features}

        if len(cmds) > 1:
            raise ValueError("Multiple commands received.")
        elif len(cmds) == 0:
            raise ValueError("No command received.")
        else:
            pass

        return cmds

    def clear(self):
        """Clear any nodes stored in self."""

        self.store.clear()

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
    minimality, it does not report mechanism states (i.e., which slots are 
    filled, which slots are open for emitting output etc.).
    """

    class Interface(NamedTuple):
        """Represents control interface for WorkingMemory propagators."""

        dims: Tuple[Hashable, ...]
        standby: Hashable
        clear: Hashable
        channel_map: Tuple[Tuple[Hashable, Symbol], ...]

        reset_dim: Hashable
        reset_vals: Tuple[Hashable, Hashable]

        switch_dims: Tuple[Hashable, ...]
        switch_vals: Tuple[Hashable, Hashable]

        @property
        def channels(self):

            return tuple(channel for channel, _ in self.channel_map)

        # TODO: Cache output for better efficiency? - Can
        @property
        def features(self):
            """Tuple listing all interface features."""

            # 'w' for 'write'
            w_dims = self.dims
            w_vals = chain((self.standby,), (self.clear,), self.channels)

            # 's' for 'switch'
            s_dims = self.switch_dims
            s_vals = self.switch_vals

            w = tuple(feature(dim, val) for dim, val in product(w_dims, w_vals))
            r = tuple(feature(self.reset_dim, val) for val in self.reset_vals)
            s = tuple(feature(dim, val) for dim, val in product(s_dims, s_vals))

            return w + r + s

        @property
        def defaults(self):
            """Tuple listing default action features."""

            # 'w' for 'write', 'r' for 'reset', 's' for 'switch' 
            
            stby_w = self.standby
            stby_r = self.reset_vals[0]
            stby_s = self.switch_vals[0]

            w_defaults = tuple(feature(dim, stby_w) for dim in self.dims)
            r_defaults = (feature(self.reset_dim, stby_r),)
            s_defaults = tuple(feature(dim, stby_s) for dim in self.switch_dims)

            return w_defaults + r_defaults + s_defaults

    # TODO: Add validation checks for interface. - Can

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

        self.controller = controller
        self.source = source
        self.interface = interface
        self.level = level

        self.cells = tuple(
            Register(
                controller=controller,
                source=source,
                interface=Register.Interface(
                    dim=dim,
                    standby=interface.standby,
                    clear=interface.clear,
                    channel_map=interface.channel_map
                ),
                filter=filter,
                level=level
            )
            for dim in interface.dims
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

    def call(self, construct, inputs, **kwds):
        """
        Activate stored nodes.

        Only activates stored nodes in opened slots.
        """

        # Could possibly also use collections.ChainMap. Need to check how it 
        # will interact w/ downstream processing. - Can
        
        cmds = self._parse_commands(inputs)

        # toggle switches
        switches = []
        for slot, dim in enumerate(self.interface.switch_dims):
            if dim in cmds:
                val = cmds[dim]
                switch = (val == self.interface.switch_vals[1])
                switches.append(switch)
                    
        d = {}
        for switch, cell in zip(switches, self.cells):
            if switch is True:
                d_cell = cell.call(construct, inputs.copy(), **kwds)
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

        # global reset
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
            # This should publish a warning in a log; remember to do this when 
            # setting up logging. - Can

        # Filter irrelevant feature symbols
        cmd_set = set(
            f for f in raw_cmds if f in self.interface.features and (
                f.dim == self.interface.reset_dim or
                f.dim in self.interface.switch_dims
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
