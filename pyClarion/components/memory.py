

__all__ = ["MemoryCell", "WorkingMemory"]


from pyClarion.base.symbols import Symbol, MatchSet, ConstructType, feature
from pyClarion.components.propagators import PropagatorB
from pyClarion.utils import simple_junction

from itertools import chain, product, groupby
from typing import Callable, Hashable, Tuple, NamedTuple, List


class MemoryCell(PropagatorB):
    """Activates a stored set of nodes."""

    class Interface(NamedTuple):
        """Control interface for MemoryCell instances."""

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
        :param interface: Defines features for controlling self.
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
        """Return stored strengths."""

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
        self.store.clear()

    @property
    def is_empty(self):
        return len(self.store) == 0

    @property
    def interface(self):

        return self._interface

    @interface.setter
    def interface(self, obj: Interface):

        self._interface = obj
        self._channel_dict = dict(obj.channel_map)


class WorkingMemory(PropagatorB):

    class Interface(NamedTuple):

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
            """Tuple listing all interface features associated with self."""

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
            """Tuple listing default action features associated with self."""

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

        self.controller = controller
        self.source = source
        self.interface = interface
        self.level = level

        self.switches: List[bool] = [False for _ in interface.switch_dims]
        self.cells = tuple(
            MemoryCell(
                controller=controller,
                source=source,
                interface=MemoryCell.Interface(
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

    def toggle(self, slot):
        """
        Toggle whether to exclude slot contents from output.

        Toggling an empty slot has no effect.
        """

        if not self.cells[slot].is_empty:
            self.switches[slot] = not self.switches[slot]

    def reset(self):
        """Reset memory state."""

        self.switches: Any = [False for _ in self._slots]
        for cell in self.cells:
            cell.clear()

    def reset_switch(self, slot):

        self.switches[slot] = False

    def call(self, construct, inputs, **kwds):

        # Could possibly also use collections.ChainMap. Need to check how it 
        # will interact w/ downstream processing. - Can
        
        d = {}
        for switch, cell in zip(self.switches, self.cells):
            if switch is True:
                d_cell = cell.call(construct, inputs.copy(), **kwds)
                d.extend(d_cell)
        
        return d

    def update(self, construct, inputs):

        cmds = self._parse_commands(inputs.copy())

        # NOTE: The order in which updates are implemented matters:
        #   Global WM reset preceeds switch toggling which, in turn, preceeds 
        #   slot updates. This has two notable consequences:
        #       1. Clearing the WM and populating it w/ new information (e.g., 
        #       in the service of a new goal) can be done in one step. 
        #       2. A slot will never be in an open switch state if it contains 
        #       nothing. 
        # - Can

        # global reset
        if self.interface.reset_dim in cmds:
            val = cmds[self.interface.reset_dim]
            if self.interface.reset_vals[1] == val:
                self.reset()

        # toggle switches
        for slot, dim in enumerate(self.interface.switch_dims):
            if dim in cmds:
                val = cmds[dim]
                if val == self.switch_vals[1]:
                    self.toggle(slot)

        # cell/slot updates
        for slot, cell in enumerate(self.cells):
            cell.update(construct, inputs.copy())
            # Clearing a slot automatically sets corresponding switch to False.
            if cell.is_empty:
                self.reset_switch(slot)

    def _parse_commands(self, inputs):

        # Filter irrelevant data
        cmd_set = set(
            f for f in inputs if f in self.interface.features and (
                f.dim == self.interface.reset_dim or
                f.dim in self.interface.switch_dims
            )
        )

        cmds = {}
        s = sorted(cmd_set, key=self._key_func)
        for k, g in groupby(s, self._key_func):
            g = list(g)
            if len(g) > 1:
                raise ValueError("Multiple commands for {} in WM.".format(k))
            cmds[k] = g.pop()
        # TODO: Need to better handle missing commands. Missing command should 
        # simply mean 'default'. Otherwise, complicates agent initialization. 
        # - Can
        # if len(cmds) != len(self.interface.switch_dims) + 1:
        #     raise ValueError("Missing WM command.")
        
        return cmds

    @staticmethod
    def _key_func(ftr):

        return ftr.dim
