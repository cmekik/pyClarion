from pyClarion.base import *
from pyClarion.components.propagators import PropagatorB
from pyClarion.components.chunks import Chunks
from typing import Iterable, List, Hashable, Any, Tuple, Callable, Mapping
from itertools import groupby, product, chain


__all__ = ["WorkingMemory", "WMUpdater"]


class WorkingMemory(PropagatorB):
    """
    A simple working memory mechanism.

    The mechanism follows a slot-based architecture. It supports writing data 
    to slots, clearing slots, excluding slots from the output and resetting the 
    memory state. Slot numbers may also be adjusted (although stored data will 
    be lost). The output is all or none and assumes activations range in 
    [0, 1], where 0 is assumed to be the default.

    This class defines the basic datastructure, update methods, WM state flags, 
    and output computation routine for a slot based WM. Although update methods 
    are defined as part of the datastructure interface, this class DOES NOT 
    handle updates to the WM state. WM state updates should be handled by a 
    dedicated updater.
    """

    store: Any
    excludes: Any

    def __init__(
        self, 
        slots: List[Hashable], 
        dims: Tuple[Hashable, Hashable],
        matches: MatchSet = None
    ) -> None:
        """
        Initialize a new WorkingMemory propagator instance.

        :param slots: A list of names for WM slots. 
        :param dims: A tuple of names for WM state flag dims. The first entry 
            names the dimension signaling which slots are occupied. The second 
            entry names the dimension signaling which slots are excluded from 
            the output. 
        """

        super().__init__(matches=matches)
        self.dims = dims 
        self.slots = slots

    def call(self, construct, inputs, **kwds):
        """Compute WM output from stored state (ignores args)."""

        d = {}
        occu, excl = self.dims
        for i, slot in enumerate(self.store):
            if len(slot) > 0:
                d[feature(occu, self.slots[i])] = 1.0
                if self.excludes[i]:
                    d[feature(excl, self.slots[i])] = 1.0
                else:
                    for node in slot:
                        d[node] = 1.0                

        return d

    def write(self, slot, nodes):
        """Set slot to contain given nodes."""

        self.store[slot].clear()
        self.store[slot].update(nodes)

    def toggle(self, slot):
        """
        Toggle whether to exclude slot contents from output.

        Toggling an empty slot has no effect.
        """

        if len(self.store[slot]) > 0:
            self.excludes[slot] = not self.excludes[slot]

    def clear(self, slot):
        """Clear contents of slot and remove exclusion marker."""

        self.store[slot].clear()
        self.excludes[slot] = False

    def reset(self):
        """Reset memory state."""

        self.store: Any = [set() for _ in self._slots]
        self.excludes: Any = [False for _ in self._slots] 

    @property
    def slots(self):
        """
        Number of WM slots.

        Warning: Setting this attribute will automatically reset the WM state.
        """

        return self._slots

    @slots.setter
    def slots(self, val):

        self._slots = val
        self.reset()


class WMUpdater(object):
    """
    Updates the state of a WorkingMemory propagator.
    
    Collaborates w/ WorkingMemory objects. 

    If an invalid command specification is passed to the object, does nothing.

    Updates to WM state occur as follows:
        First, the WM memory state is reset, if such a command is received. 
        Then, any writing actions occur. Writing actions modify the memory 
        content of specific WM slots; clearing a specific slot is considered a 
        write action. Multiple slots may be written to simultaneously. Finally, 
        output toggling actions are performed. These govern, for each slot, 
        whether the WM includes the content of the slot in its output.
    """

    def __init__(
        self,
        source: Symbol,
        controller: Tuple[Symbol, Symbol],
        reset_dim: Hashable,
        reset_vals: Mapping[Hashable, bool],
        write_dims: List[Hashable],
        write_clear: Hashable,
        write_standby: Hashable,
        write_channels: Mapping[Hashable, Symbol],
        switch_dims: List[Hashable],
        switch_vals: Mapping[Hashable, bool],
        chunks: Chunks
    ) -> None:
        """
        Initialize a new WMUpdater instance.

        :param source: Construct from which WM will be populated.
        :param controller: Tuple indicating controller construct for WM. First 
            member specifies subsystem, second member specifies terminus.
        :param reset_dim: Dimension of WM reset commands.
        :param reset_vals: Values for WM reset commands. Mapping from a 
            hashable to true and false.
        :param write_dims: Dimensions for WM write commands. One dimension per 
            slot, listed in order.
        :param write_clear: Value that will trigger clearing of a given slot.
        :param write_standby: Value that maintains current memory state of 
            given slot.
        :param write_channels: Values for WM write commands. Maps one value to 
            each data channel.
        :param switch_dims: Dimension for WM switch commands. One for each slot.
        :param switch_vals: Values for WM switch commands. Mapping from a 
            hashable to true and false.
        :param chunks: Chunk database from which to populate the WM.
        """

        if len(reset_vals) > len(reset_vals.values()):
            raise ValueError("Arg reset_vals must be injective.")
        if len(write_channels) > len(write_channels.values()):
            raise ValueError("Arg write_channels must be injective.")
        if len(switch_vals) > len(switch_vals.values()):
            raise ValueError("Arg switch_vals must be injective.")

        self.controller = controller
        self.source = source
        self.reset_dim = reset_dim
        self.reset_vals = reset_vals
        self.write_dims = write_dims
        self.write_clear = write_clear
        self.write_standby = write_standby
        self.write_channels = write_channels
        self.switch_dims = switch_dims
        self.switch_vals = switch_vals
        self.chunks = chunks

    def __call__(self, realizer):

        if not isinstance(realizer.propagator, WorkingMemory):
            raise TypeError(
                "Expected propagator of type WorkingMemory," 
                "got {} instead.".format(type(realizer))
            )
        if len(self.write_dims) != len(realizer.propagator.slots):
            raise TypeError(
            "Write dimensions must match slots in number."
            ) 
        if len(self.switch_dims) != len(realizer.propagator.slots):
            raise TypeError(
            "Switch dimensions must match slots in number."
            ) 

        inputs = {
            construct: pull_func() 
            for construct, pull_func in realizer.inputs.items()
        }

        source = inputs[self.source]
        subsys, resp = self.controller
        cmd_packet = inputs[subsys][resp] 
        cmds = self.parse_commands(packet=cmd_packet)

        # execute reset
        if self.reset_dim in cmds:
            val = cmds[self.reset_dim]
            if self.reset_vals[val] == True:
                realizer.propagator.reset()

        # write to any slots
        for slot, dim in enumerate(self.write_dims):
            if dim in cmds:
                val = cmds[dim]
                if val == self.write_clear:
                    realizer.propagator.clear(slot)
                elif val == self.write_standby:
                    pass
                else:
                    channel = self.write_channels[val]
                    data_packet = source[channel]
                    nodes = self.get_nodes(packet=data_packet)
                    realizer.propagator.write(slot, nodes)

        # toggle any switches
        for slot, dim in enumerate(self.switch_dims):
            if dim in cmds:
                val = cmds[dim]
                if self.switch_vals[val] == True:
                    realizer.propagator.toggle(slot)
   
    def get_nodes(self, packet):
        
        for node in packet:
            yield node

    def parse_commands(self, packet):

        # Filter irrelevant data
        _cmds = set(f for f in packet if f in self.interface)

        # Validate cmds
        cmds = {}
        s = sorted(_cmds, key=self._key_func)
        for k, g in groupby(s, self._key_func):
            g = list(g)
            if len(g) > 1:
                raise ValueError(
                    "Ill-formed WM command in dimension '{}'.".format(k)
                )
            cmds[k] = g.pop().val
        
        return cmds

    # Should probably cache this. But it requires some care to be robust to 
    # changes to underlying datastructures. - Can
    @property 
    def interface(self):
        
        reset = [feature(self.reset_dim, val) for val in self.reset_vals]

        write_special = [self.write_clear, self.write_standby]
        write_vals = chain(write_special, self.write_channels)
        write_dvps = product(self.write_dims, write_vals)
        write = [feature(dim, val) for dim, val in write_dvps]

        switch_dvps = product(self.switch_dims, self.switch_vals)
        switch = [feature(dim, val) for dim, val in switch_dvps]

        return reset + write + switch

    @property
    def dims(self):
        """Dims associated w/ WM updates"""

        return [self.reset_dim] + self.write_dims + self.switch_dims

    @property
    def defaults(self):
        """Features indicating default (i.e. standby) actions."""

        rv = [v for v, b in self.reset_vals.items() if not b].pop()
        wv = self.write_standby
        sv = [v for v, b in self.switch_vals.items() if not b].pop()

        r = feature(self.reset_dim, rv)
        w = [feature(d, wv) for d in self.write_dims]
        s = [feature(d, sv) for d in self.switch_dims]

        return [r] + w + s

    @staticmethod
    def _key_func(ftr):

        return ftr.dim
