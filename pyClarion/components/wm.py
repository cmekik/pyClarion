from pyClarion.base import *
from pyClarion.components.datastructures import Chunks
from typing import Iterable, List, Hashable, Any, Tuple, Callable
from itertools import groupby, product

FeatureCons = Callable[[Hashable, Hashable], ConstructSymbol]

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
    ) -> None:
        """
        Initialize a new WorkingMemory propagator instance.

        :param slots: A list of names for WM slots. 
        :param dims: A tuple of names for WM state flag dims. The first entry 
            names the dimension signaling which slots are occupied. The second 
            entry names the dimension signaling which slots are excluded from 
            the output. 
        """

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

    def write(self, slot, *nodes):
        """Set slot to contain given nodes."""

        self.store[slot].clear()
        self.store[slot].update(*nodes)

    def toggle(self, slot):
        """Toggle whether to exclude slot contents from output."""

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


class BaseWMUpdater(object):

    def __init__(self, controller: Tuple[ConstructSymbol, ConstructSymbol]):

        self.controller = controller

    def __call__(self, realizer):

        raise NotImplementedError()

    def parse_commands(self, packet):

        raw_cmds = packet.selection

        # Filter irrelevant data
        cmds = set(f for f in raw_cmds if f in self.interface)

        # Validate cmds
        s = sorted(cmds, FeatureSymbol.dim)
        for k, g in groupby(s):
            if len(g) > 1:
                raise ValueError(
                    "Ill-formed WM command in dimension '{}'.".format(k)
                )
        
        return cmds

    @property
    def interface(self):
        """List of features governing updater behavior."""

        raise NotImplementedError()



class WMWriter(BaseWMUpdater):
    """
    Updates the state of a WorkingMemory propagator.
    
    Collaborates w/ WorkingMemory objects. 

    If an invalid command specification is passed to the object, does nothing.
    """

    def __init__(
        self,
        source: ConstructSymbol,
        controller: Tuple[ConstructSymbol, ConstructSymbol],
        write_dims: List[Hashable],
        write_vals: List[Hashable],
        chunks: Chunks
    ) -> None:
        """
        Initialize a new WMUpdater instance.

        :param source: Construct from which WM will be populated.
        :param controller: Tuple indicating controller construct for WM. First 
            member specifies subsystem, second member specifies response.
        :param write_dims: Dimensions for WM write commands. One dimension per 
            slot, listed in order.
        :param write_vals: Values for WM write commands. Maps one value to each 
            data channel.
        :param chunks: Chunk database from which to populate the WM.
        """

        super().__init__(controller=controller)
        self.source = source
        self.write_dims = write_dims
        self.write_vals = write_vals
        self.chunks = chunks

    def __call__(self, realizer):

        inputs = {
            construct: pull_func() 
            for construct, pull_func in realizer.inputs.items()
        }

        source = inputs[self.source]
        subsys, resp = self.controller
        cmd_packet = inputs[subsys].decisions[resp] 
        cmds = self.parse_commands(packet=cmd_packet)

        for cmd in cmds:
            slot = self.write_dims.index(cmd.dim)
            channel = self.write_vals[cmd.val]
            data_packet = source.decisions[channel]
            nodes = self.get_nodes(packet=data_packet)
            realizer.propagator.write(slot, *nodes)
            slots.append(slot)
   
    def get_nodes(self, packet):

        for ch in packet.selection:
            yield ch
            if self.chunks is not None:
                form = self.chunks.get_form(node)
                for f in chain(*(d["values"] for d in form.values())):
                    yield f

        # Should probably cache this. - Can
        @property 
        def interface(self):
            
            iterator = product(self.write_dims, self.write_vals)
            return [feature(dim, val) for dim, val in iterator]


class WMResetter(object):

    def __init__(
        self,
        controller,
        reset_dim,
        reset_vals
    ):
        """
        Initialize a WMResetter instance.
        
        :param reset_dim: Dimension of WM reset commands.
        :param reset_vals: Values for WM reset commands. Mapping from a 
            hashable to true and false.
        """

        super().__init__(controller=controller)
        self.reset_dim = reset_dim
        self.reset_vals = reset_vals
    
    def __call__(self, realizer):
        
        inputs = {
            construct: pull_func() 
            for construct, pull_func in realizer.inputs.items()
        }

        subsys, resp = self.controller
        cmd_packet = inputs[subsys].decisions[resp] 
        cmds = self.parse_commands(packet=cmd_packet)

        for cmd in cmds:
            if self.reset_vals[cmd.val] == True:
                realizer.propagator.reset()

    @property
    def interface(self):

        return [feature(self.reset_dim, val) for val in self.reset_vals]


class WMSwitch(BaseWMUpdater):

    def __init__(
        self,
        controller,
        switch_dims,
        switch_vals
    ):
        """
        Initialize a WMResetter instance.
        
        :param read_dims: Dimension for WM switch commands. One for each slot.
        :param reset_vals: Values for WM switch commands. Mapping from a 
            hashable to true and false.
        """

        super().__init__(controller)
        self.read_dims = read_dim
        self.read_vals = read_vals

    def __call__(self, realizer):
        
        inputs = {
            construct: pull_func() 
            for construct, pull_func in realizer.inputs.items()
        }

        subsys, resp = self.controller
        cmd_packet = inputs[subsys].decisions[resp] 
        cmds = self.parse_commands(packet=cmd_packet)

        for cmd in cmds:
            if self.read_vals[cmd.val] == True:
                i = self.read_dims.index(cmd.dim)
                slot = realizer.propagator.slots[i]
                realizer.propagator.toggle(slot)

    @property
    def interface(self):

        return [feature(self.read_dim, val) for val in self.read_vals]
