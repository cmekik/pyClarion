"""Definitions for memory constructs, most notably working memory."""


__all__ = ["ParamSet", "Register", "RegisterArray", "collect_cmd_data"]


from ..base.symbols import (
    ConstructType, Symbol, 
    feature, subsystem, terminus,
    group_by_dims, lag
)
from ..base import numdicts as nd
from ..base.components import (
    Inputs, Propagator, FeatureInterface, FeatureDomain
)

from typing import Callable, Hashable, Tuple, List, Mapping, Collection
from dataclasses import dataclass
from itertools import chain, product
from types import MappingProxyType
import logging


def collect_cmd_data(
    construct: Symbol, 
    inputs: Inputs, 
    controller: Tuple[subsystem, terminus]
) -> nd.FrozenNumDict:
    """
    Extract command data from inputs. 
    
    Logs failure, but does not throw error.
    """

    subsystem, terminus = controller
    try:
        data = inputs[subsystem][terminus]
    except KeyError:
        data = frozenset()
        msg = "Failed data pull from %s in %s."
        logging.warning(msg, controller, construct)
    
    return data


class ParamSet(Propagator):
    """A controlled store of parameters."""

    _serves = ConstructType.buffer

    @dataclass
    class Interface(FeatureInterface):
        """
        Control interface for ParamSet instances.

        :param clients: Symbols to which parameters are mapped.
        :param func: Function consuming client symbols and outputting 
        :param tag: Tag for ParamSet control dimension.
            corresponding parameter tags. It is okay to map two clients to the 
            same tag. This will couple their values.
        :param standby: Value for standby action.
        :param clear: Value for clear action.
        :param update: Value for update action.
        :param overwrite: Value for overwrite action.
        :param param_val: Singleton value to be used in parameter features.
        """

        clients: Collection[Symbol]
        func: Callable[..., Hashable]
        tag: Hashable
        standby_val: Hashable = "standby"
        clear_val: Hashable = "clear"
        update_val: Hashable = "update"
        overwrite_val: Hashable = "overwrite"
        param_val: Hashable = ""

        def _set_interface_properties(self):

            _func, _pval = self.func, self.param_val
            _cmd_vals = (
                self.standby_val, self.clear_val, self.update_val, 
                self.overwrite_val
            )
            _cmds = (feature(self.tag, val) for val in _cmd_vals)
            _defaults = {feature(self.tag, self.standby_val)}
            _params = (feature(_func(s), _pval) for s in self.clients)

            self._cmds = frozenset(_cmds)
            self._defaults = frozenset(_defaults)
            self._flags = frozenset()
            self._params = frozenset(_params)

        def _validate_data(self):

            _param_tags = set(self.func(s) for s in self.clients)
            cmd_vals = set([
                self.standby_val, self.clear_val, self.update_val, 
                self.overwrite_val
            ])

            if len(cmd_vals) < 4:
                raise ValueError("Vals must contain unique values.")
            if self.tag in _param_tags:
                msg = "Tag cannot be equal to an output of func over clients."
                raise ValueError(msg)

    def __init__(
        self, 
        controller: Tuple[subsystem, terminus], 
        interface: Interface,
    ) -> None:

        self.store = nd.NumDict()
        self.flags = nd.NumDict()

        self.controller = controller
        self.interface = interface

    @property
    def expected(self):

        return frozenset((self.controller[0],))

    def call(self, inputs):
        """
        Update the paramset state and emit outputs.

        Updates are controlled by matching features emitted in the output of 
        self.controller to those defined in self.interface. If no commands are 
        encountered, default/standby behavior will be executed. The default 
        behavior is to maintain the current state.
        """

        data = collect_cmd_data(self.client, inputs, self.controller)
        cmds = self.interface.parse_commands(data)

        try:
            (dim, val), = cmds.items() # Extract unique cmd (dim, val) pair.
        except ValueError:
            msg = "{} expected exactly one command, received {}"
            raise ValueError(msg.format(type(self).__name__, len(cmds)))

        if val == self.interface.standby_val:
            pass
        elif val == self.interface.clear_val:
            self.clear_store()
        elif val == self.interface.update_val:
            param_strengths = nd.restrict(data, self.interface.params)
            self.update_store(param_strengths)
        elif val == self.interface.overwrite_val:
            self.clear_store()
            param_strengths = nd.restrict(data, self.interface.params)
            self.update_store(param_strengths)
        else:
            raise ValueError("Unexpected command value: {}.".format(repr(val)))

        d = nd.NumDict()
        strengths = nd.transform_keys(self.store, feature.tag.fget)
        for construct in self.interface.clients:
            d[construct] = strengths[self.interface.func(construct)]

        return d

    def update(self, inputs, output):

        # Flags get cleared on each update. New flags may then be added for the 
        # next cycle.
        self.clear_flags()

    def update_store(self, strengths):
        """
        Update strengths in self.store.
        
        Write op is implemented using the max operation. 
        """

        self.store |= strengths

    def clear_store(self):
        """Clear any nodes stored in self."""

        self.store.clear()

    def update_flags(self, strengths):

        self.flags |= strengths

    def clear_flags(self):

        self.flags.clear()


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
        
        :param channel_map: Tuple pairing values to termini for write 
            operation.
        :param tag: Tag for controlling write ops to register.
        :param flag_tag: Tag for defining null flag.
        :param standby_val: Value for standby action.
        :param clear_val: Value for to clear action.
        :param flag_val: Null flag value.
        """

        mapping: Mapping[Hashable, Symbol]
        tag: Hashable
        flag_tag: Hashable
        standby_val: Hashable = "standby"
        clear_val: Hashable = "clear"
        flag_val: Hashable = ""

        @property
        def null_flag(self):

            return self._null_flag

        def _set_interface_properties(self) -> None:
            
            vals = chain((self.standby_val, self.clear_val), self.mapping)
            default = feature(self.tag, self.standby_val)
            flag = feature(self.flag_tag, self.flag_val)

            self._null_flag = flag

            self._cmds = frozenset(feature(self.tag, val) for val in vals)
            self._defaults = frozenset({default})
            self._flags = frozenset({flag})
            self._params = frozenset()

        def _validate_data(self):
            
            values = set(chain((self.standby_val, self.clear_val), self.mapping))
            if len(values) < len(self.mapping) + 2:
                raise ValueError("Value set may not contain duplicates.") 

    def __init__(
        self, 
        controller: Tuple[subsystem, terminus], 
        source: subsystem,
        interface: Interface
    ) -> None:
        """
        Initialize a Register instance.

        :param controller: Reference for construct issuing commands to self.
        :param source: Reference for construct from which to pull data.
        :param interface: Defines features for controlling updates to self.
        :param forward_commands: Optional bool indicating whether or not to 
            include received commands in emitted output. False by default. If 
            set to true, received commands are outputted with a lag value of 1.
        """

        self.store = nd.NumDict() 
        self.flags = nd.NumDict()

        self.controller = controller
        self.source = source
        self.interface = interface

    @property
    def is_empty(self):
        """True iff no nodes are stored in self."""

        return len(self.store) == 0

    @property
    def expected(self):

        return frozenset((self.source, self.controller[0]))

    def call(self, inputs):
        """
        Update the register state and emit the current register output.

        Updates are controlled by matching features emitted in the output of 
        self.controller to those defined in self.interface. If no commands are 
        encountered, default/standby behavior will be executed. The default 
        behavior is to maintain the current memory state.
        """

        data = collect_cmd_data(self.client, inputs, self.controller)
        cmds = self.interface.parse_commands(data)

        try:
            (dim, val), = cmds.items() # Extract unique cmd (dim, val) pair.
        except ValueError:
            msg = "{} expected exactly one command, received {}"
            raise ValueError(msg.format(type(self).__name__, len(cmds)))

        if val == self.interface.standby_val:
            pass
        elif val == self.interface.clear_val:
            self.clear_store()
        elif val in self.interface.mapping: 
            channel = self.interface.mapping[val]
            self.clear_store()
            self.update_store(inputs[self.source][channel])
            self.store.squeeze()

        if len(self.store) == 0:
            self.flags[self.interface.null_flag] = 1.0

        d = nd.NumDict(self.store)
        d |= self.flags

        return nd.NumDict(d)

    def update(self, inputs, output):
        """
        Clear the register flag buffer.
        
        For richer update/learning behaviour, add updaters to client construct.
        """

        self.clear_flags()

    def update_store(self, strengths):
        """
        Update strengths in self.store.
        
        Write op is implemented using the max operation. 
        """

        self.store |= strengths

    def clear_store(self):
        """Clear any nodes stored in self."""

        self.store.clear()

    def update_flags(self, strengths):

        self.flags |= strengths

    def clear_flags(self):

        self.flags.clear()


class RegisterArray(Propagator):
    """
    An array of pyClarion memory registers mechanism.

    The mechanism follows a slot-based storage and control architecture. It 
    supports writing data to slots, clearing slots, excluding slots from the 
    output and resetting the memory state. 

    This class defines the basic datastructure and memory update method. For 
    minimality, it does not report mechanism states (e.g., which slots are 
    filled).
    """

    _serves = ConstructType.buffer

    @dataclass
    class Interface(FeatureInterface):
        """
        Control interface for WorkingMemory propagator.

        :param slots: Number of working memory slots.
        :param mapping: Mapping pairing a write operation value with a 
            terminus from the source subsystem. Signals register array to write 
            contents of terminus to a chosen slot.
        :param prefix: Tag prefix for identifying this particular set of 
            control features.
        :param write_marker: Marker for controlling WM slot write operations.
        :param read_marker: Marker for controlling WM slot read operations.
        :param reset_marker: Marker for controlling global WM state resets.
        :param null_marker: Marker for null flag.
        :param standby_val: Value for standby action (default).
        :param clear_val: Value for clear action.
        :param read_val: Value for read action. 
        :param flag_val: Value for null flag. 
        """

        slots: int
        mapping: Mapping[Hashable, Symbol]
        prefix: Hashable = "wm"
        write_marker: Hashable = "w"
        read_marker: Hashable = "r"
        reset_marker: Hashable = "re"
        null_marker: Hashable = "null"
        standby_val: Hashable = "standby"
        clear_val: Hashable = "clear"
        reset_val: Hashable = "reset"
        read_val: Hashable = "read"
        flag_val: Hashable = ""

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
        def null_tags(self):

            return self._null_tags

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
            na = self.null_marker

            _w_tags = tuple((pre, w, i) for i in range(slots))
            _r_tags = tuple((pre, r, i) for i in range(slots))
            _re_tag = (pre, re)
            _null_tags = tuple((pre, na, i) for i in range(slots))

            _w_chain = chain((self.standby_val, self.clear_val), self.mapping)
            _w_vals = set(_w_chain)
            _r_vals = (self.standby_val, self.read_val)
            _re_vals = (self.standby_val, self.reset_val)
            _null_val = self.flag_val

            _w_d_val = self.standby_val
            _r_d_val = self.standby_val
            _re_d_val = self.standby_val

            _w_gen = ((tag, val) for tag, val in product(_w_tags, _w_vals))
            _r_gen = ((tag, val) for tag, val in product(_r_tags, _r_vals))
            _re_gen = ((_re_tag, val) for val in _re_vals)
            _null_gen = ((tag, _null_val) for tag in _null_tags)

            _w_dgen = ((tag, val) for tag, val in product(_w_tags, _w_vals))
            _r_dgen = ((tag, val) for tag, val in product(_r_tags, _r_vals))
            _re_dgen = ((_re_tag, val) for val in _re_vals)

            _w_cmds = frozenset(feature(tag, val) for tag, val in _w_gen)
            _r_cmds = frozenset(feature(tag, val) for tag, val in _r_gen)
            _re_cmds = frozenset(feature(tag, val) for tag, val in _re_gen)
            _null_flags = frozenset(feature(tag, val) for tag, val in _null_gen)

            _w_defaults = frozenset(feature(tag, _w_d_val) for tag in _w_tags)
            _r_defaults = frozenset(feature(tag, _r_d_val) for tag in _r_tags)
            _re_defaults = frozenset({feature(_re_tag, _re_d_val)})
            _defaults = _w_defaults | _r_defaults | _re_defaults

            self._write_tags = _w_tags
            self._read_tags = _r_tags
            self._reset_tag = _re_tag
            self._null_tags = _null_tags

            self._write_dims = tuple(sorted(set(f.dim for f in _w_cmds)))
            self._read_dims = tuple(sorted(set(f.dim for f in _r_cmds)))
            self._reset_dim = (_re_tag, 0)

            self._cmds = _w_cmds | _r_cmds | _re_cmds
            self._defaults = _defaults
            self._flags = frozenset(_null_flags)
            self._params = frozenset()

        def _validate_data(self):
            
            markers = (self.write_marker, self.read_marker, self.reset_marker)
            w_vals = set((self.standby_val, self.clear_val)) | set(self.mapping)
            r_vals = set((self.standby_val, self.read_val))
            re_vals = set((self.standby_val, self.reset_val))
            
            if len(set(markers)) < 3:
                raise ValueError("Marker arguments must be mutually distinct.")
            if len(set(w_vals)) < len(self.mapping) + 2:
                raise ValueError("Write vals may not contain duplicates.")
            if len(r_vals) < 2:
                raise ValueError("Read vals may not contain duplicates.")
            if len(re_vals) < 2:
                raise ValueError("Reset vals may not contain duplicates.")

    def __init__(
        self,
        controller: Tuple[subsystem, terminus],
        source: subsystem,
        interface: Interface
    ) -> None:
        """
        Initialize a new WorkingMemory instance.

        :param controller: Reference for construct issuing commands to self.
        :param source: Reference for construct from which to pull data.
        :param interface: Defines features for controlling updates to self.
        :param forward_commands: Optional bool indicating whether or not to 
            include received commands in emitted output. False by default. If 
            set to true, received commands are outputted with a lag value of 1.
        """

        self.controller = controller
        self.source = source
        self.interface = interface

        self.flags = nd.NumDict()
        self.gate = nd.NumDict()
        self.cells = tuple(
            Register(
                controller=controller,
                source=source,
                interface=Register.Interface(
                    mapping=interface.mapping,
                    tag=self.interface.write_tags[i],
                    flag_tag=interface.null_tags[i],
                    standby_val=interface.standby_val,
                    clear_val=interface.clear_val,
                    flag_val=interface.flag_val
                ),
            )
            for i in range(self.interface.slots)
        )

    def entrust(self, construct):

        for cell in self.cells:
            cell.entrust(construct)
        super().entrust(construct)

    @property
    def expected(self):

        return frozenset((self.source, self.controller[0]))

    def call(self, inputs):
        """
        Update the memory state and emit output activations.

        Updates are controlled by matching features emitted in the output of 
        self.controller to those defined in self.interface. If no commands are 
        encountered, default/standby behavior will be executed. The default 
        behavior is to maintain the current memory state.
        
        The update cycle processes global resets first, slot contents are 
        updated next. As a result, it is possible to clear the memory globally 
        and populate it with new information (e.g., in the service of a new 
        goal) in one single update. Updating an occupied slot will result in 
        the loss of its previous contents.
        """

        data = collect_cmd_data(self.client, inputs, self.controller)
        cmds = self.interface.parse_commands(data)

        # global wm reset
        if cmds[self.interface.reset_dim] == self.interface.reset_val:
            self.reset_cells()

        d = nd.NumDict()
        for cell, dim in zip(self.cells, self.interface.read_dims):
            cell_strengths = cell.call(inputs)
            if cmds[dim] == self.interface.read_val:
                d |= cell_strengths

        d |= self.flags
        
        return d

    def update(self, inputs, output):
        """
        Clear the working memory flag buffer.
        
        For richer update/learning behaviour, add updaters to client construct.
        """

        self.clear_flags()
        for cell in self.cells:
            cell.update(inputs, output)

    def reset_cells(self):
        """
        Reset memory state.
        
        Clears all memory slots and closes all switches.
        """

        for cell in self.cells:
            cell.clear_store()

    def update_flags(self, strengths):

        self.flags |= strengths

    def clear_flags(self):

        self.flags.clear()
