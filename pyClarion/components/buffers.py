"""Definitions for memory constructs, most notably working memory."""


__all__ = ["ParamSet", "Register", "RegisterArray"]


from ..base.symbols import (
    ConstructType, Symbol,
    feature, subsystem, terminus,
    group_by_dims, lag
)
from .. import numdicts as nd
from ..base.components import Process, FeatureInterface, FeatureDomain
from .blas import BLAs

from typing import Callable, Hashable, Tuple, List, Mapping, Collection, cast
from dataclasses import dataclass
from itertools import chain, product
from types import MappingProxyType
import logging


class ParamSet(Process):
    """A controlled store of parameters."""

    _serves = ConstructType.buffer

    def __init__(
        self,
        controller: Tuple[subsystem, terminus],
        interface: "ParamSet.Interface",
    ) -> None:

        super().__init__(expected=(controller,))

        self.store = nd.MutableNumDict(default=0.0)
        self.interface = interface

    def call(self, inputs):
        """
        Update the paramset state and emit outputs.

        Updates are controlled by matching features emitted in the output of 
        self.controller to those defined in self.interface. If no commands are 
        encountered, default/standby behavior will be executed. The default 
        behavior is to maintain the current state.
        """

        data, = self.extract_inputs(inputs)
        cmds = self.interface.parse_commands(data)

        # Should extract cmd by dim, to allow for compositionality...
        try:
            (dim, val), = cmds.items()  # Extract unique cmd (dim, val) pair.
        except ValueError:
            msg = "{} expected exactly one command, received {}"
            raise ValueError(msg.format(type(self).__name__, len(cmds)))

        if val == self.interface.standby_val:
            pass
        elif val == self.interface.clear_val:
            self.clear_store()
        elif val == self.interface.update_val:
            param_strengths = nd.keep(data, keys=self.interface.params)
            self.update_store(param_strengths)
        elif val == self.interface.overwrite_val:
            self.clear_store()
            param_strengths = nd.keep(data, keys=self.interface.params)
            self.update_store(param_strengths)
        else:
            raise ValueError("Unexpected command value: {}.".format(repr(val)))

        d = nd.MutableNumDict(default=0)
        strengths = nd.transform_keys(self.store, func=feature.tag.fget)
        for construct in self.interface.clients:
            d[construct] = strengths[self.interface.func(construct)]

        return d

    def update_store(self, strengths):
        """
        Update strengths in self.store.

        Write op is implemented using the max operation. 
        """

        self.store.max(strengths)

    def clear_store(self):
        """Clear any nodes stored in self."""

        self.store.clear()

    @dataclass
    class Interface(FeatureInterface):
        """
        Control interface for ParamSet instances.

        :param clients: Symbols to which parameters are mapped.
        :param func: Function consuming client symbols and outputting 
        :param tag: Tag for ParamSet control dimension.
            corresponding parameter tags. It is okay to map two clients to the 
            same tag. This will couple their values.
        :param standby_val: Value for standby action.
        :param clear_val: Value for clear action.
        :param update_val: Value for update action.
        :param overwrite_val: Value for overwrite action.
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


class Register(Process):
    """Dynamically stores and activates nodes."""

    _serves = ConstructType.buffer

    def __init__(
        self,
        controller: Tuple[subsystem, terminus],
        source: subsystem,
        interface: "Register.Interface",
        blas: BLAs = None,
        update_blas: bool = True
    ) -> None:
        """
        Initialize a Register instance.

        :param controller: Reference for construct issuing commands to self.
        :param source: Reference for construct from which to pull data.
        :param interface: Defines features for controlling updates to self.
        :param blas: Optional BLA database. When items in store are found to 
            have BLA values below the set density threshold, they are dropped.
        :param update_blas: Optinally disable stepping blas
        """

        # This depends on dict guarantee to preserve insertion order.
        sources = tuple((source, t) for t in interface.mapping.values())
        super().__init__(expected=(controller,) + sources)

        self.interface = interface  # This should be unchangeable... - Can
        self.store = nd.MutableNumDict(default=0.0)
        self.flags = nd.MutableNumDict(default=0.0)
        self.blas = blas
        self.update_blas = update_blas

    @property
    def is_empty(self):
        """True iff no nodes are stored in self."""

        return len(self.store) == 0

    def call(self, inputs):
        """
        Update the register state and emit the current register output.

        Updates are controlled by matching features emitted in the output of 
        self.controller to those defined in self.interface. If no commands are 
        encountered, default/standby behavior will be executed. The default 
        behavior is to maintain the current memory state.
        """

        datas = self.extract_inputs(inputs)
        cmd_data, src_data = datas[0], datas[1:]

        cmds = self.interface.parse_commands(cmd_data)
        channel_map = dict(zip(self.interface.mapping.values(), src_data))

        try:
            (dim, val), = cmds.items()  # Extract unique cmd (dim, val) pair.
        except ValueError:
            msg = "{} expected exactly one command, received {}"
            raise ValueError(msg.format(type(self).__name__, len(cmds)))

        if val == self.interface.standby_val:
            pass
        elif val == self.interface.clear_val:
            self.store.clear()
        elif val in self.interface.mapping:
            channel = self.interface.mapping[val]
            data = channel_map[channel]
            self.store.clearupdate(data)
            if self.blas is not None:
                for key in data:
                    self.blas.register_invocation(key, add_new=True)
        if len(self.store) == 0:
            self.flags[self.interface.null_flag] = 1.0
        else:
            self.flags.drop(keys=(self.interface.null_flag,))

        d = nd.MutableNumDict(self.store, default=0)
        d.max(self.flags)
        d.squeeze()

        if self.blas is not None and self.update_blas:
            self.blas.step()
            # Remove items below threshold.
            drop = [x for x in self.store if self.blas[x].below_threshold]
            self.store.drop(keys=drop)
            for x in drop:
                del self.blas[x]

        return d

    @dataclass
    class Interface(FeatureInterface):
        """
        Control interface for Register instances.

        :param mapping: Tuple pairing values to termini for write 
            operation.
        :param tag: Tag for controlling write ops to register.
        :param flag_tag: Tag for defining null flag.
        :param standby_val: Value for standby action.
        :param clear_val: Value for to clear action.
        :param flag_val: Null flag value.
        """

        # TODO: Make this immutable/hard to mutate. Mutation can break client
        # Register instances. - Can

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

            values = set(
                chain((self.standby_val, self.clear_val), self.mapping))
            if len(values) < len(self.mapping) + 2:
                raise ValueError("Value set may not contain duplicates.")


class RegisterArray(Process):
    """
    An array of pyClarion memory registers.

    The mechanism follows a slot-based storage and control architecture. It 
    supports writing data to slots, clearing slots, excluding slots from the 
    output and resetting the memory state. 

    This class defines the basic datastructure and memory update method. For 
    minimality, it does not report mechanism states (e.g., which slots are 
    filled).
    """

    _serves = ConstructType.buffer

    def __init__(
        self,
        controller: Tuple[subsystem, terminus],
        source: subsystem,
        interface: "RegisterArray.Interface",
        blas: BLAs = None
    ) -> None:
        """
        Initialize a new WorkingMemory instance.

        :param controller: Reference for construct issuing commands to self.
        :param source: Reference for construct from which to pull data.
        :param interface: Defines features for controlling updates to self.
        :param blas: Optional BLA database. When items in a cell store are found
            to have BLA values below the set density threshold, they are 
            dropped. The database is shared among all register array cells.
        """

        # This depends on dict guarantee to preserve insertion order.
        sources = tuple((source, t) for t in interface.mapping.values())
        super().__init__(expected=(controller,) + sources)
        self.interface = interface

        self.flags = nd.MutableNumDict(default=0.0)
        self.gate = nd.MutableNumDict(default=0.0)
        self.blas = blas
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
                blas=blas
            )
            for i in range(self.interface.slots)
        )

    def entrust(self, construct):

        for cell in self.cells:
            cell.entrust(construct)
        super().entrust(construct)

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

        datas = self.extract_inputs(inputs)
        cmd_data = datas[0]
        cmds = self.interface.parse_commands(cmd_data)

        # global wm reset
        if cmds[self.interface.reset_dim] == self.interface.reset_val:
            self.reset_cells()

        d = nd.MutableNumDict(default=0.0)
        for cell, dim in zip(self.cells, self.interface.read_dims):
            cell_strengths = cell.call(inputs)
            if cmds[dim] == self.interface.read_val:
                d.max(cell_strengths)

        d.max(self.flags)
        if self.blas is not None:
            self.blas.step()
            for cell in self.cells:
                # Remove items below threshold.
                drop = [x for x in cell.store if self.blas[x].below_threshold]
                cell.store.drop(keys=drop)
            for x, bla in self.blas.items():
                if bla.below_threshold:
                    del self.blas[x]

        return d

    def reset_cells(self):
        """
        Reset memory state.

        Clears all memory slots and closes all switches.
        """

        for cell in self.cells:
            cell.clear_store()

    def update_flags(self, strengths):

        self.flags.max(strengths)

    def clear_flags(self):

        self.flags.clear()

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

        # TODO: Make this immutable/hard to mutate. Mutation can break client
        # RegisterArray instances. - Can

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
            _null_flags = frozenset(feature(tag, val)
                                    for tag, val in _null_gen)

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
            w_vals = set((self.standby_val, self.clear_val)
                         ) | set(self.mapping)
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
