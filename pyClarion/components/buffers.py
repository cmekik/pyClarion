"""Definitions for memory constructs, most notably working memory."""


__all__ = ["ParamSet", "Register", "RegisterArray"]


from ..base.symbols import ConstructType, Symbol, feature, subsystem, terminus
from .. import numdicts as nd
from .. import base
from .blas import BLAs

from typing import Callable, Hashable, Tuple, List, Collection, cast


class ParamSet(base.Process):
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
        cmd, = self.interface.parse_commands(data)

        cmd_index = self.interface.cmds.index(cmd)
        if cmd_index == 0:
            pass
        elif cmd_index == 1:
            self.store.clear()
        elif cmd_index == 2:
            param_strengths = nd.keep(data, keys=self.interface.params)
            self.store.max(param_strengths)
        else:
            assert cmd_index == 3
            self.store.clear()
            param_strengths = nd.keep(data, keys=self.interface.params)
            self.store.max(param_strengths)

        return self.store

    class Interface(base.Interface):
        """Control interface for ParamSet instances."""

        _config = ("name", "pmkrs", "wmkr", "vsby", "vclr", "vupd", "vclrupd")

        def __init__(
            self,
            name: Hashable,
            pmkrs: Tuple[Hashable, ...],
            wmkr: Hashable = "w", 
            vsby: Hashable = "sby",
            vclr: Hashable = "clr",
            vupd: Hashable = "upd",
            vclrupd: Hashable = "clrupd"
        ):
            """
            Initialize ParamSet.Interface instance.

            :param name: Name of interface client.
            :param pmkrs: Marker values for identifying parameters.
            :param wmkr: Marker for write ops.
            :param vsby: Value for standby action.
            :param vclr: Value for clear action.
            :param vupd: Value for update action.
            :param vclrupd: Value for clear+update action.
            """

            with self.config():
                self.name = name
                self.pmkrs = pmkrs
                self.wmkr = wmkr
                self.vsby = vsby
                self.vclr = vclr
                self.vupd = vupd
                self.vclrupd = vclrupd

        def update(self):

            wtag = (self.name, self.wmkr)
            wvals = (self.vsby, self.vclr, self.vupd, self.vclrupd)
            ptags = [(self.name, pmkr) for pmkr in self.pmkrs]

            super().__init__(
                cmds=tuple(feature(wtag, wval) for wval in wvals),
                params=tuple(feature(ptag) for ptag in ptags)
            )


class Register(base.Process):
    """A dynamic store of nodes."""

    _serves = ConstructType.buffer
    _interface: "Register.Interface"

    def __init__(
        self,
        controller: Tuple[subsystem, terminus],
        sources: Tuple[Tuple[subsystem, terminus], ...],
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
        :param blas: Option to update BLAs at call time.
        """

        super().__init__(expected=(controller,) + sources)

        self._controller = controller
        self._sources = sources

        self.store = nd.MutableNumDict(default=0.0)
        self.flags = nd.MutableNumDict(default=0.0)

        self.blas = blas
        self.update_blas = update_blas
        self.interface = interface

    @property
    def interface(self) -> "Register.Interface":
        """Interface domain associated with self."""

        return self._interface

    @interface.setter
    def interface(self, obj: "Register.Interface"):

        if len(self._sources) != len(obj.vops):
            msg = "Incompatible interface: len(vops) != len(sources)."
            raise ValueError(msg)
        self._interface = obj
        obj.lock()

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
        behavior is to maintain the current memory state (subject to deletions 
        due to BLAs being below threshold, if these are defined).

        If the register is empty, will output an empty flag.
        """

        datas = self.extract_inputs(inputs)
        cmd_data, src_data = datas[0], datas[1:]

        cmd, = self.interface.parse_commands(cmd_data)
        cmd_index = self.interface.cmds.index(cmd)
        if cmd_index == 0:
            pass
        elif cmd_index == 1:
            self.store.clear()
        else:
            data = src_data[cmd_index - 2]
            self.store.clearupdate(data)
            if self.blas is not None:
                for key in data:
                    self.blas.register_invocation(key, add_new=True)

        if self.is_empty:
            self.flags.extend(self.interface.flags, value=1.0)
        else:
            self.flags.clear()

        d = nd.MutableNumDict(self.store, default=0)
        d.max(self.flags)
        d.squeeze()

        if self.blas is not None:
            # Remove items below threshold.
            keys = self.blas.keys_below_threshold(self.store)
            self.store.drop(keys=keys)
            if self.update_blas:
                self.blas.step()
                self.blas.prune()

        return d

    class Interface(base.Interface):
        """Control interface for Register instances."""

        _config = ("name", "cmkr", "fmkr", "vops", "vsby", "vclr")

        def __init__(
            self,
            name: Hashable,
            vops: Tuple[Hashable, ...],
            wmkr: Hashable = "w", 
            fmkr: Hashable = "empty",
            vsby: Hashable = "sby",
            vclr: Hashable = "clr"
        ) -> None:
            """
            Initialize Register.Interface instance.

            :param name: Name of interface client.
            :param vops: Values for selecting write op sources.
            :param wmkr: Marker for write ops.
            :param fmkr: Marker for null flag.
            :param vsby: Value for standby action.
            :param vclr: Value for clear action.
            """

            with self.config():
                self.name = name
                self.wmkr = wmkr
                self.fmkr = fmkr
                self.vops = vops
                self.vsby = vsby
                self.vclr = vclr

        def update(self) -> None:

            vals = (self.vsby, self.vclr, *self.vops)
            wtag = (self.name, self.wmkr)
            ftag = (self.name, self.fmkr)

            super().__init__(
                cmds=tuple(feature(wtag, val) for val in vals),
                flags=(feature(ftag),)
            )


class RegisterArray(base.Process):
    """
    An array of registers.

    Exposes a slot-based storage and control architecture. Supports writing data
    to slots, clearing slots, excluding slots from the output and clearing all 
    slots. 
    """

    _serves = ConstructType.buffer
    _interface: "RegisterArray.Interface"
    cells: Tuple[Register, ...]

    def __init__(
        self,
        controller: Tuple[subsystem, terminus],
        sources: Tuple[Tuple[subsystem, terminus], ...],
        interface: "RegisterArray.Interface",
        blas: BLAs = None,
        update_blas: bool = True
    ) -> None:
        """
        Initialize RegisterArray instance.

        :param controller: Reference for construct issuing commands to self.
        :param sources: References for constructs from which to pull data.
        :param interface: Defines features for controlling updates to self.
        :param blas: Optional BLA database. When items in a cell store are found
            to have BLA values below the set density threshold, they are 
            dropped. The database is shared among all register array cells.
        :param update_blas: Option to update BLAs at call time.
        """

        super().__init__(expected=(controller,) + sources)

        self._controller = controller
        self._sources = sources

        self.blas = blas
        self.update_blas = update_blas
        self.interface = interface  # automatically spawns memory cells

    @property
    def interface(self) -> "RegisterArray.Interface":
        """Interface domain associated with self."""

        return self._interface

    @interface.setter
    def interface(self, obj: "RegisterArray.Interface") -> None:

        if len(self._sources) != len(obj.vops):
            msg = "Incompatible interface: len(vops) != len(sources)."
            raise ValueError(msg)
        self._interface = obj
        obj.lock()

        self.cells = tuple(
            Register(
                controller=self._controller,
                sources=self._sources,
                interface=self._interface._sub_interfaces[i],
                blas=self.blas,
                update_blas=False
            )
            for i in range(self._interface.slots)
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

        data = self.extract_inputs(inputs)
        cmds = self.interface.parse_commands(data[0])

        clr_cmd = self.interface.cmds.index(cmds[0])
        if clr_cmd == 1:  # Global clear
            for cell in self.cells:
                cell.store.clear()

        d = nd.MutableNumDict(default=0.0)
        for i, cell in enumerate(self.cells):
            cell_strengths = cell.call(inputs)
            read_cmd = self.interface.cmds.index(cmds[i + 1])
            if read_cmd == 2 * (i + 1) + 1:  # Read cell
                d.max(cell_strengths)

        if self.blas is not None and self.update_blas:
            self.blas.step()
            self.blas.prune()

        return d

    class Interface(base.Interface):
        """Control interface for RegisterArray instances."""

        _config = (
            "name", "slots", "vops", "wmkr", "rmkr", "cmkr", "fmkr",
            "vsby", "vclr", "vread"
        )

        def __init__(
            self,
            name: Hashable,
            slots: int,
            vops: Tuple[Hashable, ...],
            wmkr: Hashable = "w",
            rmkr: Hashable = "r",
            cmkr: Hashable = "clr",
            fmkr: Hashable = "empty",
            vsby: Hashable = "sby",
            vclr: Hashable = "clr",
            vread: Hashable = "read"
        ) -> None:
            """
            Initialize RegisterArray.Interface instance.

            :param name: Name of interface client.
            :param slots: Number of slots.
            :param vops: Values for selecting write op sources.
            :param wmkr: Marker for write ops.
            :param rmkr: Marker for read ops.
            :param cmkr: Marker for clear ops.
            :param fmkr: Marker for null flag.
            :param vsby: Value for standby action.
            :param vclr: Value for clear action.
            :param vread: Value for read action. 
            """

            with self.config():
                self.name = name
                self.slots = slots
                self.vops = vops
                self.wmkr = wmkr
                self.rmkr = rmkr
                self.cmkr = cmkr
                self.fmkr = fmkr
                self.vsby = vsby
                self.vclr = vclr
                self.vread = vread

        def update(self) -> None:

            clr_cmds = (
                feature((self.name, self.cmkr), self.vsby),
                feature((self.name, self.cmkr), self.vclr)
            )

            read_cmds = [
                (feature((self.name, (self.rmkr, i)), self.vsby),
                 feature((self.name, (self.rmkr, i)), self.vread))
                for i in range(self.slots)
            ]

            sub_interfaces = [
                Register.Interface(
                    name=self.name,
                    vops=self.vops,
                    wmkr=(self.wmkr, i),
                    fmkr=(self.fmkr, i),
                    vsby=self.vsby,
                    vclr=self.vclr
                )
                for i in range(self.slots)
            ]

            cmds: Tuple[feature, ...] = clr_cmds
            for rcmds in read_cmds:
                cmds += rcmds
            for sub_interface in sub_interfaces:
                cmds += sub_interface.cmds

            flags: Tuple[feature, ...] = ()
            for sub_interface in sub_interfaces:
                flags += sub_interface.flags

            for sub_interface in sub_interfaces:
                sub_interface.lock()
            self._sub_interfaces = tuple(sub_interfaces)

            super().__init__(cmds=cmds, flags=flags)
