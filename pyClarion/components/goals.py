"""Tools for handling goals."""


__all__ = ["GoalStay"]


from ..base.symbols import (
    ConstructType, Symbol, feature, chunk, terminus, buffer, subsystem
)
from ..base.components import Process, Domain
from .. import base
from .blas import BLAs
from .chunks_ import Chunks
from .. import numdicts as nd

from typing import FrozenSet, Tuple, Hashable, Optional, Mapping, cast, Iterable
from types import MappingProxyType
from itertools import count, groupby


class GoalStay(Process):
    """
    A propagator for coordinating goal management.

    Warning: This component is very experimental.

    Maintains the active goal and various goal flags, executes goal commands 
    issued by a controller. May create new goal chunks or retrieve existing 
    goal chunks from a source, as directed. 

    Maintains two flags: the goal state flag and the goal eval flag. The goal 
    state flag serves to inform client processes if a goal has been newly 
    initialized, resumed after an interruption, or is currently active.
    The goal eval flag serves to inform superordinate goals about the outcome 
    of subordinate goals and may also be used to caluculate reinforcement 
    signals.

    If directed to create a new goal chunk with identical features to an 
    existing goal chunk, will create a new goal chunk with a distinct chunk 
    node. This allows pushing many similar goals while keeping them distinct 
    (i.e., for repetitive actions). This behaviour is unlike other subsystems, 
    where chunks are assumed to be unique. (Not sure if this feature will stay.)
    """

    _serves = ConstructType.buffer

    def __init__(
        self, 
        controller: Tuple[subsystem, terminus], 
        source: Tuple[subsystem, terminus], 
        interface: "GoalStay.Interface",
        chunks: Chunks,
        blas: BLAs,
        update_blas: bool = True, 
        prefix: str = "goal"
    ) -> None:

        super().__init__(expected=(controller, source))

        self.interface = interface
        self.chunks = chunks
        self.blas = blas
        self.update_blas = True
        self.prefix = prefix
        self._counter = count(1)

        self.store = nd.MutableNumDict(default=0)
        self.flags = nd.MutableNumDict(default=0)
        self.prevs = nd.MutableNumDict(default=0)
        self.flags.extend((self.interface.flags[i] for i in (0, 3)), value=1.0)

    def call(self, inputs: Mapping[Any, nd.NumDict]) -> nd.NumDict:
        
        cmd_data, src_data = self.extract_inputs(inputs)
        cmds = self.interface.parse_commands(cmd_data)

        cmd, = cmds
        cmd_index = self.interface.cmds.index(cmd)
        if cmd_index == 0: # standby
            pass
        elif cmd_index == 1: # write new goal
            if len(self.store) == 0:
                self.prevs.clear()
            else:
                old_goal, = self.store
                self._update_prevs(old_goal)
            ch = chunk("{}_{}".format(self.prefix, next(self._counter)))
            goal_fs = self.interface.parse_goal_params(cmd_data)
            flags = (self.interface.flags[i] for i in (1, 3))
            self.chunks[ch] = self.chunks.Chunk(features=goal_fs) 
            self.blas.register_invocation(ch, add_new=True)
            self.store.clearupdate(nd.NumDict({ch: 1.0}))
            self.flags.clearupdate(nd.NumDict({f: 1.0 for f in flags}))
        elif cmd_index in [2, 3, 4]: # quit, pass, or fail current goal
            if len(self.store) == 0:
                pass
            else:
                old_goal, = self.store 
                self.chunks.request_del(old_goal)
                self.blas.request_del(old_goal)
                self._update_prevs(old_goal)
                if len(src_data) == 0:
                    flags = (self.interface.flags[i] for i in (0, 3))
                    self.store.clear()
                    self.flags.clearupdate(nd.NumDict({f: 1.0 for f in flags}))
                else:
                    new_goal, = src_data
                    eidx = 1 + cmd_index
                    flags = (self.interface.flags[i] for i in (2, eidx))
                    self.store.clearupdate(src_data)
                    self.flags.clearupdate(nd.NumDict({f: 1.0 for f in flags}))
        else: # engage current goal (set as active)
            assert cmd_index == 5
            flags = (self.interface.flags[i] for i in (0, 3))
            self.flags.clearupdate(nd.NumDict({f: 1.0 for f in flags}))
            self.prevs.clear()

        d = nd.MutableNumDict(default=0.0)
        d.max(self.store)
        d.max(self.flags)
        d.max(self.prevs)

        if self.update_blas:
            self.blas.step()

        return d

    def _update_prevs(self, old):

        old_fs = self.chunks[old].features
        old_idxs = (self.interface.extras.index(f) for f in old_fs)
        prev_idxs = (i + len(self.interface.goals) for i in old_idxs)
        prevs = (self.interface.extras[i] for i in prev_idxs)
        self.prevs.clear()
        self.prevs.extend(prevs, value=1.0)

    class Interface(base.Interface):
        """Goal buffer control interface."""

        _config = (
            "name", "goals", "cmkr", "smkr", "emkr", "vsby", "vna", "vwrite", 
            "vstart", "vresume", "vacton", "vquit", "vpass", "vfail"
        )
        
        def __init__(
            self,
            name: Hashable,
            goals: Tuple[feature, ...],
            cmkr: Hashable = "cmd",
            smkr: Hashable = "state",
            emkr: Hashable = "eval",
            prevmkr: Hashable = "prev",
            vsby: Hashable = "sby",
            vna: Hashable = "na",
            vwrite: Hashable = "write",
            vstart: Hashable = "start",
            vresume: Hashable = "resume",
            vengage: Hashable = "engage",
            vquit: Hashable = "quit",
            vpass: Hashable = "pass",
            vfail: Hashable = "fail",
        ) -> None:
            """
            Initialize GoalStay.Interface instance.

            :param name: Name for goal stay interface features.
            :param goals: Goal features.
            :param cmkr: Command marker. 
            :param smkr: Goal state marker.
            :param emkr: Goal eval marker.
            :param prevmkr: Previous goal marker.
            :param vsby: Standby value.
            :param vna: N/A value.
            :param vwrite: Write value.
            :param vstart: Start value.
            :param vresume: Resume value.
            :param vengage: Engage value.
            :param vquit: Quit value.
            :param vpass: Pass value.
            :param vfail: Fail value.
            """
            
            with self.config():
                self.name = name
                self.goals = goals
                self.cmkr = cmkr
                self.smkr = smkr
                self.emkr = emkr
                self.prevmkr = prevmkr
                self.vsby = vsby
                self.vna = vna
                self.vwrite = vwrite
                self.vstart = vstart
                self.vresume = vresume
                self.vengage = vengage
                self.vquit = vquit
                self.vpass = vpass
                self.vfail = vfail

        def update(self) -> None:

            ctag = (self.name, self.cmkr)
            stag = (self.name, self.smkr)
            etag = (self.name, self.emkr)

            cvals = (
                self.vsby, self.vwrite, self.vquit, self.vpass, self.vfail, 
                self.vengage
            )
            svals = (self.vna, self.vstart, self.vresume)
            evals = (self.vna, self.vpass, self.vfail)

            fspecs = ((stag, svals), (etag, evals))

            prev = self.prevmkr
            fgprev = tuple(feature((f.tag, prev), f.val) for f in self.goals)

            gdvs = tuple((f.tag, f.val) for f in self.goals)
            gsets = (set(g) for _, g in groupby(gdvs, lambda x: x[0]))

            if any(len(s) < 2 for s in gsets):
                raise ValueError("Singleton goal dimensions not allowed.")
            if any(f.lag != 0 for f in self.goals):
                raise ValueError("Goals may not have nonzero lag.")

            super().__init__(
                cmds=tuple(feature(ctag, v) for v in cvals),
                params=tuple(feature((self.name, g), v) for g, v in gdvs),
                flags=tuple(feature(t, v) for t, vs in fspecs for v in vs),
                extras=(self.goals + fgprev)
            )
        
        def parse_goal_params(self, data: Iterable[feature]) -> Tuple[feature]:

            params = tuple(f for f in self.params if f in data)
            goals = tuple(feature(f.tag[1], f.val) for f in params)

            return goals
