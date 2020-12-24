"""Tools for handling goals."""


__all__ = ["GoalStay"]


from ..base.symbols import (
    ConstructType, Symbol, SymbolTrie, feature, chunk, terminus, buffer, 
    subsystem
)
from ..base.components import Propagator, FeatureInterface, FeatureDomain
from .blas import BLAs
from .buffers import collect_cmd_data
from .chunks_ import Chunks, ChunkExtractor
from .. import numdicts as nd

from typing import FrozenSet, Tuple, Hashable, Optional, cast
from types import MappingProxyType
from dataclasses import dataclass
from itertools import product, count


class GoalStay(Propagator):
    """
    A propagator for coordinating goal management.

    Warning: This component is very experimental.

    Maintains the active goal and various goal flags, executes goal commands 
    issued by a controller. May create new goal chunks or retrieve existing 
    goal chunks from a source, as directed. 

    Maintains two flags: the goal state flag and the goal eval flag. The goal 
    state flag serves to inform client processes whether a goal has been 
    newly initialized, resumed after an interruption, or is currently active.
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

    @dataclass
    class Interface(FeatureInterface):
        """
        Goal buffer control interface.

        :param goals: Feature domain specifying goals.
        :param prefix: Prefix for goal buffer interface features.
        :param set_postfix: Postfix for goal buffer commands and parameters. 
        :param state_postfix: Postfix for goal buffer state flag features.
        :param eval_postfix: Postfix for goal buffer eval flag features.
        :param set_vals: Values for goal buffer commands.
        :param state_vals: Values for goal buffer state flags.
        :param eval_vals: Values for goal buffer eval flags.
        """
        
        goals: FeatureDomain
        prefix: Hashable = "gb"
        set_postfix: Hashable = "set"
        state_postfix: Hashable = "state"
        eval_postfix: Hashable = "eval"
        set_vals: Tuple[Hashable, ...] = (
            "standby", "push", "exit", "pass", "fail", "activate"
        ) 
        state_vals: Tuple[Hashable, Hashable, Hashable, Hashable] = (
            "null", "start", "resume", "active"
        )
        eval_vals: Tuple[Hashable, Hashable, Hashable] = (
            "null", "passed", "failed"
        )

        @property
        def set_dim(self):

            return self._set_dim

        @property
        def state_dim(self):

            return self._state_dim

        @property
        def eval_dim(self):

            return self._eval_dim

        @property
        def set_cmds(self):

            return self._set_cmds

        @property
        def state_flags(self):

            return self._state_flags

        @property
        def eval_flags(self):

            return self._eval_flags

        @property
        def null_flags(self):

            return self._null_flags

        @property
        def goal_feature_map(self):

            return self._goal_feature_map

        def _validate_data(self):

            # need to check if goal params are distinct from other features, 
            # not possible without redundancy. Maybe _validate_data should in 
            # general be called after all features have been generated... -Can

            postfixes = self.set_postfix, self.state_postfix, self.eval_postfix
            set_vals = self.set_vals
            state_vals = self.state_vals
            eval_vals = self.eval_vals
            goals_by_dims = self.goals.features_by_dims

            if len(set_vals) != 6:
                raise ValueError("Must specify exactly 6 set vals.") 
            if len(set(set_vals)) < len(set_vals):
                raise ValueError("Goals and set vals must be unique.")
            if len(set(postfixes)) < len(postfixes):
                raise ValueError("Postfixes must be distinct")
            if len(set(state_vals)) < len(state_vals):
                raise ValueError("State vals must be unique.")
            if len(set(eval_vals)) < len(eval_vals):
                raise ValueError("Eval vals must be unique.")
            if any(len(fs) <= 1 for fs in goals_by_dims.values()):
                msg = "Singleton goal parameter dimensions not supported."
                raise ValueError(msg)

        def _set_interface_properties(self):

            set_tag = (self.prefix, self.set_postfix)
            state_tag = (self.prefix, self.state_postfix)
            eval_tag = (self.prefix, self.eval_postfix)

            set_vals = self.set_vals
            state_vals = self.state_vals
            eval_vals = self.eval_vals

            goals = self.goals
            goal_feature_map = {
                feature(set_tag + (f.tag,), f.val): f 
                for f in goals.features
            }

            f_set = tuple(feature(set_tag, v) for v in set_vals)
            f_state = tuple(feature(state_tag, v) for v in state_vals)
            f_eval = tuple(feature(eval_tag, v) for v in eval_vals)

            f_init = frozenset(feature(state_tag, v) for v in state_vals[1:3])

            null_eval = eval_vals[0]
            null_state = state_vals[0]
            null_flags = frozenset({
                feature(eval_tag, null_eval), feature(state_tag, null_state)
            })

            defaults = frozenset((feature(set_tag, set_vals[0]),))

            self._set_dim = (set_tag, 0)
            self._state_dim = (state_tag, 0)
            self._eval_dim = (eval_tag, 0)
            self._set_cmds = f_set
            self._state_flags = f_state
            self._eval_flags = f_eval
            self._null_flags = null_flags
            self._goal_feature_map = MappingProxyType(goal_feature_map)

            self._cmds = frozenset(f_set)
            self._defaults = defaults
            self._params = frozenset(goal_feature_map.keys())
            self._flags = frozenset(f_state) | frozenset(f_eval)


    def __init__(
        self, 
        controller: Tuple[subsystem, terminus], 
        source: Tuple[subsystem, terminus], 
        interface: Interface,
        chunks: Chunks,
        blas: BLAs,
        prefix: str = "goal"
    ) -> None:

        self.controller = controller
        self.source = source
        self.interface = interface
        self.chunks = chunks
        self.blas = blas
        self.prefix = prefix
        self._counter = count(1)

        self.store = nd.MutableNumDict(default=0)
        self.flags = nd.MutableNumDict(default=0)
        self.flags.extend(self.interface.null_flags, value=1.0)

    @property
    def expected(self) -> FrozenSet[Symbol]:
        
        return frozenset((self.controller[0], self.source[0]))

    def call(self, inputs: SymbolTrie[nd.NumDict]) -> nd.NumDict:
        
        data = collect_cmd_data(self.client, inputs, self.controller)
        cmds = self.interface.parse_commands(data)

        assert len(self.store) <= 1
        assert len({cast(feature, f).dim for f in self.flags}) == 2
        assert all(v == 1 for v in self.store.values())
        assert all(v == 1 for v in self.flags.values())

        set_dim = self.interface.set_dim
        standby_cmd = self.interface.set_vals[0]
        push_cmd = self.interface.set_vals[1]
        eval_cmds = self.interface.set_vals[2:5]
        active_cmd = self.interface.set_vals[5]
        goal_setting_cmds = self.interface.goals
        state_flags = self.interface.state_flags
        eval_flags = self.interface.eval_flags
        null_flags = self.interface.null_flags
        goal_params = self.interface.params
        goal_map = self.interface.goal_feature_map

        cmd = cmds[set_dim]
        if cmd == standby_cmd:
            pass
        if cmd == active_cmd:
            init_flags = state_flags[1:3] 
            active_flag = state_flags[-1]
            null_eval_flag = eval_flags[0]
            existing_state_flag = set(self.flags) & set(state_flags)
            flags_to_drop = existing_state_flag & set(init_flags)
            new_flags = {null_eval_flag} 
            if 0 < len(flags_to_drop): 
                new_flags.add(active_flag)
            else:
                new_flags.union(existing_state_flag)
            self.flags.clear()
            self.flags.extend(new_flags, value=1)
        elif cmd in eval_cmds:
            source_syst, source_term = self.source
            existing_goal, = self.store 
            new_goal = inputs[source_syst][source_term]
            if not isinstance(new_goal, nd.NumDict):
                raise TypeError("Expected NumDict.")
            self.store.clearupdate(new_goal)
            ch, = new_goal
            self.blas.register_invocation(ch)
            self.chunks.request_del(existing_goal)
            self.blas.request_del(existing_goal)
            if len(new_goal) == 0:
                self.flags.clear()
                self.flags.extend(null_flags, value=1)
            elif len(new_goal) == 1:
                i = list(eval_cmds).index(cmd)
                self.flags.clear()
                self.flags.extend((eval_flags[i], state_flags[2]), value=1)
            else:
                msg = "Expected only one new goal from goal structure."
                raise ValueError(msg)
        else:
            assert cmd == push_cmd
            _goal_fs = nd.keep(data, keys=goal_params)
            assert all(v == 1 for v in _goal_fs.values())
            assert len(_goal_fs) == len(self.interface.goals.features_by_dims)
            # Create a new chunk with the current goal features. Note: it is 
            # possible to have multiple goal chunks w/ identical goal 
            # parameters.
            # The update to chunk & bla database is immediate; adding this new 
            # chunk should ideally happen at update time. But this is seems 
            # like a less repetitive solution (though it is strongly dependent 
            # on the agent cycle).
            name = "{}_{}".format(self.prefix, next(self._counter))
            ch = chunk(name)
            goal_fs = {goal_map[f] for f in _goal_fs}
            form = self.chunks.Chunk(features=goal_fs)     
            self.chunks[ch] = form 
            self.blas.add(ch)
            self.store.clear()
            self.store[ch] = 1.0
            # Update flags
            start_flag = state_flags[1]
            null_eval_flag = eval_flags[0]
            self.flags.clear()
            self.flags.extend((start_flag, null_eval_flag), value=1)

        d = self.store + self.flags

        return d

         