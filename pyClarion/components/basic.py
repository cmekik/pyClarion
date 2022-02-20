"""Provides some basic propagators for building pyClarion agents."""


__all__ = ["Repeat", "Actions", "CAM", "Shift", "BoltzmannSampler", 
    "ActionSampler", "BottomUp", "TopDown", "AssociativeRules", "ActionRules"]


from ..base import Process, dimension, feature, chunk, rule, uris
from .. import numdicts as nd
from .utils import expand_dim, lag, group_by_dims, first, second, cf2cd, eye

import re
from typing import OrderedDict, Tuple, Dict, List, TypeVar
from functools import partial


T = TypeVar("T")


class Process1(Process):
    """A single output process."""    
    initial = nd.NumDict(c=0)


class Repeat(Process1):
    """Copies signal from a single source."""

    def call(self, d: nd.NumDict[T]) -> nd.NumDict[T]:
        return d


class Actions(Process1):
    """Represents external actions."""

    _cmd_pre = "cmd"

    def __init__(self, actions: Dict[str, List[str]]) -> None:
        self.actions = OrderedDict(actions)

    def call(self, c: nd.NumDict[feature]) -> nd.NumDict[feature]:
        return (c
            .drop(sf=lambda k: k.v is None)
            .transform_keys(kf=self._cmd2repr))

    def parse_actions(self, a: nd.NumDict) -> Dict[str, str]:
        """Return a dictionary of selected action values for each dimension"""
        
        result = {}
        for d, vs in self.actions.items():
            for v in vs:
                s = a[feature(expand_dim(d, self.prefix), v)]
                if s == 1 and d not in result and v is not None:
                    result[d] = v
                elif s == 1 and d in result and v is not None:
                    raise ValueError(f"Multiple values for action dim '{d}'")
                else:
                    continue
        return result

    def _cmd2repr(self, cmd: feature):
        _d, v, l = cmd
        if l != 0: raise ValueError("Lagged cmd not allowed.")
        d = re.sub(f"{uris.FSEP}{self._cmd_pre}/", f"{uris.FSEP}", _d)
        return feature(d, v)

    def _action_items(self):
        if len(self.actions) > 0:
            return list(zip(*self.actions.items()))
        else:
            return [], []

    @property
    def reprs(self) -> Tuple[feature, ...]:
        ds, vls = list(zip(*self.actions.items()))
        ds = expand_dim(ds, self.prefix) # type: ignore
        return tuple(feature(d, v) for d, vs in zip(ds, vls) for v in vs)

    @property
    def cmds(self) -> Tuple[feature, ...]:
        ds, vls = self._action_items()
        ds = [uris.SEP.join([self._cmd_pre, d]).strip(uris.SEP) # type: ignore
            for d in ds] 
        ds = expand_dim(ds, self.prefix) # type: ignore
        vls = [[None] + l for l in vls] # type: ignore
        return tuple(feature(d, v) for d, vs in zip(ds, vls) for v in vs)

    @property
    def nops(self) -> Tuple[feature, ...]:
        ds = [uris.SEP.join([self._cmd_pre, d]).strip(uris.SEP) # type: ignore
            for d in self.actions.keys()] 
        ds = expand_dim(ds, self.prefix) # type: ignore
        return tuple(feature(d, None) for d in ds)


class CAM(Process1):
    """Computes the combined-add-max activation for each node in a pool."""

    def call(self, *inputs: nd.NumDict[T]) -> nd.NumDict[T]:
        return nd.NumDict.eltwise_cam(*inputs)


class Shift(Process1):
    """Shifts feature strengths by one time step."""

    def __init__(
        self, lead: bool = False, max_lag: int = 1, min_lag: int = 0
    ) -> None:
        """
        Initialize a new `Lag` propagator.

        :param lead: Whether to lead (or lag) features.
        :param max_lag: Drops features with lags greater than this value.
        :param min_lag: Drops features with lags less than this value.
        """

        self.lag = partial(lag, val=1 if not lead else -1)
        self.min_lag = min_lag
        self.max_lag = max_lag

    def call(self, d: nd.NumDict[feature]) -> nd.NumDict[feature]:
        return d.transform_keys(kf=self.lag).keep(sf=self._filter)

    def _filter(self, f) -> bool:
        return type(f) == feature and self.min_lag <= f.dim.lag <= self.max_lag


class BoltzmannSampler(Process):
    """Samples a node according to a Boltzmann distribution."""

    initial = (nd.NumDict(), nd.NumDict())

    def call(
        self, p: nd.NumDict[feature], d: nd.NumDict[T], *_: nd.NumDict
    ) -> Tuple[nd.NumDict[T], nd.NumDict[T]]:
        """
        Select chunks through an activation-based competition. 
        
        Selection probabilities vary with chunk strengths according to a 
        Boltzmann distribution.

        :param d: Incoming feature activations.
        :param p: Incoming parameters (temperature & threshold). 
        """

        # assuming d.c == 0
        _d = d.keep_greater(ref=p.isolate(key=self.params[0]))
        if len(_d):
            dist = _d.boltzmann(p.isolate(key=self.params[1]))
            return dist.sample().squeeze(), dist
        else:
            return self.initial

    @property
    def params(self) -> Tuple[feature, ...]:
        return tuple(feature(dim) 
            for dim in expand_dim(("th", "temp"), self.prefix))


class ActionSampler(Process):
    """
    Selects actions and relays action paramaters.

    Expects to be linked to an external 'cmds' fspace.

    Actions are selected for each command dimension according to a Boltzmann 
    distribution.
    """

    initial = nd.NumDict(c=0), nd.NumDict(c=0)

    def call(
        self, p: nd.NumDict[feature], d: nd.NumDict[feature]
    ) -> Tuple[nd.NumDict[feature], nd.NumDict[feature]]:
        """
        Select actions for each client command dimension.
        
        :param p: Selection parameters (temperature). See self.params for 
            expected parameter keys.
        :param d: Action feature strengths.

        :returns: tuple (actions, distributions)
            where
            actions sends selected actions to 1 and everything else to 0, and
            distributions contains the sampling probabilities for each action
        """

        dims = group_by_dims(self.fspaces[0]())
        temp = p.isolate(key=self.params[0])
        _dists, _actions = [], [] 
        for fs in dims.values():
            dist = d.with_keys(ks=fs).boltzmann(temp)
            _dists.append(dist)
            _actions.append(dist.sample().squeeze())
        dists = nd.NumDict().merge(*_dists) 
        actions = nd.NumDict().merge(*_actions)

        return actions, dists

    def validate(self) -> None:
        nv = len(self.fspaces)
        if not nv:
            raise RuntimeError(f"Vocabs must be of length at least 1.")
        vt0 = self.fspaces[0].args[1] 
        if vt0 != "cmds":
            raise RuntimeError(f"Expected vocab type 'cmds', got '{vt0}'.")

    @property
    def params(self) -> Tuple[feature, ...]:
        return (feature(expand_dim("temp", self.prefix)),)


class BottomUp(Process1):
    """Propagates bottom-up activations."""

    def call(
        self, 
        fs: nd.NumDict[Tuple[chunk, feature]], 
        ws: nd.NumDict[Tuple[chunk, dimension]], 
        wn: nd.NumDict[chunk], 
        d: nd.NumDict[feature]
    ) -> nd.NumDict[chunk]:
        """
        Propagate bottom-up activations.
        
        :param fs: Chunk-feature associations (binary).
        :param ws: Chunk-dimension associations (i.e., top-down weights).
        :param wn: Normalization terms for each chunk. For each chunk, expected 
            to be equal to g(sum(|w|)), where g is some superlinear function.
        :param d: Feature strengths in the bottom level.
        """

        return (fs
            .put(d, kf=second, strict=True)
            .cam_by(kf=cf2cd)
            .mul_from(ws, kf=eye)
            .sum_by(kf=first)
            .squeeze()
            .div_from(wn, kf=eye, strict=True))


class TopDown(Process1):
    """Propagates top-down activations."""

    def call(
        self, 
        fs: nd.NumDict[Tuple[chunk, feature]], 
        ws: nd.NumDict[Tuple[chunk, dimension]], 
        d: nd.NumDict[chunk]
    ) -> nd.NumDict[feature]:
        """
        Propagate top-down activations.
        
        :param fs: Chunk-feature associations (binary).
        :param ws: Chunk-dimension associations (i.e., top-down weights).
        :param d: Chunk strengths in the top level.
        """

        return (fs
            .mul_from(d, kf=first, strict=True)
            .mul_from(ws, kf=cf2cd, strict=True)
            .cam_by(kf=second) 
            .squeeze())


class AssociativeRules(Process):
    """Propagates activations according to associative rules."""

    initial = (nd.NumDict(), nd.NumDict())

    def call(
        self, 
        cr: nd.NumDict[Tuple[chunk, rule]], 
        rc: nd.NumDict[Tuple[rule, chunk]], 
        d: nd.NumDict[chunk]
    ) -> Tuple[nd.NumDict[chunk], nd.NumDict[rule]]:
        """
        Propagate activations through associative rules.
        
        :param cr: Chunk-to-rule associations (i.e., condition weights).
        :param rc: Rule-to-chunk associations (i.e., conclusion weights; 
            typically binary).
        :param d: Condition chunk strengths.
        """

        norm = (cr
            .put(cr.abs().sum_by(kf=first), kf=first)
            .set_c(1))
        s_r = (cr
            .mul_from(d, kf=first, strict=True)
            .div(norm)
            .sum_by(kf=first)
            .squeeze())
        s_c = (rc
            .mul_from(s_r, kf=first, strict=True)
            .max_by(kf=first)
            .squeeze())

        return s_c, s_r


class ActionRules(Process):
    """Selects action chunks according to action rules."""

    initial = (nd.NumDict(), nd.NumDict(), nd.NumDict())

    def call(
        self, 
        p: nd.NumDict[feature], 
        cr: nd.NumDict[Tuple[chunk, rule]], 
        rc: nd.NumDict[Tuple[rule, chunk]], 
        d: nd.NumDict[chunk]
    ) -> Tuple[nd.NumDict[chunk], nd.NumDict[rule], nd.NumDict[rule]]:
        """
        Select actions chunks through action rules.
        
        :param p: Selection parameters (threshold and temperature). See 
            self.params for expected parameter keys.
        :param cr: Chunk-to-rule associations (i.e., condition weights).
        :param rc: Rule-to-chunk associations (i.e., conclusion weights; 
            typically binary).
        :param d: Condition chunk strengths.
        """

        # assuming d.c == 0
        _d = d.keep_greater(ref=p.isolate(key=self.params[0]))
        if len(_d) and len(cr):
            dist = (cr
                .mul_from(_d, kf=first)
                .boltzmann(p.isolate(key=self.params[1]))
                .transform_keys(kf=second)
                .squeeze())
            r_sampled = rc.put(dist.sample().squeeze(), kf=first)
            r_data = r_sampled.transform_keys(kf=first).squeeze()
            action = r_sampled.mul(rc).transform_keys(kf=second).squeeze()
            return action, r_data, dist
        else:
            return self.initial

    @property
    def params(self) -> Tuple[feature, ...]:
        return tuple(feature(dim) 
            for dim in expand_dim(("th", "temp"), self.prefix))
