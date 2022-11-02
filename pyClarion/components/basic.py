"""Provides some basic propagators for building pyClarion agents."""


__all__ = ["Repeat", "Actions", "CAM", "Shift", "BoltzmannSampler", 
    "ActionSampler", "BottomUp", "TopDown", "AssociativeRules", "ActionRules"]


from ..base import dimension, feature, chunk, rule
from .. import numdicts as nd
from .. import dev as cld

import re
from typing import (OrderedDict, Tuple, Dict, List, TypeVar, Union, Sequence, 
    Generator)
from functools import partial


T = TypeVar("T")


class Repeat(cld.Process):
    """Copies signal from a single source."""

    initial = nd.NumDict()

    def call(self, d: nd.NumDict[T]) -> nd.NumDict[T]:
        return d


class Receptors(cld.Process):
    """Represents a perceptual channel."""

    initial = nd.NumDict()

    def __init__(
        self, 
        reprs: Union[List[str], Dict[str, List[str]], Dict[str, List[int]]]
    ) -> None:
        self._reprs = reprs
        self._data: nd.NumDict[feature] = nd.NumDict()

    def call(self) -> nd.NumDict[feature]:
        return self._data

    def stimulate(
        self, data: Union[List[Union[str, Tuple[str, Union[str, int]]]], 
            Dict[Union[str, Tuple[str, Union[str, int]]], float]]
    ) -> None:
        """
        Set perceptual stimulus levels for defined perceptual features.
        
        :param data: Stimulus data. If list, each entry is given an activation 
            value of 1.0. If dict, each key is set to an activation level equal 
            to its value.
        """
        if isinstance(data, list):
            self._data = nd.NumDict({f: 1.0 for f in self._fseq(data)})
        elif isinstance(data, dict):
            fspecs, strengths = zip(*data.items())
            fseq = self._fseq(fspecs) # type: ignore
            self._data = nd.NumDict({f: v for f, v in zip(fseq, strengths)})
        else:
            raise ValueError("Stimulus spec must be list or dict, "
                f"got {type(data)}.")

    def _fseq(
        self, data: Sequence[Union[str, Tuple[str, Union[str, int]]]]
    ) -> Generator[feature, None, None]:
        for x in data:
            if isinstance(x, tuple):
                f = feature(cld.prefix(x[0], self.prefix), x[1]) 
            else:
                f = feature(cld.prefix(x, self.prefix)) 
            if f in self.reprs:
                yield f
            else:
                raise ValueError(f"Unexpected stimulus feature spec: '{x}'")

    @property
    def reprs(self) -> Tuple[feature]:
        if isinstance(self._reprs, list):
            return tuple(feature(cld.prefix(x, self.prefix)) 
                for x in self._reprs)
        else:
            assert isinstance(self._reprs, dict)
            return tuple(
                feature(cld.prefix(d, self.prefix), v) if len(l) 
                else feature(cld.prefix(d, self.prefix)) 
                for d, l in self._reprs.items() for v in l 
            )


class Actions(cld.Process):
    """Represents external actions."""

    initial = nd.NumDict()
    _cmd_pre = "cmd"

    def __init__(
        self, 
        actions: Union[Dict[str, List[str]], Dict[str, List[int]]]
    ) -> None:
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
                s = a[feature(cld.prefix(d, self.prefix), v)]
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
        d = re.sub(f"{cld.FSEP}{self._cmd_pre}-", f"{cld.FSEP}", _d)
        return feature(d, v)

    def _action_items(self):
        if len(self.actions) > 0:
            return list(zip(*self.actions.items()))
        else:
            return [], []

    @property
    def reprs(self) -> Tuple[feature, ...]:
        ds, vls = list(zip(*self.actions.items()))
        ds = cld.prefix(ds, self.prefix) # type: ignore
        return tuple(feature(d, v) for d, vs in zip(ds, vls) for v in vs)

    @property
    def cmds(self) -> Tuple[feature, ...]:
        ds, vls = self._action_items()
        ds = ["-".join(filter(None, [self._cmd_pre, d])) # type: ignore
            for d in ds] 
        ds = cld.prefix(ds, self.prefix) # type: ignore
        vls = [[None] + l for l in vls] # type: ignore
        return tuple(feature(d, v) for d, vs in zip(ds, vls) for v in vs)

    @property
    def nops(self) -> Tuple[feature, ...]:
        ds = ["-".join(filter(None, [self._cmd_pre, d])) # type: ignore
            for d in self.actions.keys()] 
        ds = cld.prefix(ds, self.prefix) # type: ignore
        return tuple(feature(d, None) for d in ds)


class CAM(cld.Process):
    """Computes the combined-add-max activation for each node in a pool."""

    initial = nd.NumDict()

    def call(self, *inputs: nd.NumDict[T]) -> nd.NumDict[T]:
        return nd.NumDict.eltwise_cam(*inputs)


class Shift(cld.Process):
    """Shifts feature strengths by one time step."""

    initial = nd.NumDict()

    def __init__(
        self, lead: bool = False, max_lag: int = 1, min_lag: int = 0
    ) -> None:
        """
        Initialize a new `Lag` propagator.

        :param lead: Whether to lead (or lag) features.
        :param max_lag: Drops features with lags greater than this value.
        :param min_lag: Drops features with lags less than this value.
        """

        self.lag = partial(cld.lag, val=1 if not lead else -1)
        self.min_lag = min_lag
        self.max_lag = max_lag

    def call(self, d: nd.NumDict[feature]) -> nd.NumDict[feature]:
        return d.transform_keys(kf=self.lag).keep(sf=self._filter)

    def _filter(self, f) -> bool:
        return type(f) == feature and self.min_lag <= f.dim.lag <= self.max_lag


class BoltzmannSampler(cld.Process):
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
            for dim in cld.prefix(("th", "temp"), self.prefix))


class ActionSampler(cld.Process):
    """
    Selects actions and relays action paramaters.

    Expects to be linked to an external 'cmds' fspace.

    Actions are selected for each command dimension according to a Boltzmann 
    distribution.
    """

    initial = (nd.NumDict(), nd.NumDict())

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

        dims = cld.group_by_dims(self.fspaces[0]())
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
        return (feature(cld.prefix("temp", self.prefix)),)


class BottomUp(cld.Process):
    """Propagates bottom-up activations."""

    initial = nd.NumDict()

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
            .put(d, kf=cld.second, strict=True)
            .cam_by(kf=cld.cf2cd)
            .mul_from(ws, kf=cld.eye)
            .sum_by(kf=cld.first)
            .squeeze()
            .div_from(wn, kf=cld.eye, strict=True))


class TopDown(cld.Process):
    """Propagates top-down activations."""

    initial = nd.NumDict()

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
            .mul_from(d, kf=cld.first, strict=True)
            .mul_from(ws, kf=cld.cf2cd, strict=True)
            .cam_by(kf=cld.second) 
            .squeeze())


class AssociativeRules(cld.Process):
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
            .put(cr.abs().sum_by(kf=cld.first), kf=cld.first)
            .set_c(1))
        s_r = (cr
            .mul_from(d, kf=cld.first, strict=True)
            .div(norm)
            .sum_by(kf=cld.second)
            .squeeze())
        s_c = (rc
            .mul_from(s_r, kf=cld.first, strict=True)
            .max_by(kf=cld.second)
            .squeeze())

        return s_c, s_r


class ActionRules(cld.Process):
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
                .mul_from(_d, kf=cld.first)
                .boltzmann(p.isolate(key=self.params[1]))
                .transform_keys(kf=cld.second)
                .squeeze())
            r_sampled = rc.put(dist.sample().squeeze(), kf=cld.first)
            r_data = r_sampled.transform_keys(kf=cld.first).squeeze()
            action = r_sampled.mul(rc).transform_keys(kf=cld.second).squeeze()
            return action, r_data, dist
        else:
            return self.initial

    @property
    def params(self) -> Tuple[feature, ...]:
        return tuple(feature(dim) 
            for dim in cld.prefix(("th", "temp"), self.prefix))
