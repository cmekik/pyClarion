"""Basic tools for tracking and updating base-level activations."""


__all__ = ["BLA", "BLAs", "BLAStrengths", "BLAMaintainer"]


# from .buffers import RegisterArray
from ..base.symbols import ConstructType, Symbol, SymbolTrie, SymbolicAddress
from ..base.components import Process, CompositeProcess
from .. import numdicts as nd

from typing import Any, FrozenSet, Sequence, cast
from collections import deque
from collections.abc import Mapping, MutableMapping


class BLA(object):
    """
    A base-level activation (BLA) tracker.

    Computes or approximates:
        bla = b + c * sum(ts ** -d)
    Where b is a baseline parameter, c is an amplitude parameter, ts are 
    time lags since previous invocations and d is a decay parameter 
    controlling the rate of decay. Approximation is recommended; exact 
    computation option included for completeness and learning purposes 
    only.

    Follows equations presented in Petrov (2006), though bear in mind that 
    the Clarion equations are slightly different (e.g. no logarithm).

    Petrov, A. A. (2006). Computationally Efficient Approximation of the 
        Base-Level Learning Equation in ACT-R. In D. Fum, F. del Missier, & 
        A. Stocco (Eds.) Proccedings of the International Conference on 
        Cognitive Modeling (pp. 391-392).
    """

    __slots__ = (
        "density", "baseline", "amplitude", "decay", "lags", "uses",
        "lifetime"
    )

    def __init__(
        self,
        density: float,
        baseline: float = 0.0,
        amplitude: float = 2.0,
        decay: float = 0.5,
        depth: int = 1
    ):
        """
        Initialize a new BLA tracker.

        :param density: The density parameter.
        :param baseline: Initial BLA.
        :param amplitude: Amplitude parameter.
        :param decay: Decay parameter.
        :param depth: Depth of estimate. Setting k < 0 will result in using 
            exact computation. This option is inefficient in practice and 
            its use is discouraged.
        """

        self.density = density
        self.baseline = baseline
        self.amplitude = amplitude
        self.decay = decay

        maxlen = depth if 0 <= depth else None
        self.lags = deque([1], maxlen=depth)
        self.uses = 1
        self.lifetime = 1

    def __repr__(self):

        tname = type(self).__name__
        value = self.value
        density = self.density

        return "<{} val={} den={}>".format(tname, value, density)

    @property
    def value(self):
        """The current BLA value."""

        b = self.baseline
        c = self.amplitude
        lags = self.lags
        n = self.uses
        k = self.lags.maxlen
        t_n = self.lifetime
        d = self.decay

        bla = b + c * sum([t ** -d for t in lags])
        if not k < 0 and k < n:
            t_k = lags[-1]
            factor = (n - k) / (1 - d)
            t_term = (t_n ** (1 - d) - t_k ** (1 - d)) / (t_n - t_k)
            distant_approx = factor * t_term
            bla += c * distant_approx

        return bla

    @property
    def below_threshold(self):
        """Return True iff value is below set density parameter."""

        return self.value < self.density

    def step(self, invoked=False):
        """
        Advance the BLA tracker by one time step.

        In all cases, will increment all time lags by 1. If invoked is 
        True, will add a new use entry with lag value 1 on return and 
        increment the use counter. 
        """

        if invoked:
            self.lags.appendleft(0)
            self.uses += 1

        for i in range(len(self.lags)):
            self.lags[i] += 1
        self.lifetime += 1

    def reset(self):
        """Reset BLA tracker to initial state."""

        maxlen = self.lags.maxlen
        self.lags = deque([1], maxlen=maxlen)
        self.uses = 1
        self.lifetime = 1


class BLAs(Mapping):
    """A base-level activation database."""

    def __init__(
        self,
        density: float,
        baseline: float = 0.0,
        amplitude: float = 2.0,
        decay: float = 0.5,
        depth: int = 1
    ):
        """
        Initialize a new BLA database.

        :param baseline: Baseline parameter.
        :param amplitude: Amplitude parameter.
        :param decay: Decay parameter.
        :param depth: Depth parameter.
        """

        self.density = density
        self.baseline = baseline
        self.amplitude = amplitude
        self.decay = decay
        self.depth = depth

        self._dict: dict = {}
        self._invoked: set = set()
        self._reset: set = set()
        self._del: set = set()
        self._add: set = set()

    def __repr__(self):

        return "<BLAs {}>".format(self._dict)

    def __len__(self):

        return len(self._dict)

    def __iter__(self):

        yield from self._dict

    def __getitem__(self, key):

        return self._dict[key]

    def __delitem__(self, key):

        del self._dict[key]

    def add(self, key):
        """Add key to BLA database."""

        self._dict[key] = BLA(
            self.density, 
            self.baseline, 
            self.amplitude, 
            self.decay, 
            self.depth
        )

    def step(self):
        """
        Update BLA database according to promises.

        Steps every existing BLA, adds invocations as promised. Also adds 
        and removes entries according to promises made.
        """

        for key in self._del:
            del self[key]
        self._del.clear()

        for key, bla in self.items():
            if key in self._invoked:
                bla.step(invoked=True)
            else:
                bla.step(invoked=False)
        self._invoked.clear()

        for key in self._add:
            self.add(key)
        self._add.clear()

    def register_invocation(self, key, add_new=False):
        """
        Promise key will be treated as invoked on next update.
        
        If key does not already exist in self, add the key if add_new is True, 
        otherwise throw KeyError. 
        """

        if key in self._add or key in self._del or key in self._invoked:
            msg = "Key {} already registered for a promised update."
            raise ValueError(msg.format(key))
        else:
            if key in self:
                self._invoked.add(key)
            elif add_new:
                self._add.add(key)
            else:
                raise KeyError("Key not in BLA database.")

    def request_add(self, key):
        """Promise key will be added to database on next update."""

        if key in self._add or key in self._del or key in self._invoked:
            msg = "Key {} already registered for a promised update."
            raise ValueError(msg.format(key))
        else:
            self._add.add(key)

    def request_del(self, key):
        """Promise key will be deleted from database on next update."""

        if key in self._add or key in self._del or key in self._invoked:
            msg = "Key {} already registered for a promised update."
            raise ValueError(msg.format(key))
        else:
            self._del.add(key)


class BLAStrengths(Process):
    """
    Emit strengths for items based on their BLAs.

    Assigns to each item in a BLA database a strength equal to tanh(bla), where 
    bla is the BLA value associated with the item. The tanh function is used to 
    squash the BLA value to lie in [0, 1]. Optionally, the BLA can be scaled 
    before being squashed. Thresholding is also available.
    """

    _serves = ConstructType.flow_in

    def __init__(self, blas: BLAs, r: float = 1.0, th: float = 0.0) -> None:

        super().__init__()

        self.blas = blas
        self.th = th
        self.r = r

    def call(self, inputs):

        th = self.th
        r = self.r
        items = self.blas.items()

        d = nd.NumDict({k: r * v.value for k, v in items}, default=0)
        d = nd.threshold(d, th=th, keep_default=True)
        d = nd.tanh(d) # Squash [0, +inf] to [0, 1]

        return d


class BLAMaintainer(Process):
    """
    Maintains BLAs.

    Records invocations and culls entries that fall below density threshold.
    """

    # TODO: Needs testing. - Can

    _serves = ConstructType.updater

    def __init__(
        self, 
        sources: Sequence[SymbolicAddress], 
        blas: BLAs, 
        client_db: MutableMapping = None, # Client db is optional...
        threshold: float = 0.0
    ) -> None:

        super().__init__(expected=sources)
        self.blas = blas
        self.client_db = client_db
        self.threshold = threshold

    def call(self, inputs: SymbolTrie[nd.NumDict]) -> nd.NumDict:
        """
        Update BLA database and client DB.
        
        Issues a call to BLAs.register_invocation() with option add_new=True 
        for each element above threshold in monitored outputs. After 
        registering invocations, steps the bla database. Finally, evaluates each
        BLA entry against its density parameter. Entries below threshold will be
        removed both from client db AND the BLA database.
        """

        data = self.extract_inputs(inputs)

        for strengths in data:
            for c, v in strengths.items():
                if v > self.threshold:
                    self.blas.register_invocation(c, add_new=True)

        self.blas.step()

        items = tuple(self.blas.items())
        for entry, bla in items:
            if bla.below_threshold:
                del self.blas[entry]
                if self.client_db is not None:
                    del self.client_db[entry]

        return super().call(inputs) 
