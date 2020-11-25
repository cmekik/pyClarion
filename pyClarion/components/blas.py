"""Basic tools for tracking and updating base-level activations."""


__all__ = ["BLAs", "RegisterArrayBLAUpdater"]


from ..base.symbols import ConstructType
from ..base.components import Inputs, UpdaterC
from .buffers import collect_cmd_data, RegisterArray

from typing import Any
from collections import deque
from collections.abc import Mapping


class BLAs(Mapping):
    """A base-level activation database."""

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
        self._new: set = set()

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

        self._dict[key] = self.BLA(
            self.density, self.baseline, self.amplitude, self.decay, self.depth
        )

    def update(self):
        """
        Update BLA database according to promises.

        Steps every existing BLA, adds invocations as promised. Also adds or 
        entries according to promises made. Does NOT remove any items.
        """

        for key, bla in self.items():
            if key in self._invoked:
                bla.step(invoked=True)
            else:
                bla.step(invoked=False)
        self._invoked.clear()

        for key in self._new:
            self.add(key)
        self._new.clear()

    def register_invocation(self, key, add_new=False):
        """
        Promise key will be treated as invoked on next update.
        
        If key does not already exist in self, add the key if add_new is True, 
        otherwise throw KeyError. 
        """

        if key in self:
            self._invoked.add(key)
        elif add_new:
            self._new.add(key)
        else:
            raise KeyError()

    def request_reset(self, key, add_new=False):
        """
        Promise BLA for key will be reset on next update.
        
        If key does not already exist in self, add the key if add_new is True, 
        otherwise throw KeyError. 
        """

        if key in self:
            self._reset.add(key)
        elif add_new:
            self._new.add(key)
        else:
            raise KeyError()

    def request_add(self, key):
        """Promise key will be added to database on next update."""

        self._new.add(key)


class RegisterArrayBLAUpdater(UpdaterC[RegisterArray]):
    """Maintains and updates BLA values associated RegisterArray slots."""

    _serves = ConstructType.buffer

    def __init__(self, density, baseline=0.0, amplitude=2, decay=.5, depth=1):

        self.blas = BLAs(density, baseline, amplitude, decay, depth)

    @property
    def expected(self):

        return frozenset()

    def __call__(
        self,
        propagator: RegisterArray,
        inputs: Inputs,
        output: Any,
        update_data: Inputs
    ) -> None:
        """
        Update register array according to BLAs.

        First updates BLAs, registering invocations for any slots that require 
        it, then clears any register slot whose content's BLA is found to be 
        below threshold. A slot is considered to be invoked iff it is written 
        to. Writing to a slot resets the BLA to its initial state.
        """

        client, controller = propagator.client, propagator.controller
        data = collect_cmd_data(client, inputs, controller)

        # Register items.
        for cell in propagator.cells:
            cmds = cell.interface.parse_commands(data)
            try:
                (_, val), = cmds.items()  # Extract unique cmd (dim, val) pair.
            except ValueError:
                msg = "{} expected exactly one command, received {}"
                raise ValueError(msg.format(type(self).__name__, len(cmds)))
            try:
                (item,) = cell.store
            except ValueError:
                n = len(cell.store)
                if n > 1:
                    msg = "{} expected exactly one item, cell contains {}"
                    raise ValueError(msg.format(type(self).__name__, n))
            else:
                if val in cell.interface.mapping:
                    self.blas.register_invocation(item, add_new=True)

        # Update BLAs.
        self.blas.update()

        # Remove items below threshold.
        for cell in propagator.cells:
            try:
                (item,) = cell.store
            except ValueError:
                n = len(cell.store)
                if n > 1:
                    msg = "{} expected exactly one item, cell contains {}"
                    raise ValueError(msg.format(type(self).__name__, n))
            else:
                if self.blas[item].below_threshold:
                    cell.clear_store()
                    del self.blas[item]
