"""Basic tools for tracking and updating base-level activations."""


__all__ = ["BLAs"]


from collections import deque
from collections.abc import Mapping


class BLAs(Mapping):
    """A base level activation database."""

    class BLA(object):
        """
        A base-level activation (BLA) tracker.

        Computes or approximates:
            bla = b + sum(ts ** -d)
        Where b is a baseline, ts are time lags since previous invocations and 
        d is a density parameter controlling the rate of decay. Approximation 
        is recommended; exact computation included for completeness and 
        learning purposes.
        
        The implementation based on approximations presented in Petrov (2006), 
        though bear in mind that the Clarion equations are slightly different.

        Petrov, A. A. (2006). Computationally Efficient Approximation of the 
            Base-Level Learning Equation in ACT-R. In D. Fum, F. del Missier, & 
            A. Stocco (Eds.) Proccedings of the International Conference on 
            Cognitive Modeling (pp. 391-392).
        """

        __slots__ = ("baseline", "lags", "uses", "lifetime", "density")

        def __init__(self, baseline=0.0, density=0.5, depth=1):
            """
            Initialize a new BLA tracker.
            
            Implementation is based on a python deque.

            :param baeline: Initial BLA.
            :param density: Density parameter.
            :param depth: Depth of estimate. Setting k < 0 will result in using 
                exact computation. This option is inefficient in practice and 
                its use is discouraged.
            """

            self.baseline = baseline
            self.density = density

            self.lags = deque([1], maxlen=depth)
            self.uses = 1
            self.lifetime = 1

        def __repr__(self):

            tname = type(self).__name__
            value = self.value

            return "<{} {}>".format(tname, value)

        @property
        def value(self):
            """The current BLA value."""

            b = self.baseline
            lags = self.lags
            n = self.uses
            k = self.lags.maxlen
            t_n = self.lifetime
            d = self.density

            bla = b + sum([t ** -d for t in lags])
            if not k < 0 and k < n:
                t_k = lags[-1]
                factor = (n - k) / (1 - d)
                t_term = (t_n ** (1 - d) - t_k ** (1 - d)) / (t_n - t_k)
                distant_approx = factor * t_term
                bla += distant_approx

            return bla

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

    def __init__(self, baseline=0.0, density=0.5, depth=1):
        """
        Initialize a new BLA database.

        :param baseline: Default baseline value.
        :param density: Default density value.
        :param depth: Default depth value.
        """

        self.baseline = baseline
        self.density = density
        self.depth = depth

        self._dict = {}

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

    def add(self, key, baseline=None, density=None, depth=None):
        """
        Add key to BLA database.

        :param baseline: Baseline value, defaults to self.baseline if None.
        :param density: Density value, defaults to self.density if None.
        :param depth: Depth value, defaults to self.depth if None.
        """

        if baseline is None:
            baseline = self.baseline
        if density is None:
            density = self.density
        if depth is None:
            depth = self.depth

        self._dict[key] = self.BLA(baseline, density, depth)
