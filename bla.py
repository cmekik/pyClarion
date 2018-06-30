import typing as T
from clock import Clock

class BLA(object):
    """Generic class to keep track of base-level activations (BLAs). Can be 
    mixed into various constructs using BLAMixin class or used in a standalone 
    fashion.

    Implemented according to Sun (2016) Chapter 3. See Section 3.2.1.3 (p. 62)
    and also Section 3.2.2.2 (p. 77).

    References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(
        self,
        clock : Clock,  
        initial_activation : float = 0., 
        amplitude : float = 2., 
        decay_rate : float = .5, 
        density: float = 0.
    ) -> None:
        """Initialize a BLA instance.

        Warning: By default, a timestamp is not added at initialization. If a 
        timestamp is required at creation time, call self.add_timestamp.
        """
        
        self.clock = clock
        self.initial_activation = initial_activation
        self.amplitude = amplitude
        self.decay_rate = decay_rate
        self.density = density

        self.timestamps = []

    def update(self) -> None:
        """Record current time as an instance of use and/or activation for 
        associated construct.
        """

        self.timestamps.append(self.clock.get_time())

    def compute_bla(self) -> float:
        """Compute the current BLA.

        Warning: Will result in division by zero if called immediately after 
        update.
        """

        current_time = self.clock.get_time()
        summation_terms = [
            (current_time - t) ** (- self.decay_rate) 
            for t in self.timestamps
        ]
        bla = self.initial_activation + self.amplitude  *  sum(summation_terms)
        return bla

    def below_density(self) -> bool:
        """Return true if BLA is below density.
        """

        return self.compute_bla() < self.density

class BLAMixin(object):
    """A Mixin class providing an interface for adding BLA functionality to 
    Clarion constructs.
    """

    def __init__(self, bla : BLA, *args, **kwargs) -> None:
        """Initialize parent class with BLA mixed in.

        Args:
           bla : BLA instance intialized with desired parameters.
        """
        
        super().__init__(*args, **kwargs)
        self.bla_handler = bla

    def update_bla(self) -> None:
        """Update BLA of self.

        Calls BLA.update. See BLA.update for further details.
        """

        self.bla_handler.update()

    def get_bla(self) -> float:
        """Return current BLA of self.

        Calls BLA.compute_bla. See BLA.compute_bla for further details.
        """

        return self.bla_handler.compute_bla()

    def bla_is_below_density(self) -> bool:
        """Return true if BLA of self is below density.

        Calls BLA.below_density. See BLA.below_density for further details.
        """

        return self.bla_handler.below_density()