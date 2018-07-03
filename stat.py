"""This module provides several classes for tracking important statistics for 
the Clarion cognitive architecture.

Classes are defined for tracking the following statistics:
    - Base-level activations (Sun, 2016, Sections 3.2.1.3 & 3.2.2.2)
    - Match statistics (Sun, 2016, Section 3.3.2.1)

Warning: In general, these classes are defined only as containers and updaters 
for the relevant statistics. They do not store additional information about 
related constructs, such as references to the objects of the statistics they 
store.

References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
"""


####### BASE-LEVEL ACTIVATIONS #######

class BLA(object):
    """Keeps track of base-level activations (BLAs).

    Implemented according to Sun (2016) Chapter 3. See Section 3.2.1.3 (p. 62)
    and also Section 3.2.2.2 (p. 77).

    Warning: This class has no knowledge of the chunks, rules and other 
    constructs associated with the statistics it tracks.

    References:
        Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(
        self,  
        initial_activation : float = 0., 
        amplitude : float = 2., 
        decay_rate : float = .5, 
        density: float = 0.
    ) -> None:
        """Initialize a BLA instance.

        Warning: By default, a timestamp is not added at initialization. If a 
        timestamp is required at creation time, call self.add_timestamp.
        """
        
        self.initial_activation = initial_activation
        self.amplitude = amplitude
        self.decay_rate = decay_rate
        self.density = density

        self.timestamps = []

    def update(self, current_time : float) -> None:
        """Record current time as an instance of use and/or activation for 
        associated construct.
        """

        self.timestamps.append(current_time)

    def compute_bla(self, current_time : float) -> float:
        """Compute the current BLA.

        Warning: Will result in division by zero if called immediately after 
        update.
        """

        summation_terms = [
            (current_time - t) ** (- self.decay_rate) 
            for t in self.timestamps
        ]
        bla = (
            self.initial_activation + (self.amplitude  *  sum(summation_terms))
        )
        return bla

    def below_density(self, current_time : float) -> bool:
        """Return true if BLA is below density.
        """

        return self.compute_bla(current_time) < self.density


####### MATCH STATISTICS #######

class MatchStatistics(object):
    """Tracks positive and negative match statistics.

    Implemented according to Sun (2016) Chapter 3. See Section 3.3.2.1 (p. 90).

    Warning: This class is not responsible for testing positive or negative 
    match criteria, and has no knowledge of the chunks and actions associated 
    with the statistics it tracks.  

    References:
        Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
    """

    def __init__(self) -> None:
        """Initialize tracking for a set of match statistics.
        """

        self.positive_matches = 0.
        self.negative_matches = 0.

    def update(self, positive_match : bool) -> None:
        """Update current match statistics.

        kwargs:
            positive_match : True if the positivity criterion is satisfied, 
            false otherwise.
        """

        if positive_match:
            self.positive_matches += 1.
        else: 
            self.negative_matches += 1.

    def discount(self, multiplier : float) -> None:
        """Discount match statistics.
        """
        
        self.positive_matches *= multiplier
        self.negative_matches *= multiplier