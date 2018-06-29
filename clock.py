import abc

class Clock(abc.ABC):
    """An abstract class defining required interface for Clarion simulation 
    clocks.

    Implement simulation clocks using subclasses of this class in order to 
    ensure proper integration with Clarion constructs.
    """

    @abc.abstractmethod
    def get_time(self) -> float:
        """Return current time in simulation.
        """
        pass