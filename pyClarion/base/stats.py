"""This module provides a base class and generic functions for tracking and
managing important statistics for the Clarion cognitive architecture.
"""


import abc


####### ABSTRACTION #######

class Stat(abc.ABC):
    """Tracks a statistic.

    Instances are meant to be used as containers and updaters for the relevant 
    statistics. They do not store additional information about related 
    constructs, such as references to the objects of the statistics they store.
    """

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass


####### GENERIC FUNCTIONS #######

def initialize_stat_map(stat, *constructs):
    """
    """

    stats =  {
        construct : stat() for construct in constructs  
    }
    return stats