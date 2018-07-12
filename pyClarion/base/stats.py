"""This module provides a base class and generic functions for tracking and
managing important statistics for the Clarion cognitive architecture.
"""


import abc
import typing as T

####### ABSTRACTION #######

class Statistic(abc.ABC):
    """Tracks a statistic.

    Instances are meant to be used as containers and updaters for the relevant 
    statistics. They do not store additional information about related 
    constructs, such as references to the objects of the statistics they store.
    """
    pass