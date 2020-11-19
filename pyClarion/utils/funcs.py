
from ..base.symbols import feature
import random
import math

from typing import Iterable, Dict, Hashable, Tuple
from itertools import groupby
import logging


__all__ = ["collect_cmd_data"]


def collect_cmd_data(construct, inputs, controller):

    subsystem, terminus = controller
    try:
        data = inputs[subsystem][terminus]
    except KeyError:
        data = frozenset()
        msg = "Failed data pull from %s in %s."
        logging.warning(msg, controller, construct)
    
    return data
