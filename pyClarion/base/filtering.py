"""This module provides mixin classes for input and output filtering in the 
Clarion cognitive architecture.
"""

import typing as T
from . import node
from . import activation


class ChannelFilter(activation.Channel):
    """Filters mappings from nodes to values.

    This filter can be used for input/output filtering, assuming that 
    downstream activation channels have well-defined default behavior for 
    coping with missing expected inputs.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize an activation filter.

        kwargs:
            filter_map : A mapping from filter keys to node sets to be filtered. 
                There is no particular restriction on the type of filter keys.
        """

        super().__init__(*args, **kwargs)
        self.input_filter : node.NodeSet = set()
        self.output_filter : node.NodeSet = set()

    def __call__(
        self, input_map : node.Node2Float,
    ) -> node.Node2Float:
        """Return a filtered mapping from nodes to values.

        kwargs:
            input_map : An activation map to be filtered.
        """

        filtered_input = {
            k:v for (k,v) in input_map.items() if k not in self.input_filter
        }
        raw_output = super().__call__(filtered_input)
        filtered_output = {
            k:v for (k,v) in raw_output.items() if k not in self.output_filter
        }
        return filtered_output