"""This module provides tools for representing and manipulating activations. 

Activations may propagate within the top level (from chunks to chunks), top-down 
(chunks to microfeatures), bottom-up (microfeatures to chunks), and within the 
bottom-level (microfeatures to microfeatures). Activations from different 
sources may also be combined. 

This module provides four constructs for representing and manipulating node 
activations. These are: 

- Activation packets (see submodule ``packet``), for representing activation 
  patterns.
- Activation channels (see submodule ``channel``), for representing activation 
  flows.
- Activation junctions (see submodule ``junction``), for combining activation 
  flows.
- Activation handlers (see submodule ``handler``), for handling activation 
  propagation through individual nodes.

The constructs are presented in order from lowest-level to highest-level.
"""