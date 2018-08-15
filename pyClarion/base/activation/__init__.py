"""This module provides tools for computing and handling node activations in the 
Clarion cognitive architecture. 

Activations may propagate within the top level (from chunks to chunks), top-down 
(chunks to microfeatures), bottom-up (microfeatures to chunks), and within the 
bottom-level (microfeatures to microfeatures). Activations from different 
sources may also be combined. 

The processes described above are captured by means of two main abstractions: 
activation channels (Channel class) and junctions (Junction class). Channels 
implement mappings from node activations to node activations. Junctions 
implement routines for combining inputs from multiple channels.Other useful 
activation handlers include splits (Split class), which are meant to 
handle splitting activations into multiple streams, and selectors (Selector 
class), which choose actionable chunks on the basis of chunk activations.

In addition to defining the above, this module provides severeal utilities 
(classes and functions) for defining, filtering, and handling activation flows.

For details of activation flows, see Chapter 3 of Sun (2016). Also, see Chapter 
4 for a discussion of filtering capabilities of MCS.

References:
    Sun, R. (2016). Anatomy of the Mind. Oxford University Press.
"""