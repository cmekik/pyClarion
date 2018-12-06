---
title: "pyClarion Readme"
author: "Can Serif Mekik"
date: 2018-12-01
version : 0.12.0
---

# pyClarion Readme

This is a python implementation of the Clarion cognitive architecture. 

The package is highly experimental. 

## Implementation Goals

This package aims to satisfy the following goals:

1. Be True to the Source Material: The code should reflect the source text(s) 
as closely as possible. However, superficial deviations from original 
descriptions are acceptable if strict adherence would lead to unnecessary 
complexity or inefficiency.
2. Have One Class For Each Construct: Every major construct should be 
represented by one class that encapsulates its core functions. This does not 
mean every class should correspond to a construct, there may be interfaces, 
abstractions, and utility classes that do not directly correspond to existing 
constructs.
3. Have a Simple Interface with the Simulation Environment: It should be easy 
to transfer data between Clarion constructs and the wild.
4. Prioritize Clarity Over Optimization: The code should be easily 
interpretable whenever possible. Code should not be optimized at the expense of 
clarity.
5. Have Minimal Dependencies: The code should rely on standard or major 
libraries whenever possible.
6. Be Compositional: The code architecture should allow for components to be 
swapped in and out easily. So, if only a subset of functionality is desired 
(e.g., for prototyping/experimentation), it should be possible to implement 
that subset with minimal effort.
7. Be Extensible: It should be easy to experiment with new constructs or 
alternative implementations.
8. Be Maintainable: Every object should have a singular, well-defined, 
documented role. Type hints should be included. Assumptions, exceptions, 
failure cases should be documented whenever possible. There should be thorough 
unit tests for all methods and functions. Python properties should be used 
whenever possible as they abstract property implementation from the return data 
type.

These goals are roughly in order from most to least specific. 

## General Architecture

In pyClarion, simulated constructs are named and represented with symbolic 
tokens called construct symbols. Each construct symbol may be associated with 
one or more construct realizers, which define and implement the behavior of the 
named constructs in a specific context.

The symbol-realizer distinction was developed as a solution to the following 
issues:

1. Information about constructs frequently needs to be communicated 
among various components. 
2. Constructs often exhibit complex behavior driven by intricate datastructures 
(e.g., artificial neural networks).
3. The same constructs may exhibit different behaviors within different 
subsystems of a given agent (e.g., the behavior of a same chunk may differ 
between the action-centered subsystem and the non-action-centered subsystem).
4. Construct behavior is not exactly prescribed by source material; several 
implementational possibilities are available for each class of constructs.

Construct symbols allow consistent and efficient communication of construct 
information using basic datastructures such as dicts, lists and sets of 
construct symbols. Construct realizers encapsulate complex behaviors associated 
with client constructs and provide a clean interface for multiple distinct 
realizations of the same construct.

## Installation

After downloading the repo, use `pip` with `setup.py`. In a command line, 
navigate to the pyClarion folder, then:

- To install in developer mode (recommended), run
```pip install -e .```
- To install as a regular library, run
```pip install .```

WARNING: Be sure to include the '`.`' in the install commands. Otherwise, your 
installation may fail.

Developer mode is recommended due to the experimental status of the library. 
Installing in this mode means that changes made to the pyClarion folder will be 
reflected in the pyClarion package, enabling fast prototyping and 
experimentation.