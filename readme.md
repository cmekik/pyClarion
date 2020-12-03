`pyClarion` is a python package for implementing agents according to the Clarion cognitive architecture.

It is highly experimental, and aims to be easy to learn, read, extend, and experiment with.

The primary resource for the implementation is Ron Sun's *Anatomy of the Mind* (2016; OUP).

# Key Features

- Pure `Python` (`>=3.7`)
- Highly modular & compositional
- Easily customizable

# Installation

After downloading the repo, use `pip` with `setup.py`. In a terminal, navigate to the pyClarion folder, then:

- To install in developer mode (recommended), run
```pip install -e .```
- To install as a regular library, run
```pip install .```

WARNING: Be sure to include the '`.`' in the install commands. Otherwise, your installation may fail.

Developer mode is recommended due to the experimental status of the library. Installing in this mode means that changes made to the pyClarion folder will be reflected in the pyClarion package, enabling fast prototyping and experimentation.

# Examples

The `examples/` folder provides some simple examples demonstrating various aspects of using the `pyClarion` library to assemble and simulate Clarion agents.

The recommended reading order is as follows:

- `free_association.py` - Introduces the basic concepts of the `pyClarion` library using the example of a very simple free association task.
- `lagged_features.py` - Demonstrates how to set up lagged features, which may be useful in various contexts such as recurrent processing and temporal difference learning.
- `flow_control.py` - An introduction to how `pyClarion` handles control through the example of using gates to select the mode of reasoning.
- `chunk_extraction.py` - An introduction to how `pyClarion` supports learning processes. Demonstrates a simple case of learning through chunk extraction.  
- `working_memory.py` - An introduction to more complex modeling using `pyClarion`. Demonstrates a simple case of question answering, where the non-action-centered subsystem drives action selection in the action-centered subsystem through working memory.
- `gradients.py` - An overview of the minimalistic automatic differentiation functionality available in `pyClarion`. May be read at any time.

# Implementation Overview

`pyClarion` views Clarion agents as a hierarchical networks of neural networks. Thus, constructing a `pyClarion` agent amounts to declaring what components exist, what they do, where they are placed in the hierarchy, and how they network with other components.

Simulated constructs are named and represented with symbolic tokens called construct symbols. Each construct symbol may be associated with one or more construct realizers, which define and implement the behavior of the named constructs in a specific context. Construct symbols allow consistent and efficient communication of construct information using basic datastructures such as dicts, lists and sets of construct symbols. Construct realizers encapsulate complex behaviors associated with client constructs and provide a clean interface for multiple distinct realizations of the same construct.

Minimally, a construct realizer pairs a construct symbol, which names the construct represented by the realizer, with an `Emitter` object. The emitter is responsible for implementing the input/output and basic learning behavior associated with the simulated construct, while the realizer handles networking with other `pyClarion` components. Realizers may additionally be given `Updater` objects to handle more complex or customized learning behavior, and, in some cases, they may house resources shared by subordinate constructs (e.g., chunk and rule databases). To implement customized behaviors, it is sufficient to write suitable `Emitter` or `Updater` classes and pass them to a construct realizer.

# Reading Guide

The `pyClarion` library source code is organized as follows:

- `pyClarion/base/` contains definitions for the basic abstractions used by the library.

    This folder contains the following files:

    - `symbols.py` - Defines construct symbols.
    - `numdicts.py` - Defines numerical dictionaries, which are essentially dicts that support mathematical operations.
    - `gradients.py` - Defines some simple tools for supporting basic automatic differentiation.
    - `components.py` - Defines basic abstractions for defining emitters and updaters.
    - `realizers.py` - Defines realizer objects.

    The recommended reading order for `base/` is to start with `symbols.py` or `numdicts.py`, then to move on to `realizers.py`. While reading `realizers.py`, refer to `components.py` as necessary. Reading `components.py` on its own may be confusing. `gradients.py` may be read at any time after reading `numdicts.py`.

- `pyClarion/components/` contains definitions for concrete component implementations. Assuming familiarity with `base/`, the files in this folder may be read in any order after an initial reading of `propagators.py` and `cycles.py`. The former defines some basic emitters for basic constructs, while the latter defines activations sequences at the agent and subsystem levels. 

- `pyClarion/utils/` contains definitions for various utilities. For now, it only contains `pprint.py`, which extends python's built-in pretty-printing tools to support various `pyClarion` objects.
