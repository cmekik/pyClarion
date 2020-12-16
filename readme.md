`pyClarion` is a python package for implementing agents in the Clarion cognitive architecture.

It is highly experimental and aims to be easy to learn, read, use and extend.

The primary resource for the implementation is Ron Sun's [*Anatomy of the Mind*](https://oxford.universitypressscholarship.com/view/10.1093/acprof:oso/9780199794553.001.0001/acprof-9780199794553). For a short overview of the architecture, see the chapter on Clarion in the [*Oxford Handbook of Cognitive Science*](https://www.oxfordhandbooks.com/view/10.1093/oxfordhb/9780199842193.001.0001/oxfordhb-9780199842193). 

# Key Features

- Pure `Python` (`>=3.7`)
- Highly modular & compositional
- Easily customizable

# Installation

After downloading the repo, use `pip` with `setup.py`. In a terminal, navigate to the pyClarion folder then:

- To install in developer mode (recommended), run
```pip install -e .```
- To install as a regular library, run
```pip install .```

WARNING: Be sure to include the '`.`' in the install commands. Otherwise, your installation will most likely fail.

Developer mode is recommended to encourage and facilitate referring to pyClarion source code. Prior to installing in this mode, please ensure that the pyClarion folder is located at a convenient long-term location. Installing in developer mode means that changes made to the pyClarion folder will be reflected in the pyClarion package.

# Examples

The `examples/` folder provides some simple examples demonstrating various aspects of using the pyClarion library to assemble and simulate Clarion agents.

The recommended reading order is as follows:

- `free_association.py` - Introduces the basic concepts of the pyClarion library using the example of a very simple free association setup.
- `flow_control.py` - An introduction to how pyClarion handles control through the example of using gates to select the mode of reasoning.
- `q_learning.py` - Demonstrates action-centered implicit learning in pyClarion. Trains a Q-network in the bottom level of the action-centered subsystem to perform a simple task.
- `chunk_extraction.py` - Demonstrates bottom-up learning in pyClarion through the example of a simple chunk extraction scenario.  
- `working_memory.py` - Demonstrates more complex patterns of control through the example of a simple question-answering scenario where the non-action-centered subsystem drives action selection in the action-centered subsystem through working memory.

The following examples are optional:

- `lagged_features.py` - Demonstrates how to set up lagged features, which may be useful in various contexts such as recurrent processing. May be read at any time after `free_association.py`.
- `gradients.py` - An overview of the automatic differentiation tools available in pyClarion. May be read at any time.

# Implementation Overview

PyClarion views Clarion agents primarily as hierarchical networks of neural networks. Thus, constructing a pyClarion agent amounts to declaring what components exist, what they do, where they are placed in the hierarchy, and how they network with other components.

Simulated constructs are named and represented with symbolic tokens called construct symbols. Each construct symbol may be associated with one or more construct realizers, which define and implement the behavior of the named constructs in a specific context. Construct symbols allow consistent and efficient communication of construct information using basic datastructures such as dicts, lists and sets of construct symbols. Construct realizers encapsulate complex behaviors associated with client constructs and provide a clean interface for multiple distinct realizations of the same construct.

Minimally, a construct realizer pairs a construct symbol, which names the construct represented by the realizer, with an `Emitter` object. The emitter is responsible for implementing the input/output and basic learning behavior associated with the simulated construct, while the realizer handles networking with other pyClarion components. Realizers may additionally be given `Updater` objects to handle more complex or customized learning behavior, and, in some cases, they may house resources shared by subordinate constructs (e.g., chunk and rule databases). To implement customized behaviors, it is sufficient to write suitable `Emitter` or `Updater` classes and pass instances of these custom classes to a construct realizer.

There are two realizer types: `Construct` and `Structure`. The `Construct` type is for basic constructs, which are the leaves of the construct hierarchy (e.g., buffers, feature pools, implicit decision networks, etc.). The `Structure` type is for constructs higher up in the hierarchy (e.g., subsystems and agents). The emitter type for `Construct` is `Propagator`, and, for `Structure`, it is `Cycle`. 

# Reading Guide

The pyClarion source code is organized as follows:

- `pyClarion/base/` contains basic resources and abstractions for defining clarion agents.

    This folder contains the following files:

    - `symbols.py` - Defines construct symbols.
    - `components.py` - Defines abstractions for emitters and updaters as well as feature domains and feature interfaces.
    - `realizers.py` - Defines realizers.

    The recommended reading order for `base/` is to start with `symbols.py`, then to move on to `realizers.py`. While reading `realizers.py`, refer to `components.py` as necessary. It may be useful to skim `components.py` prior to reading `realizers.py`, but reading it on its own may be confusing. 

- `pyClarion/components/` contains definitions for concrete component implementations. Assuming familiarity with `base/`, the files in this folder may be read in any order after an initial reading of `propagators.py` and `cycles.py`. The former defines some basic activation propagators, while the latter defines activation cycles at the agent and subsystem levels. Propagators make ample use of the tools provided in `pyClarion/numdicts/`, so it may be helpful to refer to this folder while reading `pyClarion/components/`. 

- `pyClarion/numdicts/` - Defines numerical dictionaries (numdicts), which are dict-like mappings that support mathematical operations and automatic differentiation. Basic definitions are in `numdicts.py`; `funcs.py` provides useful functions on numdicts and `ops.py` defines functions over numdicts with automatic differentiation support.

- `pyClarion/utils/` contains definitions for various utilities. For now, it only contains `pprint.py`, which extends python's built-in pretty-printing tools to support various pyClarion objects.
