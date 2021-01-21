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

Developer mode is recommended to encourage and facilitate referring to pyClarion source code. Prior to installing in this mode, please ensure that the pyClarion folder is located at a convenient long-term location. Installing in developer mode means that changes made to the pyClarion folder will be reflected in the pyClarion package (moving the folder will break the package).

# Snippets

The following snippets illustrate the style and capabilities of the library. 

See the Detailed Examples and Design sections below for further details. 

**Agent Assembly**

```python
from pyClarion import (
    chunks, terminus, buffer, subsystem, agent,
    Construct, Structure,
    MaxNodes, BoltzmannSelector, Stimulus,
    Chunks, Assets
)

my_agent = Structure(
    name=agent("my_agent"),
)

with my_agent: # Automatically adds and links constructs defined in scope

    # Order of definition == Order of execution

    sensory = Construct(
        name=buffer("sensory"),
        process=Stimulus()
    )

    nacs = Structure(
        name=subsystem("nacs"),
        assets=Assets(
            cdb=Chunks()
        )
    )

    with nacs: # Nesting allowed

        chunk_pool = Construct(
            name=chunks("main"),
            process=MaxNodes(
                sources=[buffer("sensory")]
            )
        )

        Construct(
            name=terminus("retrieval"),
            process=BoltzmannSelector(
                source=chunks("main"),
                temperature=.01
            )
        )

assert my_agent[subsystem("nacs"), chunks("main")] == chunk_pool
```

**Rule and Chunk Specification**

```python
from pyClarion import feature, chunk, rule, Chunks, Rules

cdb = Chunks()
rdb = Rules()

rdb.define(
    rule(1),
    cdb.define( # chunk("conclusion") passed to rdb.define()
        chunk("conclusion"),
        feature("A"),
        feature("B")
    ),
    cdb.define(
        chunk("condition"),
        feature("X"),
        feature("Y"),
        feature("Z")
    )
)
```

**Mathematical Operations with Numerical Dictionaries**

```python
from pyClarion import nd

tape = nd.GradientTape()

with tape:
    d1 = nd.NumDict({1: 1.0, 2: 2.0})
    d2 = nd.NumDict({1: 3.0, 2: 4.0})
    d3 = d1 * d2 # elementwise multiplication by key
    d4 = nd.sum_by(d3, keyfunc=lambda x: 3) # Sum values mapping to same key

assert d4 == nd.NumDict({3: 11.0})

d4, grads = tape.gradients(d4, (d1, d2))
d1_grad, d2_grad = grads

assert d1_grad == nd.NumDict({1: 3.0, 2: 4.0})
assert d2_grad == nd.NumDict({1: 1.0, 2: 2.0})
```

# Detailed Examples

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

# Design

PyClarion views Clarion agents primarily as hierarchical networks of neural networks. Thus, constructing a pyClarion agent amounts to declaring what components exist, what they do, where they are placed in the hierarchy, and how they network with other components.

Simulated constructs are named and represented with symbolic tokens called construct symbols. Each construct symbol may be associated with one or more construct realizers, which define and implement the behavior of the named constructs in a specific context. 

Construct symbols allow consistent and efficient communication of construct information using basic datastructures such as dicts, lists and sets. Construct realizers encapsulate complex behaviors associated with client constructs and provide a clean interface for multiple distinct realizations of the same construct.

There are two realizer types: `Construct` and `Structure`. The `Construct` type is for basic constructs, which are the leaves of the construct hierarchy (e.g., buffers, feature pools, implicit decision networks, etc.). The `Structure` type is for constructs higher up in the hierarchy (e.g., subsystems and agents). 

A `Construct` instance pairs a construct symbol, a `Symbol` instance which names the construct represented by the realizer, with a `Process` instance. The process object is responsible for implementing the input/output and basic learning behavior associated with the simulated construct, while the realizer handles networking with other pyClarion components. Customized behaviors and/or components may be implemented by subclassing the `Process` class.


# Reading Guide

The pyClarion source code is organized as follows:

- `pyClarion/base/` contains basic resources and abstractions for defining clarion agents.

    This folder contains the following files:

    - `symbols.py` - Defines construct symbols.
    - `components.py` - Defines abstractions for processes as well as process domains and interfaces.
    - `realizers.py` - Defines realizers.

    The recommended reading order for `base/` is to start with `symbols.py`, then to move on to `realizers.py`. While reading `realizers.py`, refer to `components.py` as necessary. It may be useful to skim `components.py` prior to reading `realizers.py`, but reading it on its own may be confusing. 

- `pyClarion/components/` contains definitions for concrete component implementations. Assuming familiarity with `base/`, the files in this folder may be read in any order after an initial reading of `propagators.py`, which defines some basic process objects. Process objects make ample use of the tools provided in `pyClarion/numdicts/`, so it may be helpful to refer to this folder while reading `pyClarion/components/`. 

- `pyClarion/numdicts/` - Defines numerical dictionaries (numdicts), which are dict-like objects that support mathematical operations and automatic differentiation. Basic definitions are in `numdicts.py`; `funcs.py` provides useful functions on numdicts and `ops.py` defines functions over numdicts with automatic differentiation support.

- `pyClarion/utils/` contains definitions for various utilities. For now, it only contains `pprint.py`, which extends python's built-in pretty-printing tools to support various pyClarion objects.
