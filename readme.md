`pyClarion` is a python package for implementing agents in the Clarion cognitive architecture.

It is highly experimental and aims to be easy to learn, read, use and extend.

The primary resource for the implementation is Ron Sun's [*Anatomy of the Mind*](https://oxford.universitypressscholarship.com/view/10.1093/acprof:oso/9780199794553.001.0001/acprof-9780199794553). For a short overview of the architecture, see the chapter on Clarion in the [*Oxford Handbook of Cognitive Science*](https://www.oxfordhandbooks.com/view/10.1093/oxfordhb/9780199842193.001.0001/oxfordhb-9780199842193). 

# Key Features

- Convenient and modular agent assembly
```python
import pyClarion as cl
from pyClarion import nd

# Order of definition == Order of activation

# Constructs defined within a Structure context are automatically linked
with cl.Structure("agent") as agent: 

    # sensory stimulus module to maintain stimulus
    cl.Module(name="stim", process=Repeat(), i_uris=["stim"])

    # parameter module to house process parameters
    p = cl.Module(name="p", process=Repeat(), i_uris=["p"])

    # Nested structures allowed
    with cl.Structure("nacs") as nacs: 

        # chunk pool to combine different chunk strength recommendations
        cp = cl.Module(name="cp", process=CAM(), i_uris=["../stim"])

        # boltzmann sampler to select chunks for retrieval
        cl.Module(name="ret", process=BoltzmannSampler(), i_uris=["p", "cp"])

# set boltzmann sampler's temperature parameter in the parameter module
p.output = nd.NumDict({cl.feature("nacs/ret#temp"): 1e-3})

# Access objects with uris
assert agent["nacs/cp"] == cp
```
- A simple language for initializing explicit knowledge  
```yaml
rule:
    conc:
        A
        B
    cond:
        X
        Y
        Z
```
- Numerical dictionaries with autodiff support
```python
from pyClarion import nd

tape = nd.GradientTape()

with tape:
    d1 = nd.NumDict({1: 1.0, 2: 2.0})
    d2 = nd.NumDict({1: 3.0, 2: 4.0})
    d3 = d1 * d2 # elementwise multiplication by key
    d4 = d3.sum_by(kf=lambda x: 3) # Sum values mapping to same key

assert d4 == nd.NumDict({3: 11.0})

d4, grads = tape.gradients(d4, (d1, d2))
d1_grad, d2_grad = grads

assert d1_grad == nd.NumDict({1: 3.0, 2: 4.0})
assert d2_grad == nd.NumDict({1: 1.0, 2: 2.0})
```

# Installation

In a terminal, navigate to the pyClarion folder then:

- To install in developer mode (recommended), run
```pip install -e .```
- To install as a regular library, run
```pip install .```

WARNING: Be sure to include the '`.`' in the install commands. Otherwise, your installation will most likely fail.

Developer mode is currently recommended due to the evolving nature of the source code. Prior to installing in this mode, please ensure that the pyClarion folder is located at a convenient long-term location. Installing in developer mode means that changes made to the pyClarion folder will be reflected in the pyClarion package (moving the folder will break the package).
