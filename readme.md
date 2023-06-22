`pyClarion` is a python package for implementing agents in the Clarion cognitive architecture.

It is highly experimental and aims to be easy to learn, read, use and extend.

The primary resource for the implementation is Ron Sun's [*Anatomy of the Mind*](https://global.oup.com/academic/product/anatomy-of-the-mind-9780199794553). For a short overview of the architecture, see the chapter on Clarion in the [*Oxford Handbook of Cognitive Science*](https://www.oxfordhandbooks.com/view/10.1093/oxfordhb/9780199842193.001.0001/oxfordhb-9780199842193). 

# Key Features

- Convenient and modular agent assembly
- A simple language for initializing explicit knowledge  
- Numerical dictionaries with autodiff support

See the tutorial for a demonstration of most of these features.

# Installation

In a terminal, navigate to the pyClarion folder then:

- To install in developer mode (recommended), run
```pip install -e .```
- To install as a regular library, run
```pip install .```

WARNING: Be sure to include the '`.`' in the install commands. Otherwise, your installation will most likely fail.

Developer mode is currently recommended due to the evolving nature of the source code. Prior to installing in this mode, please ensure that the pyClarion folder is located at a convenient long-term location. Installing in developer mode means that changes made to the pyClarion folder will be reflected in the pyClarion package (moving the folder will break the package).
