"""
Framework for simulating Clarion constructs.

Views Clarion constructs as networks of networks that propagate activations 
among their nodes. Each node is named by a single symbolic token and 
connections among nodes are decided based on formal properties of construct 
symbols.

Activation propagation follows a pull-based architecture. 
"""


from .symbols import *
from .components import *
from .realizers import *
