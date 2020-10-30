"""
Framework for simulating Clarion constructs.

Views Clarion constructs as networks of networks that propagate activations 
among their nodes. Each node is named by a single symbolic token and 
connections among nodes are decided based on formal properties of construct 
symbols.

Activation propagation follows a pull-based message-passing architecture and 
updates to persistent data (i.e., knowledge) follow a blackboard pattern. 
"""


from pyClarion.base.symbols import *
from pyClarion.base.components import *
from pyClarion.base.realizers import *
