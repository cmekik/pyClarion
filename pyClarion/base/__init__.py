"""
Basic computational constructs for building Clarion agents.

This module is an umbrella for four submodules:
    ``pyClarion.base.symbols``, 
    ``pyClarion.base.packets``, 
    ``pyClarion.base.processors``, and 
    ``pyClarion.base.realizers`` 

For a deeper understanding of the basic infrastructure of pyClarion, submodules 
may be read in the order they are presented above. An overview of the 
architecture of pyClarion is presented below along with some discussion of the 
underlying motivations.

In pyClarion, simulated constructs are named and represented with symbolic 
tokens called construct symbols. Each construct symbol may be associated with 
one or more construct realizers, which define and implement the behavior of the 
named constructs in a specific context.

The symbol-realizer distinction was developed as a solution to the following 
issues:
    1. Information about constructs frequently needs to be communicated 
       among various components. 
    2. Constructs often exhibit complex behavior driven by intricate 
       datastructures (e.g., artificial neural networks).
    3. The same constructs may exhibit different behaviors within different 
       subsystems of a given agent (e.g., the behavior of a same chunk may 
       differ between the action-centered subsystem and the non-action-centered 
       subsystem).
    4. Construct behavior is not exactly prescribed by source material; several 
       implementational possibilities are available for each class of 
       constructs.
Construct symbols allow consistent and efficient communication of construct 
information using basic datastructures such as dicts, lists and sets of 
construct symbols. Construct realizers encapsulate complex behaviors associated 
with client constructs and provide a clean interface for multiple distinct 
realizations of the same construct.
"""


from pyClarion.base.symbols import (
    Node, 
    Microfeature, 
    Chunk, 
    FlowType,
    Flow, 
    Appraisal, 
    Behavior, 
    Buffer, 
    Subsystem, 
    Agent
)
from pyClarion.base.packets import (
    At,
    DefaultActivation,
    ActivationPacket,
    DecisionPacket
)
from pyClarion.base.processors import (
    Channel,
    Junction,
    Selector,
    Effector,
    Source
)
from pyClarion.base.realizers import (
    NodeRealizer,
    FlowRealizer,
    AppraisalRealizer,
    BehaviorRealizer,
    BufferRealizer,
    SubsystemRealizer,
    AgentRealizer,
    UpdateManager
)
