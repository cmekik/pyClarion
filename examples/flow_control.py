"""Demonstrates selection & control of reasoning methods."""


from pyClarion import (
    feature, chunk, rule, features, chunks, flow_in, flow_tt, flow_bt, flow_tb, 
    terminus, buffer, subsystem, agent,
    Structure, Construct,
    AgentCycle, NACSCycle, ACSCycle,
    Stimulus, Constants, ParamSet, MaxNodes, Repeater, TopDown, BottomUp, 
    AssociativeRules, ActionSelector, BoltzmannSelector, Gated, Filtered,
    Chunks, Rules, Assets,
    nd, pprint
)

from itertools import count


#############
### SETUP ###
#############

# This example will be addressing the basics of control in pyClarion. By way of 
# example we present a simple recipe for controlling the mode of reasoning. We 
# will extend the model presented in `free_association.py` to support control 
# of the mode of reasoning.

# In the proposed recipe, control of the mode of reasoning is achieved by 
# gating the output of various flows present in the original model. Gating is 
# implemented by an emitter that wraps the base emitter of the flow and 
# multiplicatively gates its output according to gating parameters received 
# from a parameter buffer. The parameter buffer is modeled using a ParamSet 
# emitter, which roughly captures the concept of a psychological set (e.g., 
# task set, goal set, perceptual set, etc.). It can be configured by a 
# controller to store and emit parameters to input filters, output gates, and 
# other parametric processes.

### Knowledge Setup ###

# The semantic feature domain used in this simulation is identical to that used 
# in `free_association.py`, so we do not specify again here. 

# To control emitter gating, we need to specify a feature interface. In 
# general, pyClarion emitters that are subject to control specify constructors 
# for their control interfaces. These constructors offer a standardization of 
# control interfaces and automate the process of creating control features 
# based on provided arguments. In general, feature interfaces define two sets 
# of features: commands and parameters. It is assumed that the controlled 
# emitter expects to receive exactly one activated value for each command 
# feature dimension at each time step. For parameters, it is assumed that each 
# parameter dimension contains exactly one value, and what is desired is an 
# activation strength for that value.

# Below, we specify the control interface for the ParamSet emitter that will 
# serve to control gating in our simulation. The tag argument will be passed in 
# as the tag for the command features, whereas vals defines the values of these 
# features. The value semantics are defined by the order of the elements passed 
# to vals and is reflected by the chosen value strings. The clients argument 
# specifies the constructs for which parameters will be emitted. Basically, the 
# emitter will map activations of its parameter features to the client 
# constructs and output the result. The func argument specifies how to map 
# parameter dimensions to clients, taking in a client construct and outputting 
# the corresponding dimensional tag. Finally, the param_val argument specifies 
# the expected feature value for parameter features.

gate_interface = ParamSet.Interface(
    tag="gate",
    clients={
        flow_in("stimulus"),
        flow_tt("associations"),
        flow_bt("main")
    },
    func=lambda c: ("gate", c.ctype.name, c.cid),
)

# Feature interfaces often also define default values for commands. For this 
# simulation, we would like the default values to be selected unless some other 
# command receives higher activation. To do this, we will setup a constant 
# buffer to activate default values to a constant level. 

# To set up default action activations using numdicts. Numdicts, numerical 
# dictionaries, are a pyClarion native datatype. Fundamentally, they are fancy 
# wrappers around dictionaries that allow for mathematical operations. 
# Operations are carried out elementwise, where elements are matched by keys.
# Mathematical operations are supported between numdicts and constant numerical 
# values (ints, floats, etc.). Furthermore, numdicts typically define default 
# values for missing keys, which are appropriately updated under mathematical 
# operations.
 
# In this particular simulation, we set our default action values to have a 
# constant activation of 0.5

default_strengths = nd.MutableNumDict()
default_strengths.extend(gate_interface.defaults, value=0.5)

# We initialize and populate chunk and rule databases as in 
# `free_association.py`.

chunk_db = Chunks()
rule_db = Rules()

rule_db.link(rule("1"), chunk("FRUIT"), chunk("APPLE")) 

chunk_db.link( 
    chunk("APPLE"), 
    feature("color", "#ff0000"), 
    feature("color", "#008000"),
    feature("tasty", True)
)

chunk_db.link( 
    chunk("JUICE"),
    feature("tasty", True),
    feature("state", "liquid")
)

chunk_db.link( 
    chunk("FRUIT"),
    feature("tasty", True),
    feature("sweet", True)
)


### Agent Assembly ###

# The agent assembly process is very similar to `free_association.py`, but we 
# define some additional constructs and structures.

# Note that we place a handle to the gate interface at the agent level.

alice = Structure(
    name=agent("Alice"),
    emitter=AgentCycle(),
    assets=Assets(
        gate_interface=gate_interface
    )
)

with alice:

    stimulus = Construct(
        name=buffer("stimulus"), 
        emitter=Stimulus()
    )

    # The acs_ctrl construct is an entry point for manually passing in commands 
    # to the action-centered subsystem (ACS). Normally, the ACS would select 
    # actions based on perceptual stimuli, working memory etc. For simplicity, 
    # we directly stimulate the action features in the ACS to drive action 
    # selection.

    acs_ctrl = Construct(
        name=buffer("acs_ctrl"), 
        emitter=Stimulus()
    )

    # The gate is implemented as a buffer entrusted to a ParamSet emitter, as 
    # mentioned earlier. To initialize the ParamSet instance, we specify a 
    # controller and pass in our gate interface.
    
    gate = Construct(
        name=buffer("gate"),
        emitter=ParamSet(
            controller=(subsystem("acs"), terminus("nacs")),
            interface=alice.assets.gate_interface
        )
    )

    # The defaults are handled by a buffer entrusted to a Constants emitter, 
    # which simply outputs the defaults we defined above.

    defaults = Construct(
        name=buffer("defaults"),
        emitter=Constants(strengths=default_strengths)
    )

    # In this simulation, we add an entirely new subsystem to the model: the 
    # action-centered subystem, which handles action selection. This subsystem 
    # takes inputs from the acs_ctrl and default buffers, and it has its own 
    # activation sequence as specified by ACSCycle.

    acs = Structure(
        name=subsystem("acs"),
        emitter=ACSCycle()
    )

    with acs:

        # The ACS feature pool listens to both the control and default buffers 
        # and outputs, for each feature, the maximum activation recommended by 
        # these sources.

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    buffer("acs_ctrl"), 
                    buffer("defaults")
                }
            )
        )

        # We define an action terminus in ACS for controlling flow gating in 
        # NACS. To do this, we make use of the ActionSelector emitter, which 
        # selects, for each command dimension in its client interface, a single 
        # value through boltzmann sampling, and forwards activations of any 
        # parameter features defined in the interface.

        Construct(
            name=terminus("nacs"),
            emitter=ActionSelector(
                source=features("main"),
                client_interface=alice.assets.gate_interface,
                temperature=0.01
            )
        )

    # NACS setup is almost identical to `free_association.py`, only the NACS 
    # listens to the gate buffer in addition to the stimulus buffer.

    nacs = Structure(
        name=subsystem("nacs"),
        emitter=NACSCycle(),
        assets=Assets(
            chunk_db=chunk_db, 
            rule_db=rule_db
        )
    )

    with nacs:

        # Node Pools

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    flow_tb("main")
                }
            )
        )

        # In this simulation we use two chunk pools, `chunks("in")` and 
        # `chunks("out")`. Since we will be presenting sequences of stimuli and 
        # including both top-down and bottom-up flows, using only one chunk 
        # pool, as in `free_association.py`, would result in a feed-back loop 
        # between chunks and features. In standard Clarion, this is generally 
        # not desirable. One simple way to solve this issue is to compute 
        # top-down activations from chunks("in") but write bottom-up 
        # activations to `chunks("out")`.

        Construct(
            name=chunks("in"),
            emitter=MaxNodes(
                sources={
                    flow_in("stimulus") 
                }
            )
        )

        Construct(
            name=chunks("out"),
            emitter=MaxNodes(
                sources={
                    chunks("in"), 
                    flow_bt("main"), 
                    flow_tt("associations")
                }
            )
        )

        # Flows

        # The model includes all flows already present in the 
        # `free_association.py` model: an associative rule flow, a top-down 
        # flow, and a bottom-up flow. We additionally include some flow_in 
        # constructs.

        # The first instance of gating in this example is on the stimulus. We 
        # do not gate the stimulus buffer. Instead, we create a flow_in 
        # construct which repeats the output of the stimulus buffer and gate 
        # that. To implement the gate, we initialize a `Gated` object which 
        # wraps the activation repeater.
        
        Construct(
            name=flow_in("stimulus"),
            emitter=Gated(
                base=Repeater(source=buffer("stimulus")),
                gate=buffer("gate")
            )
        )

        # In addition to the stimulus, we gate the associative rule flow and 
        # the bottom-up flow.

        Construct(
            name=flow_tt("associations"),
            emitter=Gated(
                base=AssociativeRules(
                    source=chunks("in"),
                    rules=nacs.assets.rule_db
                ),
                gate=buffer("gate")
            ) 
        )

        Construct(
            name=flow_bt("main"), 
            emitter=Gated(
                base=BottomUp(
                    source=features("main"),
                    chunks=nacs.assets.chunk_db
                ),
                gate=buffer("gate") 
            )
        )

        Construct(
            name=flow_tb("main"), 
            emitter=TopDown(
                source=chunks("in"),
                chunks=nacs.assets.chunk_db
            ) 
        )

        # Termini

        # Aside from adjusting for the new chunk pool setup, the retrieval 
        # terminus is unchanged from `free_association.py`. 

        Construct(
            name=terminus("retrieval"),
            emitter=Filtered(
                base=BoltzmannSelector(
                    source=chunks("out"),
                    temperature=.1
                ),
                sieve=flow_in("stimulus")
            )
        )

# Agent setup is complete!


##################
### Simulation ###
##################

# To simplify exposition, we define some convenience functions for the 
# simulations.

# For this simulation, we will present various control sequences while keeping 
# the task cue constant. This setup will demonstrate the different behaviours 
# afforded by controlling the mode of reasoning.


def print_step(agent, step):

    print("Step {}:".format(step))
    print()
    print("Activations")
    output = dict(agent.output)
    del output[buffer("defaults")]
    pprint(output)
    print()


def execute_sequence(agent, cue, control_sequence, counter=None):

    if counter is None:
        counter = count(1)

    for commands in control_sequence:
        stimulus.emitter.input(cue)
        acs_ctrl.emitter.input(commands)
        agent.step()
        print_step(alice, next(counter))


# We set the cue and initialize the step counter for reporting purposes.

counter = count(1)
cue = {chunk("APPLE"): 1.}

# Now we run the simulation.

print("Sequence 1: Open stimulus only.\n") 
print("NACS should output nothing on sequence end b/c flows not enabled...\n")

control_sequence = [
    {
        feature("gate", "update"): 1.0,
        feature(("gate", "flow_in", "stimulus")): 1.0
    },
    {}
]

execute_sequence(alice, cue, control_sequence, counter)


print("Sequence 2: Open stimulus & associations only.\n")
print("NACS should output 'FRUIT' on sequence end due to assoc. rules...\n")

control_sequence = [
    {
        feature("gate", "update"): 1.0,
        feature(("gate", "flow_tt", "associations")): 1.0
    },
    {}
]

execute_sequence(alice, cue, control_sequence, counter)


print("Sequence 3: Open stimulus & bottom-up only.\n")
print(
    "NACS should output 'FRUIT' or 'JUICE' with equal probability on sequence " 
    "end due to bottom-up activations...\n"
)

control_sequence = [
    {
        feature("gate", "overwrite"): 1.0,
        feature(("gate", "flow_in", "stimulus")): 1.0,
        feature(("gate", "flow_bt", "main")): 1.0
    },
    {}
]

execute_sequence(alice, cue, control_sequence, counter)


##################
### CONCLUSION ###
##################

# This simple simulation sought to demonstrate the following:
#   - The basics of action control,
#   - Using numdicts to encode activations and other numerical data associated 
#     with pyClarion constructs, and
#   - A recipe for reasoning mode selection. 
