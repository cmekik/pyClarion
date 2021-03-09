"""Demonstrates selection & control of reasoning methods."""


from pyClarion import (
    feature, chunk, rule, features, chunks, flow_in, flow_tt, flow_bt, flow_tb, 
    terminus, buffer, subsystem, agent,
    Structure, Construct,
    Stimulus, Constants, ParamSet, MaxNodes, Repeater, TopDown, BottomUp, 
    AssociativeRules, ActionSelector, BoltzmannSelector, Gated, Filtered,
    Chunks, Rules, Assets,
    nd, pprint
)

from itertools import count


#############
### SETUP ###
#############

# This example will be addressing the basics of control in pyClarion. We will 
# extend the model presented in `free_association.py` to support control 
# of the mode of reasoning.

# In the proposed recipe, control of the mode of reasoning is achieved by 
# gating the output of various flows present in the original model. Gating is 
# implemented by wrapping the base process of the flow with multiplicative gate 
# which receives gating parameters from a parameter buffer. The parameter 
# buffer is modeled using a ParamSet process. It can be configured by a 
# controller to store and emit parameters to input filters, output gates, and 
# other parametrized processes.

### Knowledge Setup ###

# The semantic feature domain used in this simulation is identical to that used 
# in `free_association.py`, so we do not specify it again here. 

# To control emitter gating, we need to specify a feature interface. In 
# general, pyClarion processes subject to control specify constructors for their 
# control interfaces. These constructors offer a standardization of control 
# interfaces and automate the creation of control features. In general, feature 
# interfaces define two sets of features: commands and parameters. It is assumed
# that the controlled process expects to receive exactly one activated value for
# each command feature dimension at each time step. For parameters, it is 
# assumed that each parameter dimension contains exactly one value, and what is 
# desired is an activation strength for that value.

# Below, we specify the control interface for the ParamSet emitter that will 
# serve to control gating in our simulation. The tag argument defines the tag 
# for the command features. The values of the command features are set to their 
# defaults which are: "clear", "overwrite", "update", "standby". These can be 
# modified if desired.
# The clients argument specifies the constructs for which parameters will be 
# emitted. Basically, the ParamSet process will map activations of its 
# parameter features to the client constructs. The func argument specifies how 
# to map parameter dimensions to clients, taking in a client construct and 
# outputting the corresponding dimensional tag.

gate_interface = ParamSet.Interface(
    name="gate",
    pmkrs=("stimulus", "associations", "bottom-up")
)

# Feature interfaces also define default values for commands (one for each 
# command dimension). For this simulation, we would like the default values to 
# be selected unless some other command receives higher activation. To do this, 
# we will setup a constant buffer to activate default values to a constant 
# level.

# We set up default action activations using numdicts. Numdicts, or numerical 
# dictionaries, are a pyClarion native datatype. Fundamentally, they are fancy 
# wrappers around dictionaries that allow for mathematical operations. 
# Operations are carried out elementwise, where elements are matched by keys.
# Mathematical operations are supported between numdicts and constant numerical 
# values (ints and floats). Furthermore, numdicts may define default values to 
# handle missing keys. These defaults are appropriately updated under 
# mathematical operations.
 
# In this particular simulation, we set our default actions to have a constant 
# activation of 0.5.

default_strengths = nd.MutableNumDict(default=0)
default_strengths.extend(gate_interface.defaults, value=0.5)

# Next, we initialize and populate chunk and rule databases as in the original 
# example. 

cdb = Chunks()
rule_db = Rules()

rule_db.define(
    rule("1"), 
    cdb.define( 
        chunk("FRUIT"),
        feature("tasty", True),
        feature("sweet", True)
    ),
    cdb.define( 
        chunk("APPLE"), 
        feature("color", "#ff0000"), 
        feature("color", "#008000"),
        feature("tasty", True)
    )
) 

cdb.define( 
    chunk("JUICE"),
    feature("tasty", True),
    feature("state", "liquid")
)


### Agent Assembly ###

# The agent assembly process is very similar to `free_association.py`, but we 
# define some additional constructs and structures.

alice = Structure(
    name=agent("Alice"),
    assets=Assets(
        gate_interface=gate_interface
    )
)

with alice:

    stimulus = Construct(
        name=buffer("stimulus"), 
        process=Stimulus()
    )

    # The acs_ctrl construct is an entry point for manually passing in commands 
    # to the action-centered subsystem (ACS). Normally, the ACS would select 
    # actions using action-centered knowledge based on perceptual stimuli, 
    # working memory etc. For simplicity, we directly stimulate the action 
    # features in the ACS to drive action selection.

    acs_ctrl = Construct(
        name=buffer("acs_ctrl"), 
        process=Stimulus()
    )

    # The gate is implemented as a buffer entrusted to a ParamSet process, as 
    # mentioned earlier. To initialize the ParamSet instance, we must specify a 
    # controller and pass in a gate interface.
    
    gate = Construct(
        name=buffer("gate"),
        process=ParamSet(
            controller=(subsystem("acs"), terminus("nacs")),
            interface=alice.assets.gate_interface
        )
    )

    # The defaults are handled by a buffer entrusted to a Constants process, 
    # which simply outputs the defaults we defined above.

    defaults = Construct(
        name=buffer("defaults"),
        process=Constants(strengths=default_strengths)
    )

    # This simulation adds an entirely new subsystem to the model: the 
    # action-centered subystem, which handles action selection. We keep this 
    # ACS to a bare minimum.

    acs = Structure(
        name=subsystem("acs")
    )

    with acs:
        
        # Assembly of the ACS is similar to the NACS, but features are the 
        # (primary) entry points for activations.

        Construct(
            name=features("main"),
            process=MaxNodes(
                sources=[
                    buffer("acs_ctrl"), 
                    buffer("defaults")
                ]
            )
        )

        # We define an action terminus in ACS for controlling flow gating in 
        # NACS. To do this, we make use of the ActionSelector emitter, which 
        # selects, for each command dimension in its client interface, a single 
        # value through boltzmann sampling, and forwards activations of any 
        # parameter features defined in the interface.

        Construct(
            name=terminus("nacs"),
            process=ActionSelector(
                source=features("main"),
                interface=alice.assets.gate_interface,
                temperature=0.01
            )
        )

    # Next, we set up the NACS, adding the `Gated` wrapper where necessary.

    nacs = Structure(
        name=subsystem("nacs"),
        assets=Assets(
            cdb=cdb, 
            rule_db=rule_db
        )
    )

    with nacs:

        # The first instance of gating in this example is on the stimulus. We 
        # do not gate the stimulus buffer. Instead, we create a flow_in 
        # construct, which repeats the output of the stimulus buffer, and we 
        # gate that. This allows us to gate the stimulus buffer in the NACS 
        # independently from other subsystems. To implement the gate, we 
        # initialize a `Gated` object which wraps the activation repeater.

        Construct(
            name=flow_in("stimulus"),
            process=Gated(
                base=Repeater(source=buffer("stimulus")),
                controller=buffer("gate"),
                interface=gate_interface,
                pidx=0
            )
        )

        Construct(
            name=chunks("in"),
            process=MaxNodes(
                sources=[flow_in("stimulus")]
            )
        )

        Construct(
            name=flow_tb("main"), 
            process=TopDown(
                source=chunks("in"),
                chunks=nacs.assets.cdb
            ) 
        )

        Construct(
            name=features("main"),
            process=MaxNodes(
                sources=[flow_tb("main")]
            )
        )

        Construct(
            name=flow_tt("associations"),
            process=Gated(
                base=AssociativeRules(
                    source=chunks("in"),
                    rules=nacs.assets.rule_db
                ),
                controller=buffer("gate"),
                interface=gate_interface,
                pidx=1
            ) 
        )

        Construct(
            name=flow_bt("main"), 
            process=Gated(
                base=BottomUp(
                    source=features("main"),
                    chunks=nacs.assets.cdb
                ),
                controller=buffer("gate"),
                interface=gate_interface,
                pidx=2 
            )
        )

        Construct(
            name=chunks("out"),
            process=MaxNodes(
                sources=[
                    chunks("in"), 
                    flow_bt("main"), 
                    flow_tt("associations")
                ]
            )
        )

        Construct(
            name=terminus("retrieval"),
            process=Filtered(
                base=BoltzmannSelector(
                    source=chunks("out"),
                    temperature=.1
                ),
                controller=flow_in("stimulus")
            )
        )

# Agent setup is complete!


##################
### Simulation ###
##################

# For this simulation, we will present various control sequences while keeping 
# the task cue constant. This setup will demonstrate the different behaviours 
# afforded by controlling the mode of reasoning.

# To simplify exposition, we define some convenience functions for the 
# simulations.


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
        stimulus.process.input(cue)
        acs_ctrl.process.input(commands)
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
        feature(("gate", "w"), "upd"): 1.0,
        feature(("gate", "stimulus")): 1.0
    },
    {}
]

execute_sequence(alice, cue, control_sequence, counter)


print("Sequence 2: Open stimulus & associations only.\n")
print("NACS should output 'FRUIT' on sequence end due to assoc. rules...\n")

control_sequence = [
    {
        feature(("gate", "w"), "upd"): 1.0,
        feature(("gate", "associations")): 1.0
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
        feature(("gate", "w"), "clrupd"): 1.0,
        feature(("gate", "stimulus")): 1.0,
        feature(("gate", "bottom-up")): 1.0
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
