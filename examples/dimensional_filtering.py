"""A demo for setting up a controlled dimensional feature filter."""


from pyClarion import *
from itertools import groupby


# Need dv-pairs to be predefined.

fdomain = [
    feature("color", "red"),
    feature("color", "green"),
    feature("shape", "square"),
    feature("shape", "circle")
]

filter_interface = ParamSet.Interface(
    tag="filter",
    vals=("standby", "clear", "update", "clear+update"),
    clients=fdomain,
    func=lambda c: ("filter", "feature", c.dim),
    param_val="param"
)

chunk_db = Chunks()


alice = Structure(
    name=agent("alice"),
    emitter=AgentCycle(),
    assets=Assets(
        chunk_db=chunk_db,
        filter_interface=filter_interface
    )
)

with alice:
    
    stimulus = Construct(
        name=buffer("stimulus"), 
        emitter=Stimulus()
    )

    controller = Construct(
        name=buffer("controller"), 
        emitter=Stimulus()
    )

    relay = Construct(
        name=buffer("filter"),
        emitter=ParamSet(
            controller=(subsystem("acs"), terminus("nacs")),
            interface=alice.assets.filter_interface
        )
    )

    defaults = Construct(
        name=buffer("defaults"),
        emitter=Constants(
            strengths={f: 0.5 for f in alice.assets.filter_interface.defaults}
        )
    )

    acs = Structure(
        name=subsystem("acs"),
        emitter=ACSCycle(
            sources={
                buffer("controller"), 
                buffer("defaults")
            }
        )
    )

    with acs:

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    buffer("controller"), 
                    buffer("defaults")
                }
            )
        )

        Construct(
            name=terminus("nacs"),
            emitter=ActionSelector(
                source=features("main"),
                client_interface=alice.assets.filter_interface,
                temperature=0.01
            )
        )

    nacs = Structure(
        name=subsystem("nacs"),
        emitter=NACSCycle(
            sources={
                buffer("stimulus"), 
                buffer("filter")
            }
        )
    )

    with nacs:

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    buffer("stimulus"),
                    flow_in("filter_meta")
                }
            )
        )

        Construct(
            name=flow_in("filter_meta"),
            emitter=ParamSet.MetaKnowledge(
                source=buffer("filter"),
                client_interface=alice.assets.filter_interface
            )
        )

        Construct(
            name=terminus("bl-retrieval"),
            emitter=Filtered(
                base=ThresholdSelector(
                    source=features("main"),
                    threshold=.85
                ),
                sieve=buffer("filter")
            )
        )


##################
### Simulation ###
##################


stimulus_1 = {feature("shape", "square"): 1.0, feature("color", "red"): 1.0}

alice.start()

print("CYCLE 1: All open.") 

print("Step 1: Set filter values.")
controller.emitter.input({})
alice.step()
pprint(alice.output)
print() # Empty line

print("Step 2: Propagate stimulus.")
stimulus.emitter.input(stimulus_1)
alice.step()
pprint(alice.output)
print() # Empty line


print("CYCLE 2: Block shape only.")

print("Step 1: Set filter values.")
controller.emitter.input({
    feature("filter", "update"): 1.0,
    feature(("filter", "feature", ("shape", 0)), "param"): 1.0
})
alice.step()
pprint(alice.output)
print() # Empty line

print("Step 2: Propagate stimulus.")
stimulus.emitter.input(stimulus_1)
alice.step()
pprint(alice.output)
print() # Empty line
