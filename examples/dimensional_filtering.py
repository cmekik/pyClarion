"""Demonstrates setting up a dimensional feature filter."""


from pyClarion import *
import pprint
from itertools import groupby


# Need dv-pairs to be predefined.

fdomain = [
    ("color", "red"),
    ("color", "green"),
    ("shape", "square"),
    ("shape", "circle")
]

chunk_db = Chunks()

grouped = group_by_dims(feature(dim, val) for dim, val in fdomain)
mapping = {("dof", "nacs", dim): set(entry) for dim, entry in grouped.items()}

filter_interface = FilteringRelay.Interface(
    mapping=mapping, # type: ignore
    vals=(0, 1)
)

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
        name=buffer("dimensional-filter"),
        emitter=FilteringRelay(
            controller=(subsystem("acs"), terminus("nacs")),
            interface=alice.assets.filter_interface
        )
    )

    defaults = Construct(
        name=buffer("defaults"),
        emitter=ConstantBuffer(
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
                buffer("dimensional-filter")
            }
        )
    )

    with nacs:

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={
                    buffer("stimulus")
                }
            )
        )

        Construct(
            name=terminus("bl-retrieval"),
            emitter=FilteredT(
                base=ThresholdSelector(
                    source=features("main"),
                    threshold=.85
                ),
                filter=buffer("dimensional-filter")
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
pprint.pprint(alice.output)
print() # Empty line

print("Step 2: Propagate stimulus.")
stimulus.emitter.input(stimulus_1)
alice.step()
pprint.pprint(alice.output)
print() # Empty line


print("CYCLE 2: Block shape only.")

print("Step 1: Set filter values.")
controller.emitter.input({feature(("dof", "nacs", ("shape", 0)), 1): 1.})
alice.step()
pprint.pprint(alice.output)
print() # Empty line

print("Step 2: Propagate stimulus.")
stimulus.emitter.input(stimulus_1)
alice.step()
pprint.pprint(alice.output)
print() # Empty line
