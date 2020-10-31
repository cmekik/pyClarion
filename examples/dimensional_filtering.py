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

grouped = group_by_dims(feature(dim, val) for dim, val in fdomain)
mapping = {("dof", "nacs", dim): set(entry) for dim, entry in grouped.items()}

filter_interface = FilteringRelay.Interface(
    mapping=mapping, # type: ignore
    vals=(0, 1)
)

alice = Structure(
    name=agent("alice"),
    emitter=AgentCycle(),
    assets=Assets(chunks=Chunks())
)

with alice:
    
    stimulus = Construct(
        name=buffer("stimulus"), 
        emitter=Stimulus()
    )

    relay = Construct(
        name=buffer("dimensional-filter"),
        emitter=FilteringRelay(
            controller=(subsystem("acs"), terminus("nacs")),
            interface=filter_interface
        )
    )

    defaults = Construct(
        name=buffer("defaults"),
        emitter=ConstantBuffer(
            strengths={f: 0.5 for f in filter_interface.defaults}
        )
    )

    acs = Structure(
        name=subsystem("acs"),
        emitter=ACSCycle(
            sources={buffer("stimulus"), buffer("defaults")}
        )
    )

    with acs:

        Construct(
            name=features("main"),
            emitter=MaxNodes(
                sources={buffer("stimulus"), buffer("defaults")},
                ctype=ConstructType.feature
            )
        )

        Construct(
            name=terminus("nacs"),
            emitter=ActionSelector(
                source=features("main"),
                dims=relay.emitter.interface.dims,
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
                sources={buffer("stimulus")},
                ctype=ConstructType.feature
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
stimulus.emitter.input({feature(("dof", "nacs", ("shape", 0)), 1): 1.})
alice.step()
pprint.pprint(alice.output)
print() # Empty line

print("Step 2: Propagate stimulus.")
stimulus.emitter.input(stimulus_1)
alice.step()
pprint.pprint(alice.output)
print() # Empty line
