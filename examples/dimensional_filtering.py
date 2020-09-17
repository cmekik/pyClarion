"""Demonstrates setting up a dimensional feature filter."""


from pyClarion import *
import pprint
from itertools import groupby


# Need dv-pairs to be predefined.

features = [
    ("color", "red"),
    ("color", "green"),
    ("shape", "square"),
    ("shape", "circle")
]

alice = Structure(
    name=agent("Alice"),
    emitter=AgentCycle(),
    assets=Assets(chunks=Chunks())
)

with alice:
    
    stimulus = Construct(
        name=buffer("stimulus"), 
        emitter=Stimulus()
    )

    # setting up the filtering relay.

    grouped = group_by_dims(feature(dim, val) for dim, val in features)
    _dims, _clients = zip(*list(grouped.items()))
    dlbs = tuple(("dof", "nacs", dim) for dim in _dims) 
    clients = tuple(tuple(entry) for entry in _clients)

    relay = Construct(
        name=buffer("dimensional-filter"),
        emitter=FilteringRelay(
            controller=(subsystem("ACS"), terminus("NACS")),
            interface=FilteringRelay.Interface(
                clients=clients,
                dlbs=dlbs,
                vals=(0, 1)
            )
        )
    )

    defaults = Construct(
        name=buffer("defaults"),
        emitter=ConstantBuffer(
            strengths={f: 0.5 for f in relay.emitter.interface.defaults}
        )
    )

    acs = Structure(
        name=subsystem("ACS"),
        emitter=ACSCycle(
            matches=MatchSet(
                constructs={buffer("stimulus"), buffer("defaults")}
            )
        )
    )

    with acs:

        Construct(
            name=terminus("NACS"),
            emitter=ActionSelector(
                dims=relay.emitter.interface.dims,
                temperature=0.01
            )
        )

        for f in relay.emitter.interface.features:
            Construct(
                name=f, 
                emitter=MaxNode(
                    matches=MatchSet(ctype=ConstructType.buffer)
                )
            ) 

    nacs = Structure(
        name=subsystem("NACS"),
        emitter=NACSCycle(        
            matches=MatchSet(
                constructs={
                    buffer("stimulus"), 
                    buffer("defaults"), 
                    buffer("dimensional-filter")
                }
            )
        )
    )

    with nacs:

        Construct(
            name=terminus("bl-retrieval"),
            emitter=FilteredT(
                base=ThresholdSelector(threshold=.85),
                filter=buffer("dimensional-filter")
            )
        )

        for dim, val in features:
            Construct(
                name=feature(dim, val), 
                emitter=MaxNode(
                    matches=MatchSet(
                        ctype=ConstructType.flow_xb,
                        constructs={buffer("stimulus")}
                    )
                )
            ) 


##################
### Simulation ###
##################


stimulus_1 = {feature("shape", "square"): 1.0, feature("color", "red"): 1.0}

print("CYCLE 1: All open.") 

alice.propagate()
alice.update()
pprint.pprint(alice.output)

stimulus.emitter.input(stimulus_1)
alice.propagate()
alice.update()
pprint.pprint(alice.output)


print("CYCLE 2: Block shape only.")

stimulus.emitter.input({feature(("dof", "nacs", ("shape", 0)), 1): 1.})
alice.propagate()
alice.update()
pprint.pprint(alice.output)

stimulus.emitter.input(stimulus_1)
alice.propagate()
alice.update()
pprint.pprint(alice.output)
