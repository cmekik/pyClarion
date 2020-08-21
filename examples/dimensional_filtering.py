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

stimulus = Construct(
    name=buffer("stimulus"), 
    emitter=Stimulus()
)


# setting up the filtering relay.

grouped = group_by_dims(feature(dim, val) for dim, val in features)
_dims, _clients = zip(*list(grouped.items()))
dims = tuple("nacs-df-{}".format(dim) for dim in _dims) 
clients = tuple(tuple(entry) for entry in _clients)

relay = Construct(
    name=buffer("dimensional-filter"),
    emitter=FilteringRelay(
        controller=(subsystem("ACS"), terminus("NACS")),
        interface=FilteringRelay.Interface(
            clients=clients,
            dims=dims,
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
alice.add(stimulus, relay, defaults)

acs = Structure(
    name=subsystem("ACS"),
    emitter=ACSCycle(
        matches=MatchSet(
            constructs={buffer("stimulus"), buffer("defaults")}
        )
    )
)
alice.add(acs)

acs.add(
    Construct(
        name=terminus("NACS"),
        emitter=ActionSelector(
            dims=relay.emitter.interface.dims,
            temperature=0.01
        )
    )
)

fnodes = [
    Construct(
        name=f, 
        emitter=MaxNode(
            matches=MatchSet(ctype=ConstructType.buffer)
        )
    ) 
    for f in relay.emitter.interface.features
]
acs.add(*fnodes)

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
alice.add(nacs)

nacs.add(
    Construct(
        name=terminus("bl-retrieval"),
        emitter=FilteredT(
            base=ThresholdSelector(threshold=.85),
            filter=buffer("dimensional-filter")
        )
    )
)

fnodes = [
    Construct(
        name=feature(dim, val), 
        emitter=MaxNode(
            matches=MatchSet(
                ctype=ConstructType.flow_xb,
                constructs={buffer("stimulus")}
            )
        )
    ) for dim, val in features
]
nacs.add(*fnodes)


##################
### Simulation ###
##################

stimulus_1 = {
    feature("shape", "square"): 1.0,
    feature("color", "red"): 1.0
}

print("CYCLE 1: All open.") 

alice.propagate()
alice.update()
pprint.pprint(alice.output)

alice.propagate(kwds={buffer("stimulus"): {"stimulus": stimulus_1}})
alice.update()
pprint.pprint(alice.output)


print("CYCLE 2: Block shape only.")

alice.propagate(
    kwds={buffer("stimulus"): {"stimulus": {feature("nacs-df-shape", 1): 1.}}}
)
alice.update()
pprint.pprint(alice.output)

alice.propagate(kwds={buffer("stimulus"): {"stimulus": stimulus_1}})
alice.update()
pprint.pprint(alice.output)
