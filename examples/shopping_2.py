import pyClarion as pcl
from pyClarion import nd
from pyClarion import feature, chunk, rule

from contextlib import ExitStack
from math import isclose

#overall comments needed

def print_step(simulation_time, state, output, actions, goal_actions, goals):
    print("Step {}".format(simulation_time))
    print()

    print("State")
    pcl.pprint(state)
    print()

    print("ACS Features")
    pcl.pprint(output)
    print()

    print("External Actions")
    pcl.pprint(actions)
    print()

    print("Goal Actions")
    pcl.pprint(goal_actions)
    print()

    print("Goals")
    pcl.pprint(goals)
    print()

domain = pcl.Domain(
    features=(
        feature("location", "office"),
        feature("location", "enroute"),
        feature("location", "store"),
        feature("thirsty"),
        feature("hungry")
    )
)

goal_interface = pcl.GoalStay.Interface(# needs added
    name="gctl",
    goals=(
        feature("goal", "travel"),
        feature("goal", "buy"),
        feature("goal", "check"),
        feature("gobj", "office"),
        feature("gobj", "store"),
        feature("gobj", "food"),
        feature("gobj", "drink"),
        feature("gobj", "hunger"),
        feature("gobj", "thirst")
    )
)

external_interface = pcl.Interface(
    cmds=(
        feature("travel", "sby"),
        feature("travel", "office"),
        feature("travel", "store"),
        feature("buy", "sby"),
        feature("buy", "food"),
        feature("buy", "drink")
    ),

)

domains = (domain, goal_interface, external_interface)
interfaces = domains[1:]

# Make sure the domains are disjoint to avoid unexpected behavior
assert pcl.Domain.disjoint(*domains)

default_strengths = nd.MutableNumDict(default=0)
default_strengths.extend(
    *(itf.defaults for itf in interfaces),
    value=0.5
)

acs_cdb_c = pcl.Chunks()
acs_cdb_a = pcl.Chunks()
acs_fr = pcl.Rules(max_conds=1)

goal_cdb = pcl.Chunks()
goal_blas = pcl.BLAs(density=0.0)

with ExitStack() as exit_stack:

    for ctxmgr in (
        acs_cdb_a.enforce_support(*domains), 
        acs_cdb_c.enforce_support(*domains),
        acs_fr.enforce_support(acs_cdb_a, acs_cdb_c), 
    ):
        exit_stack.enter_context(ctxmgr)    


    
    acs_fr.define(
        rule("IF_HUNGRY_THEN_GOAL_BUY_FOOD"),
        acs_cdb_a.define(
            chunk("GOAL_BUY_FOOD"),
            feature(("gctl", "cmd"), "write"),
            feature(("gctl", "goal"), "buy"),
            feature(("gctl", "gobj"), "food")
        ),
        acs_cdb_c.define(
            chunk("HUNGRY"),
            feature("hungry"),
            feature("goal", "check"),
            feature("gobj", "hunger")
        )
    )

    acs_fr.define(
        rule("IF_THIRSTY_THEN_GOAL_BUY_DRINK"),
        acs_cdb_a.define(
            chunk("GOAL_BUY_DRINK"),
            feature(("gctl", "cmd"), "write"),
            feature(("gctl", "goal"), "buy"),
            feature(("gctl", "gobj"), "drink")
        ),
        acs_cdb_c.define(
            chunk("THIRSTY"),
            feature("thirsty"),
            feature("goal", "check"),
            feature("gobj", "thirst")
        )
    )

    acs_fr.define(
        rule("IF_WANT_TO_BUY_AND_AT_OFFICE_THEN_GOAL_GO_TO_STORE"),
        acs_cdb_a.define(
            chunk("GOAL_GO_TO_STORE"),
            feature(("gctl", "cmd"), "write"),
            feature(("gctl", "goal"), "travel"),
            feature(("gctl", "gobj"), "store")
        ),
        acs_cdb_c.define(
            chunk("WANT_TO_BUY_AND_AT_OFFICE"),
            feature("goal", "buy"),
            feature(("gctl", "state"), "start"),
            feature("location", "office")
        )
    )

    acs_fr.define(
        rule("IF_TRAVELING_TO_STORE_AND_AT_STORE_THEN_REGISTER_SUCCESS"),
        acs_cdb_a.define(
            chunk("REGISTER_SUCCESS"),
            feature(("gctl", "cmd"), "pass"),
        ),
        acs_cdb_c.define(
            chunk("TRAVELING_TO_STORE_AND_AT_STORE"),
            feature("goal", "travel"),
            feature("gobj", "store"),
            feature("location", "store")
        )
    )

    acs_fr.define(
        rule("IF_WANT_TO_BUY_FOOD_AND_IN_STORE_THEN_BUY_FOOD"),
        acs_cdb_a.define(
            chunk("BUY_FOOD"),
            feature(("gctl", "cmd"), "pass"),
            feature("buy", "food")
        ),
        acs_cdb_c.define(
            chunk("WANT_TO_BUY_FOOD_AND_IN_STORE"),
            feature("goal", "buy"),
            feature("gobj", "food"),
            feature("location", "store")
        )
    )

    acs_fr.define(
        rule("IF_WANT_TO_BUY_DRINK_AND_IN_STORE_THEN_BUY_DRINK"),
        acs_cdb_a.define(
            chunk("BUY_DRINK"),
            feature(("gctl", "cmd"), "pass"),
            feature("buy", "drink")
        ),
        acs_cdb_c.define(
            chunk("WANT_TO_BUY_DRINK_AND_IN_STORE"),
            feature("goal", "buy"),
            feature("gobj", "drink"),
            feature("location", "store")
        )
    )

agent = pcl.Structure(
    name=pcl.agent("agent")
)

with agent:

    stim = pcl.Construct(
        name=pcl.buffer("stimulus"),
        process=pcl.Stimulus()
    )

    gb = pcl.Construct(
        name=pcl.buffer("goals"),
        process=pcl.GoalStay(
            controller=(pcl.subsystem("acs"), pcl.terminus("goals")),
            source=(pcl.subsystem("ms"), pcl.terminus("goals")),
            interface=goal_interface,
            chunks=goal_cdb,
            blas=goal_blas
        )
    )

    pcl.Construct(
        name=pcl.buffer("defaults"),
        process=pcl.Constants(
            strengths=default_strengths
        )
    )

    acs = pcl.Structure(
        name=pcl.subsystem("acs"),
        assets=pcl.Assets(
            cdb_c=acs_cdb_c,
            cdb_a=acs_cdb_a
        )
    )

    with acs:

        acs_gb = pcl.Construct(
            name=pcl.flow_in("goals"),
            process=pcl.Pruned(
                base=pcl.TopDown(
                    source=pcl.buffer("goals"),
                    chunks=goal_cdb
                ),
                accept=pcl.ConstructType.chunk
            )
        )

        fs = pcl.Construct(
            name=pcl.features("in"),
            process=pcl.MaxNodes(
                sources=(
                    pcl.buffer("stimulus"), 
                    pcl.buffer("goals"),
                    pcl.flow_in("goals") 
                )
            )
        )

        pcl.Construct(
            name=pcl.flow_bt("main"),
            process=pcl.BottomUp(
                source=pcl.features("in"),
                chunks=acs_cdb_c
            )
        )

        pcl.Construct(
            name=pcl.chunks("condition"),
            process=pcl.MaxNodes(
                sources=(pcl.flow_bt("main"),)
            )
        )

        pcl.Construct(
            name=pcl.flow_tt("fixed_rules"),
            process=pcl.ActionRules(
                source=pcl.chunks("condition"),
                rules=acs_fr,
                #threshold=.99,
                temperature=0.0001
            )
        )

        pcl.Construct(
            name=pcl.chunks("action"),
            process=pcl.MaxNodes(
                sources=(pcl.flow_tt("fixed_rules"),)
            )
        )

        pcl.Construct(
            name=pcl.flow_tb("main"),
            process=pcl.TopDown(
                source=pcl.chunks("action"),
                chunks=acs_cdb_a
            )
        )

        pcl.Construct(
            name=pcl.features("out"),
            process=pcl.MaxNodes(
                sources=(pcl.flow_tb("main"), pcl.buffer("defaults"))
            )
        )

        exta = pcl.Construct(
            name=pcl.terminus("external_actions"),
            process=pcl.ActionSelector(
                source=pcl.features("out"),
                interface=external_interface,
                temperature=0.0001
            )
        )

        gt = pcl.Construct(
            name=pcl.terminus("goals"),
            process=pcl.ActionSelector(
                source=pcl.features("out"),
                interface=goal_interface,
                temperature=0.0001
            )
        )

    ms = pcl.Structure(
        name=pcl.subsystem("ms"),
        assets=pcl.Assets(
            cdb=goal_cdb,
            blas=goal_blas
        )
    )

    with ms:

        pcl.Construct(
            name=pcl.flow_in("goals"),
            process=pcl.Pruned(
                base=pcl.Repeater(source=pcl.buffer("goals")),
                accept=pcl.ConstructType.chunk
            )
        )

        pcl.Construct(
            name=pcl.flow_in("goal_strengths"),
            process=pcl.BLAStrengths(blas=goal_blas)
        )

        pcl.Construct(
            name=pcl.chunks("goals"),
            process=pcl.MaxNodes(sources=(pcl.flow_in("goal_strengths"),))
        )

        # Note we have to filter the current goal out, otherwise the goal stack 
        # will not work properly.
        pcl.Construct(
            name=pcl.terminus("goals"),
            process=pcl.Filtered(
                base=pcl.BoltzmannSelector(
                    source=pcl.chunks("goals"), 
                    temperature=0.0001,
                    threshold=0.0
                ),
                controller=pcl.flow_in("goals")
            )
        )

################### change initial
state = nd.MutableNumDict({
        feature("thirsty"): 1.0,
        feature("location", "office"): 1.0,
        feature("goal", "check"): 1.0,
        feature("gobj", "thirst"): 1.0
        #feature("hungry"): 1.0,
    },default=0.0)

final_actions = nd.NumDict({
    feature("buy", "food"): 1.0, 
    feature("buy", "drink"): 1.0
    }, default=0.0)

goal_travel_to_store = nd.NumDict({
    feature("goal", "travel"): 1.0, 
    feature("gobj", "store"): 1.0
}, default=0.0)

actions = nd.NumDict({}, default=0.0)


# Simulation time limit included just in case.
simulation_time = 0
while (
    not isclose(nd.val_sum(actions * final_actions), 1.0) #?????
    and simulation_time < 100
):

    simulation_time += 1

    stim.process.input(state)
    agent.step()
    goals = gb.output
    actions = exta.output
    goal_actions = gt.output

    #print function
    print_step(simulation_time, state, fs.output, actions, goal_actions, goals)

    # This is a hack, need more sophisticated control structures to prevent new 
    # goals from needlessly being spawned.
    if feature("goal", "check") in state:#??????
        del state[feature("goal", "check")]

        if feature("gobj", "thirst") in state:
            del state[feature("gobj", "thirst")]
        elif feature("gobj", "hunger") in state:
            del state[feature("gobj", "hunger")]

    if nd.val_sum(fs.output * goal_travel_to_store) >= 1.0:#???????
        if feature("location", "office") in state:
            del state[feature("location", "office")]
        state[feature("location", "store")] = 1.0

pcl.pprint(goal_cdb)