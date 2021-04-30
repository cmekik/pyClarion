import pyClarion as pcl
from pyClarion import nd
from pyClarion import feature, chunk, rule

from contextlib import ExitStack
from math import isclose
from itertools import count

#############
### Setup ###
#############

# This simulation demonstrate how pyClarion agent behave in a given rule database.

# The main task of this program is that given the agent starting out thirsty/hungry,
# the agent will self-navigate itself to the store and buy drink/food,
# under the help of Environment. 

# The whole simulation is automatic with the given fixed rule data base.

class Environment(object):
# An automated class which help iterate through the simulation

    # Define elements in the simulation that are explicit to the environment
    # The internal elements (i.e. goal interface) will be defined in the agent
    domain = pcl.Domain(
        features=(
            feature("location", "office"),
            feature("location", "enroute"),
            feature("location", "store"),
            feature("start")
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

    def __init__(self, travel_time):

        self.state = nd.NumDict({}, default=0.0)
        # Agent current state
        self.len_goal_stack = 0
        # Agent goal stack
        self.goToStore = False
        # Whether "go to store" is in agent's external actions
        self.travel_time = travel_time
        # Steps needed to travel from office to store

    def __iter__(self):
        # yield state of agent for each step of simulation
        # Determine simulation stopping condition, place switching condition

        time = count()
        travel_time = 0

        for t in time:

            # Stop the simulation when the agent has no goals anymore.
            # Never stop for t < 2 because the agent has no goals initially.
            if self.len_goal_stack == 0 and t >= 2:
                break

            # Simulation time limit included just in case.
            if t >= 100:
                break

            else:
                # Delete the "starting flag" after the simulation has started
                if t == 1:
                    del self.state[feature("start")]

                # Under certain state, move agent from office to enroute (on the way)
                if self.goToStore:
                    if feature("location", "office") in self.state:
                        del self.state[feature("location", "office")]
                    self.state[feature("location", "enroute")] = 1.0
                    travel_time += 1

                # Count the travel time 
                elif travel_time > 0 and travel_time < self.travel_time:
                    travel_time += 1

                # Once travel time has been reached, move agent from enroute to store
                elif travel_time == self.travel_time:
                    if feature("location", "enroute") in self.state:
                        del self.state[feature("location", "enroute")]
                    self.state[feature("location", "store")] = 1.0

                yield self.state, time

    def update(self, inputND, inputGS, inputEa):
        # Update info stored in environment, necessary for __iter__
        # Update for each step of simulation

        self.state = inputND
        self.len_goal_stack = len(inputGS)
        self.goToStore = feature("travel", "store") in inputEa


def print_step(simulation_time, state, output, actions, goal_actions, goals, goal_stack):
    # Print state of agent for each step of simulation

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

    print("Goal stack")
    pcl.pprint(goal_stack)
    print()

# Interface definition

# Elements implicit to the Environment declared below
domain = pcl.Domain(
    features=(
        feature("thirsty"),
        feature("hungry"),
    )
)

goal_interface = pcl.GoalStay.Interface(
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

domains = (domain, Environment.domain, goal_interface, Environment.external_interface)
interfaces = domains[2:]

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

    # Fixed rules

    # feature(start) indicate the first step of the simulation
    # "check thirst/hunger" is an intermediate goal which 
    # make sure that certain goal doesn't get initiated consecutively.
    acs_fr.define(
        rule("INITIALIZE"),
        acs_cdb_a.define(
            chunk("INITIALIZE"),
            feature(("gctl", "cmd"), "write"),
            feature(("gctl", "goal"), "check"),
            feature(("gctl", "gobj"), "thirst")
        ),
        acs_cdb_c.define(
            chunk("INITIAL_CONDITION"),
            feature("start")
        )
    )
    
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
            feature("gobj", "hunger"),
            feature(("gctl", "state"), "start")
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
            feature("gobj", "thirst"),
            feature(("gctl", "state"), "start")
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
        rule("IF_NEW_GOAL_TRAVELING_TO_STORE_THEN_START_GOING_TO_STORE"),
        acs_cdb_a.define(
            chunk("START_GOING_TO_STORE"),
            feature("travel", "store"),
            feature(("gctl", "cmd"), "engage")
        ),
        acs_cdb_c.define(
            chunk("NEW_GOAL_TRAVELING_TO_STORE"),
            feature("goal", "travel"),
            feature("gobj", "store"),
            feature(("gctl", "state"), "start")
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

    acs_fr.define(
        rule("IF_BUY_GOAL_PASSED_THEN_PASS_CHECK_GOAL"),
        acs_cdb_a.define(
            chunk("PASS_CHECK_GOAL"),
            feature(("gctl", "cmd"), "pass")
        ),
        acs_cdb_c.define(
            chunk("BUY_GOAL_PASSED"),
            feature("goal", "check"),
            feature(("goal", "prev"), "buy"),
            feature(("gctl", "state"), "resume")
        )
    )


agent = pcl.Structure(
    name=pcl.agent("agent")
)

# Agent definition

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

        fr = pcl.Construct(
            name=pcl.flow_tt("fixed_rules"),
            process=pcl.ActionRules(
                source=pcl.chunks("condition"),
                rules=acs_fr,
                threshold=.9,
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
                interface=Environment.external_interface,
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

        gs = pcl.Construct(
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

# Actual simulation starts

# Initial state 
state = nd.MutableNumDict({
        feature("thirsty"): 1.0,
        feature("location", "office"): 1.0,
        feature("start"): 1.0
        #feature("hungry"): 1.0,
    },default=0.0)

actions = nd.NumDict({}, default=0.0)

# Set traveling steps to, for example, 2
env = Environment(2)
# Fill env with initial state
env.update(state, gs.output, exta.output)

for curState, simulation_time in env:

    # Push agent to next state
    stim.process.input(curState)
    agent.step()
    goals = gb.output
    actions = exta.output
    goal_actions = gt.output
    goal_stack = gs.output

    # Print current state
    print_step(simulation_time, curState, fs.output, actions, goal_actions, goals, goal_stack)
    print("-" * 80)

    # Update env
    env.update(curState, gs.output, exta.output)

# Print goal database
pcl.pprint(goal_cdb)