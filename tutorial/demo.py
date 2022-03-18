"""
This demo simulates a simple Braitenberg vehicle in pyClarion.

The world is two dimensional. The agent can move left, right, up or down.
The agent also has four virtual light sensors, one for each cardinal direction.

The agent is configured to move away from light. The behavior is programmed as 
fixed rules in frs.ccml. 
"""


import pyClarion as cl
import random


def build(scfg, acfg, path):
    """
    Build a simple pyClarion agent.

    The agent has one stimulus module for inputs, one action module for 
    external actions. Action selection is driven by explicit fixed rules.  
    
    :param scfg: Stimulus config, passed to a `Receptors` constructor.
    :param acfg: Action config, passed to an `Actions` constructor.
    :param path: Path to a ccml file. Contents loaded to a `Store` instance.
    """

    # construct agent 
    with cl.Structure("agent") as agent:
        cl.Module("vis", cl.Receptors(scfg))
        params = cl.Module("params", cl.Repeat(), ["params"])
        cl.Module("null", cl.Repeat(), ["null"])
        with cl.Structure("acs"):
            cl.Module("bi", cl.CAM(), ["../vis"])
            cl.Module("bu", cl.BottomUp(), 
                ["fr_store#0", "fr_store#1", "fr_store#2", "bi"])
            cl.Module("fr", cl.ActionRules(), 
                ["../params", "fr_store#3", "fr_store#4", "bu"])
            cl.Module("td", cl.TopDown(), ["fr_store#0", "fr_store#1", "fr#0"])
            cl.Module("bo", cl.CAM(), ["td"])
            cl.Module("mov", cl.ActionSampler(), ["../params", "bo"], 
                ["../mov#cmds"])
            cl.Module("fr_store", cl.Store(), 
                ["../params", "../null", "../null", "../null"])
        cl.Module("mov", cl.Actions(acfg), ["acs/mov#0"])

    # set temperature parameters for rule & action selection
    params.output = cl.NumDict({
        cl.feature("acs/fr#temp"): 1e-2,  
        cl.feature("acs/mov#temp"): 1e-2,
    })    

    # load fixed rules
    with open(path) as f:
        cl.load(f, agent)

    return agent


def main():
    # Stimulus config
    scfg = ["lum-L", "lum-R", "lum-U", "lum-D"]
    acfg = {"move": ["L", "R", "U", "D"]}

    # Build agent
    agent = build(scfg, acfg, "frs.ccml")
    vis = agent["vis"] # retrieve visual module
    mov = agent["mov"] # retrieve movement module

    # Pretty print all features defined in agent
    print("DEFINED FEATURES", end="\n\n")
    cl.pprint(cl.inspect.fspace(agent))
    print() # leave a blank line

    # Visualize agent structure (if matplotlib is installed)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        import pyClarion.utils.visualize as clv
        fig, ax = plt.subplots()
        ax = clv.adjacency_matrix(ax, agent)
        fig.set_size_inches(6, 6)
        fig.tight_layout()
        plt.show()

    # Run simulation for 20 steps 
    print("SIMULATION", end="\n\n")
    for i in range(20):
        vis.process.stimulate([random.choice(scfg)]) # Stimulate random sensor
        agent.step()

        display = [
            f"Stimulus: {cl.pformat(vis.output)}",
            f"Response: {cl.pformat(mov.output)}"
        ]
        if i: print()
        print(f"Step {i}:")
        for s in display:
            print("    ", s, sep="")


if __name__ == "__main__":
    main()
