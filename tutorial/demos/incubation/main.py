import pyClarion as cl
import random


def build():

    # perceptual input feature definitions
    pcfg = [
        "color-red", "color-yellow", "color-green",
        "shape-round", "shape-oblong",
        "size-medium", "size-large"
    ]

    # Key Point:
    # In this implementation, NDRAM training is enabled by an external command 
    # which theoretically may be issued by the ACS. In this simulation, it is 
    # directly controlled though the 'ctrl' module.

    with cl.Structure("agent") as agent:
        stim = cl.Module("stim", cl.Receptors(pcfg)) 
        ctrl = cl.Module("ctrl", cl.Repeat(), ["ctrl"])
        cl.Module("null", cl.Repeat(), ["null"])
        with cl.Structure("nacs"):
            cl.Module("bi", cl.CAM(), ["../stim", "ndram"])
            ndram = cl.Module("ndram", cl.NDRAM(), ["../ctrl", "bi"])
            bu = cl.Module("bu", cl.BottomUp(), 
                ["chunks#0", "chunks#1", "chunks#2", "ndram"])
            chunks = cl.Module("chunks", cl.Store(), 
                ["../null", "../null", "../null", "../null"])

    with open("chunks.ccml") as fr_data:
        cl.load(fr_data, agent)

    return agent, (stim, ctrl, ndram, bu, chunks) 


def main():

    # INITIALIZATION

    agent, (stim, ctrl, ndram, bu, chunks) = build()

    print("Perceptual Feature Set:")
    cl.pprint(stim.process.reprs)
    print()

    print("Pre-defined Chunks:")
    cl.pprint(chunks.process.cw)
    print()

    # TRAINING

    training_stimuli = {
        "apple": {
            "color-red"    :  1.0,
            "color-yellow" : -1.0,
            "color-green"  : -1.0,
            "shape-round"  :  1.0,
            "shape-oblong" : -1.0,
            "size-medium"  :  1.0,
            "size-large"   : -1.0
        },
        "banana": {
            "color-red"    : -1.0,
            "color-yellow" :  1.0,
            "color-green"  : -1.0,
            "shape-round"  : -1.0,
            "shape-oblong" :  1.0,
            "size-medium"  :  1.0,
            "size-large"   : -1.0
        },
        "watermelon": {
            "color-red"    : -1.0,
            "color-yellow" : -1.0,
            "color-green"  :  1.0,
            "shape-round"  :  1.0,
            "shape-oblong" : -1.0,
            "size-medium"  : -1.0,
            "size-large"   :  1.0
        },
    }

    ctrl.output = cl.NumDict({ndram.process.cmds[1]: 1.0}) # enable training
    for key in random.choices(list(training_stimuli),k=2000):
        stim.process.stimulate(training_stimuli[key])
        agent.step()
    ctrl.output = cl.NumDict() # disable training
    ndram.output = cl.NumDict() # clear NDRAM output for test

    # TESTING

    N = 50
    apple, banana, watermelon = [], [], []
    stim.process.stimulate(["size-medium"])
    for i in range(N):
        agent.step()
        if i == 0:
            stim.process.stimulate([]) # clear stimulus
        apple.append(bu.output[cl.chunk("nacs/chunks#0002-apple")])
        banana.append(bu.output[cl.chunk("nacs/chunks#0010-banana")])
        watermelon.append(bu.output[cl.chunk("nacs/chunks#0018-watermelon")])

    print("Retrieval Output:")
    cl.pprint(bu.output)
    print()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("To see plots, install matplotlib!")
    else:
        time = list(range(N))
        fig, ax = plt.subplots()
        ax.plot(time, apple, label="apple", color="tab:red", alpha=.7)
        ax.plot(time, banana, label="banana", color="tab:olive", alpha=.7)
        ax.plot(time, watermelon, 
            label="watermelon", color="tab:green", alpha=.7)
        ax.legend()
        ax.set(
            title="NDRAM dynamics on input 'size-medium'", 
            xlabel="Time", 
            ylabel="Activation")
        plt.show()


if __name__ == "__main__":
    main()
