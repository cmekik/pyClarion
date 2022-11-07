import pyClarion as cl
import random


def build():

    # perceptual input feature definitions
    pcfg = [
        "color-red", "color-yellow", "color-green",
        "shape-round", "shape-oblong",
        "size-medium", "size-large"
    ]

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

    agent, (stim, ctrl, ndram, bu, chunks) = build()

    print("Perceptual Feature Set:")
    cl.pprint(stim.process.reprs)
    print()

    print("Pre-defined Chunks:")
    cl.pprint(chunks.process.cw)
    print()

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

    ctrl.output = cl.NumDict({ndram.process.cmds[1]: 1.0})
    for key in random.choices(list(training_stimuli),k=2000):
        stim.process.stimulate(training_stimuli[key])
        agent.step()
    ctrl.output = cl.NumDict()
    ndram.output = cl.NumDict()

    N = 50
    apple, banana, watermelon = [], [], []
    stim.process.stimulate(["size-medium"])
    for _ in range(N):
        agent.step()
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
        ax.plot(time, apple, label="apple", color="tab:red")
        ax.plot(time, banana, label="banana", color="tab:olive")
        ax.plot(time, watermelon, label="watermelon", color="tab:green")
        ax.legend()
        ax.set(
            title="NDRAM dynamics on input 'size-medium'", 
            xlabel="Time", 
            ylabel="Activation")
        plt.show()


if __name__ == "__main__":
    main()
