#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def drawCDF(labels, times, fig, title):
    ax = fig.add_subplot(1, 2, 1)
    for i, label in enumerate(labels):
        times_sorted = np.sort(times[i])
        # Normalize
        p = 1.0 * np.arange(len(times[i])) / (len(times[i])-1)
        ax.plot(times_sorted,p, label=label)

    ax.set_ylabel("Probability")
    ax.set_xlabel("Time [ns]")
#    ax.legend(loc='upper left')
    ax.set_title(title)

def drawHist(labels, times, fig, title):
    ax = fig.add_subplot(1, 2, 2)
    for i, label in enumerate(labels):
        ax.hist(times[i], alpha = 0.5, label=label)

    ax.set_ylabel("Count")
    ax.set_xlabel("Time [ns]")
    ax.legend(loc='upper left')
    ax.set_title(title)


def showTimesAll(filenames, titles):
    for filename, title in zip(filenames, titles):
        phases = []
        print(filename +"/"+title)
        with open(filename) as f1:
            data = json.load(f1)

        nofBlocks      = int(data['nofBlocks'])
        nofKernel      = int(data['nofKernel'])
        nofRep         = int(data['nof_repetitions'])
        tileCount      = int(data['tileCount'])
        prefetchtimes  = data['prefetchtimes']
        computetimes   = data['computetimes']
        writebacktimes = data['writebacktimes']

        phases.append([int(i) for i in prefetchtimes])
        phases.append([int(i) for i in computetimes])
        phases.append([int(i) for i in writebacktimes])

        prefetchtimes.sort()
        print(prefetchtimes)

        labels = ['prefetch', 'compute', 'writeback']

        fig = plt.figure()
        fig.suptitle("PREM phases - " + title)
        drawHist(labels, phases, fig, "Histogram")
        drawCDF(labels, phases, fig, "CDF")


if __name__ == "__main__":
    titles1 = [
            "Without warm-up 512 threads, 1 blocks, 1 kernel",
            "Without warm-up 512 threads, 2 blocks, 1 kernel",
            "With warm-up 512 threads, 1 blocks, 1 kernel",
            "With warm-up 512 threads, 2 blocks, 1 kernel",
            "With warm-up 512 threads, 1 blocks, 2 kernel",
            "With warm-up 512 threads, 1 blocks, 4 kernel",
            ]

    filenames1 = [
                "phases/prem-prof-1-block.json",
                "phases/prem-prof-2-block.json",
                "phases/prem-prof-1-block-warmup.json",
                "phases/prem-prof-2-block-warmup.json",
                "phases/prem-prof-1-block-2-kernel-warmup.json",
                "phases/prem-prof-1-block-4-kernel-warmup.json",
                ]

    showTimesAll(filenames1, titles1)
    plt.show()
