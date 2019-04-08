#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def splitStartEndTimes(blockTimes):
    start_times = []
    stop_times = []
    for i in range(len(blockTimes)):
        if (i%2) == 0:
            start_times.append(blockTimes[i])
        else:
            stop_times.append(blockTimes[i])
    return start_times, stop_times

def getTimesOfRepetition(nofKernel, nofBlocks, nofRepetitions, thisRepetition, blockTimes):
    timesRep = []
    for kernel in range(0,nofKernel):
        # Get time subset of this blocks
        startIndex = 2 * nofBlocks*kernel*nofRepetitions
        repIndex = 2 * nofBlocks * thisRepetition
        # Take only the first repetitions of blocktimes
        times = blockTimes[startIndex + repIndex: startIndex + repIndex + 2 *nofBlocks]
        timesRep.extend(times)
    return timesRep

def getMinStartMaxEndTimeOfRepetition(times):
    start_times, end_times = splitStartEndTimes(times)
    minStart = min(start_times)
    maxEnd = max(end_times)
    return minStart, maxEnd

def getKernelTimes(nofKernel, nofBlocks, nofRepetitions, blockTimes):
    times = []
    for rep in range(0, nofRepetitions):
        timesRep = getTimesOfRepetition(nofKernel, nofBlocks, nofRepetitions, rep, blockTimes)
        minStart, maxEnd = getMinStartMaxEndTimeOfRepetition(timesRep)
        duration = maxEnd- minStart
        times.append(duration)

    return times

def drawBarGraph(labels, times, fig, title):
    ax = fig.add_subplot(1, 1, 1)
    y = []
    minerr = []
    maxerr = []
    jitter_per = []
    centers_mean = np.arange(0,len(labels),1)
    centers_jitter = np.arange(0.425,len(labels),1)
    centers_labels = np.arange(0.425/2, len(labels),1)
    width = 0.4
    for i, label in enumerate(labels):
        mean = np.mean(times[i])
        minv = np.min(times[i])
        maxv = np.max(times[i])
        y.append(mean)
        minerr.append(mean-minv)
        maxerr.append(maxv-mean)
        jitter = maxv-minv
        jitter_per.append((jitter/mean)*100)
        print("{:<30s}: mean: {:f}ms, min: {:f}ms, max: {:f}ms, jitter: {:f}ms".format(label, mean, minv, maxv, maxv-minv))

    ax.bar(centers_mean, y, width=width, yerr=[minerr, maxerr], alpha =0.5, hatch='/',ecolor='r', capsize=5)#, yerr=menStd) 
    ax.set_ylabel("Average execution time [ms]")
    ax.set_xticks(centers_labels)
    ax.set_xticklabels(labels,rotation=45, ha='right')
    ax.set_title(title)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.bar(centers_jitter, jitter_per, width=width,color='r', hatch='//', alpha=0.5)
    ax2.set_ylabel("Jitter compared to average execution time [%]")

def drawCDF(labels, times, fig, title):
    ax = fig.add_subplot(1, 2, 1)
    for i, label in enumerate(labels):
        times_sorted = np.sort(times[i])
        # Normalize
        p = 1.0 * np.arange(len(times[i])) / (len(times[i])-1)
        ax.plot(times_sorted,p, label=label)

    ax.set_ylabel("Probability")
    ax.set_xlabel("Time [ms]")
    ax.set_title(title)

def drawHist(labels, times, fig, title):
    ax = fig.add_subplot(1, 2, 2)
    for i, label in enumerate(labels):
        ax.hist(times[i], alpha = 0.5, label=label)

    ax.set_ylabel("Count")
    ax.set_xlabel("Time [ms]")
    ax.legend(loc='upper left')
    ax.set_title(title)

def showTimesAll(filenames, titles):
    times_agg = []
    for filename, title in zip(filenames, titles):
        print(filename +"/"+title)
        with open(filename) as f1:
            data = json.load(f1)

        nofThreads = int(data['nofThreads'])
        nofBlocks  = int(data['nofBlocks'])
        nofKernel  = int(data['nofKernel'])
        nofRep     = int(data['nof_repetitions'])
        dataSize   = int(data['data_size'])
        blockTimes = [float(i)*1e-6 for i in data['blocktimes']]
        kernelTimes = getKernelTimes(nofKernel, nofBlocks, nofRep, blockTimes)
        times_agg.append(kernelTimes)

    fig = plt.figure()
    fig.suptitle("Kernel times")
    drawHist(titles, times_agg, fig, "Histogram")
    drawCDF(titles, times_agg, fig, "CDF")
    fig = plt.figure()
    drawBarGraph(titles, times_agg, fig, "Mean execution times")

if __name__ == "__main__":
    titles1 = [
            "Legacy:1 kernel",
            "Legacy:4 kernel",
            "PREM:4 kernel, no scheduler",
#            "PREM:4 kernel, 0ns offset",
#            "PREM:4 kernel, 500ns offset",
#            "PREM:4 kernel, 1000ns offset",
#            "PREM:4 kernel, 1100ns offset",
#            "PREM:4 kernel, 1200ns offset",
            "PREM:4 kernel, 1300ns offset",
#            "PREM:4 kernel, 1400ns offset",
#            "PREM:4 kernel, 1500ns offset",
#            "PREM:4 kernel, 1600ns offset",
#            "PREM:4 kernel, 1700ns offset",
    ]


    filenames1 = [
            "data/512t-2b-1k-1024-legacy.json",
            "data/512t-2b-4k-1024-legacy.json",
            "data/512t-2b-4k-1024-prem-nosched.json",
#            "data/512t-2b-4k-1024-prem-kernelsched-0pfo.json",
#            "data/512t-2b-4k-1024-prem-kernelsched-500pfo.json",
#            "data/512t-2b-4k-1024-prem-kernelsched-1000pfo.json",
#            "data/512t-2b-4k-1024-prem-kernelsched-1100pfo.json",
#            "data/512t-2b-4k-1024-prem-kernelsched-1200pfo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-1300pfo.json",
#            "tile-sched/512t-2b-4k-1024-prem-kernelsched-1400pfo.json",
#            "tile-sched/512t-2b-4k-1024-prem-kernelsched-1500pfo.json",
#            "tile-sched/512t-2b-4k-1024-prem-kernelsched-1600pfo.json",
#            "tile-sched/512t-2b-4k-1024-prem-kernelsched-1700pfo.json",
                ]
    
    showTimesAll(filenames1, titles1)
    plt.tight_layout()
    plt.show()
