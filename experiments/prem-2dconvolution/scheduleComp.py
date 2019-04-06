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

def drawCDF(labels, times, fig, title):
    ax = fig.add_subplot(1, 2, 1)
    for i, label in enumerate(labels):
        times_sorted = np.sort(times[i])
        # Normalize
        p = 1.0 * np.arange(len(times[i])) / (len(times[i])-1)
        ax.plot(times_sorted,p, label=label)

    ax.set_ylabel("Probability")
    ax.set_xlabel("Time [ms]")
#    ax.legend(loc='upper left')
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
        blockTimes = [float(i)*1e-9 for i in data['blocktimes']]
        kernelTimes = getKernelTimes(nofKernel, nofBlocks, nofRep, blockTimes)
        times_agg.append(kernelTimes)

    fig = plt.figure()
    fig.suptitle("Kernel times")
    drawHist(titles, times_agg, fig, "Histogram")
    drawCDF(titles, times_agg, fig, "CDF")



if __name__ == "__main__":
    titles1 = [
            "Legacy: ni,nj = 4096, 512 threads, 2 blocks, 4 kernel",
            "Legacy: ni,nj = 4096, 512 threads, 2 blocks, 2 kernel",
            "Legacy: ni,nj = 4096, 512 threads, 2 blocks, 1 kernel",
            "Legacy: ni,nj = 4096, 512 threads, 1 blocks, 1 kernel",
            "Legacy: ni,nj = 4096, 512 threads, 1 blocks, 2 kernel",
            "Legacy: ni,nj = 4096, 1024 threads, 1 blocks, 1 kernel",
    ]

    titles2 = [
            "Legacy: Legacy: ni,nj = 1024, 512 threads, 2 blocks, 4 kernel",
            "Legacy: Legacy: ni,nj = 1024, 512 threads, 2 blocks, 2 kernel",
            "Legacy: Legacy: ni,nj = 1024, 512 threads, 2 blocks, 1 kernel",
            "Legacy: Legacy: ni,nj = 1024, 512 threads, 1 blocks, 1 kernel",
            "Legacy: Legacy: ni,nj = 1024, 512 threads, 1 blocks, 2 kernel",
            "Legacy: Legacy: ni,nj = 1024, 1024 threads, 1 blocks, 1 kernel",
            ]

    titles3 = [
            "ni,nj = 512, 256 threads, 2 blocks, 4 kernel",
            "ni,nj = 512, 256 threads, 2 blocks, 2 kernel",
            "ni,nj = 512, 256 threads, 2 blocks, 1 kernel",
            "ni,nj = 512, 256 threads, 1 blocks, 1 kernel",
            "ni,nj = 512, 256 threads, 1 blocks, 2 kernel",
            "ni,nj = 512, 512 threads, 1 blocks, 1 kernel",
            ]

    titles4 = [
            "Legacy: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel",
            "Legacy: ni,nj = 1026x1022, 512 threads, 2 blocks, 2 kernel",
            "Legacy: ni,nj = 1026x1022, 512 threads, 2 blocks, 1 kernel",
            "Legacy: ni,nj = 1026x1022, 512 threads, 1 blocks, 1 kernel",
            "Legacy: ni,nj = 1026x1022, 512 threads, 1 blocks, 2 kernel",
            ]

    titles5 = [
            "PREM: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel",
            "Legacy: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel",
            "PREM old: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel",
            "PREM nosched: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel",
            #"PREM scheduled: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel",
            ]

    titles6 = [
            "PREM with barrier: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel",
            "PREM with barrier: ni,nj = 1026x1022, 512 threads, 2 blocks, 2 kernel",
            "PREM with barrier: ni,nj = 1026x1022, 512 threads, 2 blocks, 1 kernel",
            "PREM with barrier: ni,nj = 1026x1022, 512 threads, 1 blocks, 1 kernel",
            "PREM with barrier: ni,nj = 1026x1022, 512 threads, 1 blocks, 2 kernel",
            ]

    titles7 = [
            "PREM scheduled: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel",
            ]

    filenames1 = [
                "data-legacy/512t-2b-4k-4096.json",
                "data-legacy/512t-2b-2k-4096.json",
                "data-legacy/512t-2b-1k-4096.json",
                "data-legacy/512t-1b-1k-4096.json",
                "data-legacy/512t-1b-2k-4096.json",
                "data-legacy/1024t-1b-1k-4096.json",
                ]
    
    filenames2 = [
                "data-legacy-1024/512t-2b-4k-1024.json",
                "data-legacy-1024/512t-2b-2k-1024.json",
                "data-legacy-1024/512t-2b-1k-1024.json",
                "data-legacy-1024/512t-1b-1k-1024.json",
                "data-legacy-1024/512t-1b-2k-1024.json",
                "data-legacy-1024/1024t-1b-1k-1024.json",
                ]

    filenames3 = [
                "data-legacy-512/256t-2b-4k-512.json",
                "data-legacy-512/256t-2b-2k-512.json",
                "data-legacy-512/256t-2b-1k-512.json",
                "data-legacy-512/256t-1b-1k-512.json",
                "data-legacy-512/256t-1b-2k-512.json",
                "data-legacy-512/512t-1b-1k-512.json",
                ]

    filenames4 = [
                "out/512t-2b-4k-1024-legacy.json",
                "out/512t-2b-2k-1024-legacy.json",
                "out/512t-2b-1k-1024-legacy.json",
                "out/512t-1b-1k-1024-legacy.json",
                "out/512t-1b-2k-1024-legacy.json",
                ]

    filenames5 = [
                "out/512t-2b-4k-1024-prem.json",
                "out/512t-2b-4k-1024-legacy.json",
                "prem-leg-comp/512t-2b-4k-1024-prem.json",
                "out/512t-2b-4k-1024-prem-nosched.json"
                #"out/threads.json",
                ]
    filenames6 = [
                "prem-barrier/512t-2b-4k-1024-prem.json",
                "prem-barrier/512t-2b-2k-1024-prem.json",
                "prem-barrier/512t-2b-1k-1024-prem.json",
                "prem-barrier/512t-1b-1k-1024-prem.json",
                "prem-barrier/512t-1b-2k-1024-prem.json",
                ]
    filenames7 = [
                "out/threads.json",
                ]
    
    #for title, filename in zip(titles, filenames):
    #    showTimesKernel(filename, title)

    #showTimesAll(filenames1, titles1)
    #showTimesAll(filenames2, titles2)
    #showTimesAll(filenames3, titles3)
    showTimesAll(filenames4, titles4)
    showTimesAll(filenames5, titles5)
    #showTimesAll(filenames6, titles6)
    #showTimesAll(filenames7, titles7)
    plt.show()
