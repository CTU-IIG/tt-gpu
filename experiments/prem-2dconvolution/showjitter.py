#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def getBlockIntervalsOfKernel(blockTimes):
    # Split into start end times
    start_times = []
    stop_times = []
    for i in range(len(blockTimes)):
        if (i%2) == 0:
            start_times.append(blockTimes[i])
        else:
            stop_times.append(blockTimes[i])

    intervals = []
    for start,stop in zip(start_times, stop_times):
        intervals.append(stop-start)

    return intervals

def getBlockIntervals(nofKernel, nofBlocks, blockTimes, nofRepetitions=1):
    intervalsKernel = []
    for kernel in range(0,nofKernel):
        # Get time subset of this blocks
        startindex = nofBlocks*kernel*nofRepetitions
        # Take only the first repetitions of blocktimes
        times = blockTimes[2*startindex:2*startindex+2*nofBlocks*nofRepetitions]
        
        # Retrive the blocks of the kernel
        intervals = getBlockIntervalsOfKernel(times)
        intervalsKernel.append(intervals)
    return intervalsKernel

def getKernelTimes(nofKernel, kernelTimes, nofRepetitions=1):
    timesKernel = []
    for kernel in range(0,nofKernel):
        # Get time subset of this blocks
        startindex = kernel*nofRepetitions
        # Take only the first repetitions of blocktimes
        times = kernelTimes[startindex:startindex+nofRepetitions]
        
        timesKernel.append(times)
    return timesKernel


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


def showTimesKernel(filename, title):
    with open(filename) as f1:
        data = json.load(f1)

    nofThreads = int(data['nofThreads'])
    nofBlocks  = int(data['nofBlocks'])
    nofKernel  = int(data['nofKernel'])
    nofRep     = int(data['nof_repetitions'])
    dataSize   = int(data['data_size'])
    measOH     = int(data['measOH'])
    blockTimes = data['blocktimes']
    kernelTimes= data['kerneltimes']
    smids      = data['smid']

    blockTimes = [float(i)*1e-6 for i in blockTimes]
    kernelTimes = [float(i) for i in kernelTimes]
    smids = [int(i) for i in smids]

    blockInt = getBlockIntervals(nofKernel, nofBlocks, blockTimes, nofRep)
    kernelExeT =  getKernelTimes(nofKernel, kernelTimes, nofRep)

    labels = []
    for i in range(0, nofKernel):
        labels.append("Kernel "+str(i))

    fig = plt.figure()
    fig.suptitle(title+" - Block intervals")
    drawHist(labels, blockInt, fig, "Blockintervals")
    drawCDF(labels, blockInt, fig, "Blockintervals")

    fig = plt.figure()
    fig.suptitle(title+" - Kernel times")
    drawHist(labels, kernelExeT, fig, "Kernel times")
    drawCDF(labels, kernelExeT, fig, "Kernel times")

def showTimesAll(filenames, titles):
    times_agg = []
    interval_agg = []
    for filename, title in zip(filenames, titles):
        print(filename +"/"+title)
        with open(filename) as f1:
            data = json.load(f1)

        nofThreads = int(data['nofThreads'])
        nofBlocks  = int(data['nofBlocks'])
        nofKernel  = int(data['nofKernel'])
        nofRep     = int(data['nof_repetitions'])
        dataSize   = int(data['data_size'])
        measOH     = int(data['measOH'])
        blockTimes = data['blocktimes']
        kernelTimes= data['kerneltimes']
        smids      = data['smid']

        blockTimes = [float(i)*1e-6 for i in blockTimes]
        kernelTimes = [float(i) for i in kernelTimes]
        smids = [int(i) for i in smids]

        blockInt = getBlockIntervals(nofKernel, nofBlocks, blockTimes, nofRep)
        kernelExeT =  getKernelTimes(nofKernel, kernelTimes, nofRep)

        times = []
        intervals = []
        for i in range(0,nofKernel):
            times.extend(kernelExeT[i])
            intervals.extend(blockInt[i])

        times_agg.append(times)
        interval_agg.append(intervals)


    fig = plt.figure()
    fig.suptitle("Block intervals")
    drawHist(titles, interval_agg, fig, "Histogram")
    drawCDF(titles, interval_agg, fig, "CDF")

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

    title13 = [
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
            "PREM: ni,nj = 1026x1022, 512 threads, 2 blocks, 2 kernel",
            "PREM: ni,nj = 1026x1022, 512 threads, 2 blocks, 1 kernel",
            "PREM: ni,nj = 1026x1022, 512 threads, 1 blocks, 1 kernel",
            "PREM: ni,nj = 1026x1022, 512 threads, 1 blocks, 2 kernel",
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
                "prem-leg-comp/512t-2b-4k-1024-legacy.json",
                "prem-leg-comp/512t-2b-2k-1024-legacy.json",
                "prem-leg-comp/512t-2b-1k-1024-legacy.json",
                "prem-leg-comp/512t-1b-1k-1024-legacy.json",
                "prem-leg-comp/512t-1b-2k-1024-legacy.json",
                ]

    filenames5 = [
                "prem-leg-comp/512t-2b-4k-1024-prem.json",
                "prem-leg-comp/512t-2b-2k-1024-prem.json",
                "prem-leg-comp/512t-2b-1k-1024-prem.json",
                "prem-leg-comp/512t-1b-1k-1024-prem.json",
                "prem-leg-comp/512t-1b-2k-1024-prem.json",
                ]
    
    #for title, filename in zip(titles, filenames):
    #    showTimesKernel(filename, title)

    #showTimesAll(filenames1, titles1)
    #showTimesAll(filenames2, titles2)
    #showTimesAll(filenames3, titles3)
    showTimesAll(filenames4, titles4)
    showTimesAll(filenames5, titles5)
    plt.show()
