#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def getIntervalsOfKernelAndRep(data, kernel, block):
    pfi = []
    ci  = []
    wbi = []
    # Get time subset of this blocks
    startIndex = 2 * data['tileCount'] * data['nofBlocks'] * kernel + 2 * data['tileCount'] * block
    nofElem = 2 * data['tileCount']

    # Take only the first repetitions of times
    pft = data['prefetchtimes'][startIndex: startIndex + nofElem]
    ct = data['computetimes'][startIndex: startIndex + nofElem]
    wbt = data['writebacktimes'][startIndex: startIndex + nofElem]

    for i in range(0, len(pft),2):
        pfi.append(pft[i+1]-pft[i])
        ci.append(ct[i+1]-ct[i])
        wbi.append(wbt[i+1]-wbt[i])
        
        if pfi[-1]<=0 or ci[-1] <= 0 or wbi[-1] <= 0:
                return pft, ct, wbt

    return pfi, ci, wbi

def getPhaseTimes(data, withWarmup=False):
    prefetchTimes = []
    computeTimes = []
    writebackTimes = []
    for kernel in range(0, data['nofKernel']):
        for block in range(0, data['nofBlocks']):
            pft, ct, wbt = getIntervalsOfKernelAndRep(data, kernel, block)

            if not withWarmup:
                pft = pft[1:]
                ct = ct[1:]
                wbt = wbt[1:]

            prefetchTimes.extend(pft)
            computeTimes.extend(ct)
            writebackTimes.extend(wbt)
    return  prefetchTimes, computeTimes, writebackTimes 

def drawCDF(labels, times, fig, title, plotIndex=1, plotdim=(1,1)):
    ax = fig.add_subplot(plotdim[0], plotdim[1], plotIndex)
    for i, label in enumerate(labels):
        times_sorted = np.sort(times[i])
        print("{:<10s} WCET: {:f}".format(label, np.max(times[i])))
        # Normalize
        p = 1.0 * np.arange(len(times[i])) / (len(times[i])-1)
        ax.plot(times_sorted,p, label=label)

    ax.set_ylabel("$<=x$ [\%]")
    ax.set_xlabel("Time [ns]")
    ax.set_title(title)
    ax.grid(True)

    if(plotIndex == plotdim[0] * plotdim[1]):
            ax.legend(labels, loc='lower right')

def addPlot(filename, title, fig, index, dim):
    data = {}
    with open(filename) as f1:
        jsondata = json.load(f1)

    data['nofThreads'] = int(jsondata['nofThreads'])
    data['nofBlocks']  = int(jsondata['nofBlocks'])
    data['nofKernel']  = int(jsondata['nofKernel'])
    data['nofRep']     = int(jsondata['nof_repetitions'])
    data['tileCount']     = int(jsondata['tileCount'])
    data['prefetchtimes'] = [float(i) for i in jsondata['prefetchtimes']]
    data['computetimes'] = [float(i) for i in jsondata['computetimes']]
    data['writebacktimes'] = [float(i) for i in jsondata['writebacktimes']]

    pT, cT, wbT = getPhaseTimes(data, withWarmup=True)
    drawCDF(['Prefetch', 'Compute','Writeback'],[pT, cT, wbT] , fig,
            title, plotIndex=index, plotdim=dim)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
file1="data/512t-1b-1k-1024-prem-nosched-prof.json"
file2="data/512t-2b-4k-1024-prem-nosched-prof.json"
dim = (1,2)

fig = plt.figure(figsize=[8,1.75])
addPlot(file1, "PREM phases: 1 kernel with 1 block", fig, 1, dim)
addPlot(file2, "PREM phases: 4 kernels with 2 blocks",  fig, 2, dim)
fig.savefig('allphasescdf.pdf', format='pdf', bbox_inches='tight')
#plt.show()


