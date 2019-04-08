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
        #print("{:<30s}: mean: {:f}ms, min: {:f}ms, max: {:f}ms, jitter: {:f}ms".format(label, mean, minv, maxv, maxv-minv))

    ax.bar(centers_mean, y, width=width, yerr=[minerr, maxerr], alpha =0.5, hatch='/',ecolor='r', capsize=5)#, yerr=menStd) 
    ax.set_ylabel("Average execution time [ns]")
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
    ax.set_xlabel("Time [ns]")
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
    pf_agg = []
    c_agg = []
    wb_agg = []
    for filename, title in zip(filenames, titles):
        data = {}
        #print(filename +"/"+title)
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

        prefetchTimes, computeTimes, writebackTimes = getPhaseTimes(data, withWarmup=False)
        maxpf = max(prefetchTimes)
        maxc = max(computeTimes)
        maxwb = max(writebackTimes)
        print("Theoretical WCET for {:<10s}: {:f}ms".format(title, ((maxpf+maxc+maxwb)*data['tileCount'])*1e-6))
        pf_agg.append(prefetchTimes)
        c_agg.append(computeTimes)
        wb_agg.append(writebackTimes)

    fig = plt.figure()
    fig.suptitle("Prefetch times")
    drawHist(titles, pf_agg, fig, "Histogram")
    drawCDF(titles, pf_agg, fig, "CDF")
    fig = plt.figure()
    drawBarGraph(titles, pf_agg, fig, "Prefetch times")

    fig = plt.figure()
    fig.suptitle("Compute times")
    drawHist(titles, c_agg, fig, "Histogram")
    drawCDF(titles, c_agg, fig, "CDF")
    fig = plt.figure()
    drawBarGraph(titles, c_agg, fig, "Compute times")

    fig = plt.figure()
    fig.suptitle("Writeback times")
    drawHist(titles, wb_agg, fig, "Histogram")
    drawCDF(titles, wb_agg, fig, "CDF")
    fig = plt.figure()
    drawBarGraph(titles, wb_agg, fig, "Writeback times")

if __name__ == "__main__":
    titles1 = [
            "0ns",
            "500ns",
            "1000ns",
            "1500ns",
            "2000ns",
            "3000ns",
    ]

    filenames1 = [
            "data/512t-2b-4k-1024-prem-kernelsched-0pfo-3000wbo.json",  
            "data/512t-2b-4k-1024-prem-kernelsched-500pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-1000pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-1500pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-2000pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-2500pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-3000pfo-3000wbo.json",
                ]
    
    titles2 = [
            "0ns",
            "500ns",
            "1000ns",
            "1500ns",
            "2000ns",
            "3000ns",
    ]

    filenames2 = [
            "data/512t-2b-4k-1024-prem-tilesched-0pfo-3000wbo.json",  
            "data/512t-2b-4k-1024-prem-tilesched-500pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-tilesched-1000pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-tilesched-1500pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-tilesched-2000pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-tilesched-2500pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-tilesched-3000pfo-3000wbo.json",
                ]

    titles3 = [
            "0ns",
            "200ns",
            "400ns",
            "600ns",
            "800ns",
            "1000ns",
    ]

    filenames3 = [
            "data/512t-2b-4k-1024-prem-kernelsched-3000pfo-0wbo.json",  
            "data/512t-2b-4k-1024-prem-kernelsched-3000pfo-200wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-3000pfo-400wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-3000pfo-600wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-3000pfo-800wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-3000pfo-1000wbo.json"
                ]


    titles4 = [
            "0ns",
            "200ns",
            "400ns",
            "600ns",
            "800ns",
            "1000ns",
    ]

    filenames4 = [
            "data/512t-2b-4k-1024-prem-tilesched-3000pfo-0wbo.json",  
            "data/512t-2b-4k-1024-prem-tilesched-3000pfo-200wbo.json",
            "data/512t-2b-4k-1024-prem-tilesched-3000pfo-400wbo.json",
            "data/512t-2b-4k-1024-prem-tilesched-3000pfo-600wbo.json",
            "data/512t-2b-4k-1024-prem-tilesched-3000pfo-800wbo.json",
            "data/512t-2b-4k-1024-prem-tilesched-3000pfo-1000wbo.json"
                ]


    titles5 = [
            "No scheduler",
    ]

    filenames5 = [
            "data/512t-2b-4k-1024-prem-nosched-prof.json",  
                ]



#    showTimesAll(filenames1, titles1)
#    showTimesAll(filenames2, titles2)
#    showTimesAll(filenames3, titles3)
#    showTimesAll(filenames4, titles4)
    showTimesAll(filenames5, titles5)

    plt.tight_layout()
    plt.show()
