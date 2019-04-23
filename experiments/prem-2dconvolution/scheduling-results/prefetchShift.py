#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


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

def drawBarGraph(xlabels, labels, times, fig, title, times_stack=[]):
    ax = fig.add_subplot(1, 1, 1)
    y = []
    minerr = []
    maxerr = []
    jitter_per = []
    centers_mean = np.arange(0,len(xlabels),1)
    jitter_text = []
    maxax1 = 0
    maxax2 = 0
    if len(times_stack) == len(times):
        centers_jitter = np.arange(0.425-0.1,len(xlabels),1)
        centers_jitter_stack = np.arange(0.425+0.1,len(xlabels),1)
        width_jitter = 0.2
        err_offset = 0.1

        y_stack = []
        minerr_stack = []
        maxerr_stack = []
        jitter_per_stack = []
        jitter_text_stack = []
    else:
        centers_jitter = np.arange(0.425,len(xlabels),1)
        width_jitter = 0.4
        err_offset = 0

    centers_labels = np.arange(0.425/2, len(xlabels),1)
    width = 0.4
    handles = []

    for i, label in enumerate(xlabels):
        mean = np.mean(times[i])
        minv = np.min(times[i])
        maxv = np.max(times[i])
        y.append(mean)
        minerr.append(mean-minv)
        maxerr.append(maxv-mean)
        jitter = maxv-minv
        maxax1 = max(maxax1, maxv)

        if len(times_stack) == len(times):
            mean_stack = np.mean(times_stack[i])
            minv = np.min(times_stack[i])
            maxv = np.max(times_stack[i])
            y_stack.append(mean_stack)
            minerr_stack.append(mean_stack-minv)
            maxerr_stack.append(maxv-mean_stack)
            jitter_stack = maxv-minv
            jitter_per_stack.append((jitter_stack/(mean+mean_stack))*100)
            jitter_text_stack.append("{:.2f}\%".format(jitter_per_stack[-1]))
            maxax1 = max(maxax1, maxv)
            maxax2 = max(maxax2, jitter_per_stack[-1])

            jitter_per.append((jitter/(mean+mean_stack))*100)
        else:
            jitter_per.append((jitter/mean)*100)

        maxax2 = max(maxax2, jitter_per[-1])
        jitter_text.append("{:.2f}\%".format(jitter_per[-1]))


    h = ax.bar(centers_mean, y, width=width, alpha =0.5, hatch='\\\\', label=labels[0])
    ax.errorbar(x=centers_mean-err_offset, y=y, yerr=[minerr, maxerr], ecolor='r', capsize=5, fmt='r.')
    handles.append(h)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    h = ax2.bar(centers_jitter, jitter_per,
            width=width_jitter,color='r', hatch='\\\\\\\\', alpha=0.5, label=labels[1])
    handles.append(h)
    ax2.set_ylabel("Jitter relative to\naverage execution time [\%]")
    
    # Write jitter bar text labels
    for i in range(len(centers_jitter)):
        ax2.text(x = centers_jitter[i] , y = jitter_per[i], s = jitter_text[i], size = 8, va='bottom', ha='center', rotation=90)

    if len(times_stack) == len(times):
        for i in range(len(centers_jitter_stack)):
            ax2.text(x = centers_jitter_stack[i] , y = jitter_per_stack[i], s = jitter_text_stack[i], size = 8, va='bottom', ha='center', rotation=90)

        h = ax.bar(centers_mean, y_stack, width=width, bottom=y, alpha =0.5, hatch='//',  label=labels[2])
        handles.append(h)
        ax.errorbar(x=centers_mean+err_offset, y=[i+j for i,j in zip(y,y_stack)], yerr=[minerr_stack, maxerr_stack], ecolor='g', capsize=5, fmt='g.')
        h = ax2.bar(centers_jitter_stack, jitter_per_stack,
                width=width_jitter, color='g', hatch='////', alpha=0.5, label=labels[3])
        handles.append(h)


    # Set axis properties
    ax.set_ylabel("Average execution time [ns]")
    ax.set_xticks(centers_labels)
    ax.set_xticklabels(xlabels,rotation=45, ha='right')
    #ax.set_title(title)

    #ax.set_ylim(0,maxax1+75)
    ax2.set_ylim(0,maxax2+75)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=handles, loc='lower left', fontsize=8,  bbox_to_anchor=(0,1.02,1,0.2), mode= "expand", borderaxespad=0, ncol=4)
    ax.grid(True)

def drawCDF(labels, times, fig, title, nofPlots=2):
    ax = fig.add_subplot(1, nofPlots, 1)
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

def drawHist(labels, times, fig, title):
    ax = fig.add_subplot(1, 2, 2)
    for i, label in enumerate(labels):
        ax.hist(times[i], alpha = 0.5, label=label)

    ax.set_ylabel("Count")
    ax.set_xlabel("Time [ns]")
    ax.legend(loc='upper left')
    ax.set_title(title)

def showTimesAll(filenames, titles, mergePhaseCDF=False):
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
    fig = plt.figure(figsize=[7,2])
    labels = ['Prefetch time', 'Prefetch jitter', 'Compute time', 'Compute jitter']
    drawBarGraph(titles, labels, pf_agg, fig, "Prefetch times", c_agg)
    fig.savefig('phaseshift-prefetch.pdf', format='pdf', bbox_inches='tight')

    fig = plt.figure()
    fig.suptitle("Compute times")
    drawHist(titles, c_agg, fig, "Histogram")
    drawCDF(titles, c_agg, fig, "CDF")
    fig = plt.figure()
    labels = ['Compute time', 'Compute jitter']
    drawBarGraph(titles, labels, c_agg, fig, "Compute times")

    fig = plt.figure(figsize=[7,2])
    fig.suptitle("Writeback times")
    drawHist(titles, wb_agg, fig, "Histogram")
    drawCDF(titles, wb_agg, fig, "CDF")
    fig = plt.figure(figsize=[7,2])
    labels = ['Writeback execution time', 'Writeback jitter']
    drawBarGraph(titles, labels, wb_agg, fig, "Writeback times")
    fig.savefig('phaseshift-writeback.pdf', format='pdf', bbox_inches='tight')

    if mergePhaseCDF:
        fig = plt.figure(figsize=[7,2])
        fig.suptitle("Phase times")
        tmp = []
        tmp.extend(pf_agg)
        tmp.extend(c_agg)
        tmp.extend(wb_agg)
        drawCDF(['Prefetch', 'Compute','Writeback'], tmp, fig, "", nofPlots=1)
        fig.savefig('allphasescdf.pdf', format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    titles1 = [
            "0 ns",
            "500 ns",
            "1000 ns",
            "1500 ns",
            "2000 ns",
            "3000 ns",
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
            "0 ns",
            "500 ns",
            "1000 ns",
            "1500 ns",
            "2000 ns",
            "3000 ns",
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
            "0 ns",
            "200 ns",
            "400 ns",
            "600 ns",
            "800 ns",
            "1000 ns",
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
            "0 ns",
            "200 ns",
            "400 ns",
            "600 ns",
            "800 ns",
            "1000 ns",
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


    #kernelsched PF
    showTimesAll(filenames1, titles1)
    #tilesched PF
#    showTimesAll(filenames2, titles2)
    #kernelsched WB
#    showTimesAll(filenames3, titles3)
    #tilesched WB
#    showTimesAll(filenames4, titles4)
    #nosched 
#    showTimesAll(filenames5, titles5, mergePhaseCDF=True)

#    plt.tight_layout()
#    plt.show()
