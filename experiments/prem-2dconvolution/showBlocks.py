#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

hatches = ['//', '\\\\', '/', '\\', 'x', 'xx','o','O', '.']

class Block:
    def __init__ (self, kernel, nofThreads, start, end, smid, bid, offset=0):
        self.kernel = kernel
        self. nofThreads = nofThreads
        self.threadOffset = offset
        self.start = start
        self.end = end
        self.smid = smid
        self.id = bid
        self.maxThreadsOnSM = 2048
        self.intervals = []

    def draw(self, ax, colors):
        thread_lower = self.threadOffset
        thread_upper = self.threadOffset + self.nofThreads
        if len(self.intervals) == 0:
            # Print without prem intervals
            ax.broken_barh([(self.start, self.end-self.start)], (thread_lower, thread_upper-thread_lower), facecolor=colors[self.kernel], alpha=0.4, hatch=hatches[self.kernel],edgecolor='k',linewidth=1.0)

            #ax.fill_between([self.start, self.end], [thread_upper, thread_upper], [thread_lower, thread_lower], facecolor=colors[self.kernel], alpha=0.4, hatch=hatches[self.kernel],edgecolor='k',linewidth=1.0)
        else:
            # Print with prem intervals
            ax.broken_barh(self.intervals, (thread_lower, thread_upper-thread_lower), facecolor=(colors[0],colors[1],colors[2]), alpha=0.4, hatch=hatches[self.kernel],edgecolor='k',linewidth=1.0)

        
        ax.text(self.start+(self.end-self.start)/2,thread_lower+(thread_upper-thread_lower)/2,"K:"+str(self.kernel)+":"+str(self.id),horizontalalignment='center',verticalalignment='center')

    def isOverlap(self, block):
        mythread_lower = self.threadOffset
        mythread_upper = self.threadOffset + self.nofThreads
        otherthread_lower = block.threadOffset
        otherthread_upper = block.threadOffset + block.nofThreads
        # Time overlap and thread overlap check
        if (block.end <= self.start) or (self.end <= block.start) or (otherthread_upper<=mythread_lower) or (mythread_upper <= otherthread_lower):
            #print("K"+str(self.kernel)+":"+str(self.id)+" No overlap")
            # No overlap
            return False
        else:
            #print("K"+str(self.kernel)+":"+str(self.id)+" Overlap")
            return True
    
    def incThreadOffset(self, amount):
        if self.threadOffset + self.nofThreads + amount <= self.maxThreadsOnSM:
            self.threadOffset += amount

    def addPREMintervals(self, prefetch, compute, writeback):
        for i in range(0, len(prefetch), 2):
            self.intervals.append((prefetch[i], prefetch[i+1]-prefetch[i]))
            self.intervals.append((compute[i], compute[i+1]-compute[i]))
            self.intervals.append((writeback[i], writeback[i+1]-writeback[i]))

        print("Interval len: "+str(len(self.intervals)))
def retrieveBlocksOfKernel(kernel, nofThreads, times, smid, tileCount=0, prefetchtimes=[], computetimes=[], writebacktimes=[]):
    # Split into start end times
    start_times = []
    stop_times = []
    for i in range(len(times)):
        if (i%2) == 0:
            start_times.append(times[i])
        else:
            stop_times.append(times[i])

    blocks = []
    for blockid in range(0,len(smid)):
            block = Block(kernel, nofThreads, times[2*blockid], times[2*blockid+1], smid[blockid], blockid)
            block.addPREMintervals(prefetchtimes[blockid*2*tileCount:blockid*2*tileCount + 2*tileCount], computetimes[blockid*2*tileCount:blockid*2*tileCount + 2*tileCount], writebacktimes[blockid*2*tileCount:blockid*2*tileCount + 2*tileCount])
            blocks.append(block)

    return blocks

def assignBlocksToSM(nofKernel, nofBlocks, nofThreads, blockTimes, smids, nofRepetitions=1, tileCount=0, prefetchtimes=[], computetimes=[], writebacktimes=[]):
    agg_sm = {}
    for kernel in range(0,nofKernel):

        # Get time subset of this blocks
        startindex = nofBlocks*kernel*nofRepetitions
        # Take only the first repetitions of blocktimes
        times = blockTimes[2*startindex:2*startindex+2*nofBlocks]
        smid = smids[startindex:startindex+nofBlocks]

        startindex = kernel*nofBlocks*2*tileCount
        
        pt =   prefetchtimes[startindex:startindex+2*nofBlocks*tileCount]
        ct =    computetimes[startindex:startindex+2*nofBlocks*tileCount]
        wbt = writebacktimes[startindex:startindex+2*nofBlocks*tileCount]
        
        # Retrive the blocks of the kernel
        blocks = retrieveBlocksOfKernel(kernel, nofThreads, times, smid, tileCount, pt, ct, wbt)
        # Put the retrived blocks to the according SM
        for block in blocks:
            if block.smid not in agg_sm:
                agg_sm[block.smid] = []
            agg_sm[block.smid].append(block)
            print("K"+str(block.kernel)+":"+str(block.id) +" - Start|End " +str(block.start)+"|"+str(block.end)+ " Interval: "+str((block.end-block.start)*1e-9)+"s")
    return agg_sm
              

def findFreeSlot(occupied, block):
    offset = 0
    restart = True
    while restart:
        for block_fix in occupied:
            if block.isOverlap(block_fix):
                # We have some overlap int 2D space here
                block.incThreadOffset(block_fix.nofThreads)
                # Restart the search
                restart = True
                break
            else:
                restart = False
    #print("Overlap K"+str(block.kernel)+":"+str(block.id)+" with K"+str(block_fix.kernel)+":"+str(block_fix.id)+ "Interval: "+str(interval)+"s")

def adjustThreadOffset(agg_sm):

    #Adjust the thread offset for the blocks on each SM
    for sm in agg_sm.keys():
        # Iterate through all blocks
        print("Adjust threads on SM " + str(sm))
        occupied = []
        for i, block in enumerate(agg_sm[sm]):
            if len(occupied) == 0:
                occupied.append(block)
                continue
            # Find appropriate slow
            findFreeSlot(occupied, block)

            #Store this block to occupied
            occupied.append(block)


def drawBlocks(agg_sm, nofKernel, minTime, maxTime, title):
    fig = plt.figure()
    fig.suptitle("Blocks scheduled on SM's\n"+ title)
    colors = cm.get_cmap('viridis', 4).colors
    if len(agg_sm.keys()) < 2:
        # add dummy key
        agg_sm[1]= []

    for i, sm in enumerate(agg_sm.keys()):
        ax = fig.add_subplot(2, 1, i+1)
        for block in agg_sm[sm]:
            block.draw(ax, colors)
        ax.set_ylabel("NofThreads")
        ax.set_yticks(range(0,2049,512))
        ax.set_xlim(0.001,0.00112)
        ax.set_xlabel("Time [s]")
        ax.set_title("SM "+str(i))
        ax.grid(True)


def drawScenario(filename, title):
    with open(filename) as f1:
        data = json.load(f1)

    nofThreads = int(data['nofThreads'])
    nofBlocks  = int(data['nofBlocks'])
    nofKernel  = int(data['nofKernel'])
    nofRep     = int(data['nof_repetitions'])
    dataSize   = int(data['data_size'])
    blockTimes = data['blocktimes']
    kernelTimes= data['kerneltimes']
    smids      = data['smid']

    tileCount = 0
    prefetchtimes = []
    computetimes = []
    writebacktimes = []
    if "tileCount" in data:
        tileCount = int(data['tileCount'])
        prefetchtimes = [float(i) for i in data['prefetchtimes']]
        computetimes = [float(i) for i in data['computetimes']]
        writebacktimes = [float(i) for i in data['writebacktimes']]

    blockTimes = [float(i) for i in blockTimes]
    kernelTimes = [float(i) for i in kernelTimes]
    smids = [int(i) for i in smids]

    minTime = min(blockTimes)
    maxTime = max(blockTimes)
    blockTimes= [(i-minTime)*1e-9 for i in blockTimes]
    prefetchtimes= [(i-minTime)*1e-9 for i in prefetchtimes[:-1]]
    computetimes= [(i-minTime)*1e-9 for i in computetimes[:-1]]
    writebacktimes= [(i-minTime)*1e-9 for i in writebacktimes[:-1]]


    agg_sm = assignBlocksToSM(nofKernel, nofBlocks, nofThreads, blockTimes, smids, nofRep, tileCount, prefetchtimes, computetimes, writebacktimes)
    adjustThreadOffset(agg_sm)
    drawBlocks(agg_sm, nofKernel, minTime, maxTime, title)


if __name__ == "__main__":
    titles = [
            "512 threads, 2 blocks, 4 kernel",
            "512 threads, 2 blocks, 2 kernel",
            "512 threads, 2 blocks, 1 kernel",
            "512 threads, 1 blocks, 1 kernel",
            "512 threads, 1 blocks, 2 kernel",
            "1024 threads, 1 blocks, 1 kernel",
            "PREM: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel",
            "Legacy: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel",
            "PREM barrier: ni,nj = 1026x1022, 512 threads, 2 blocks, 4 kernel",
    #         " TEST"
            ]

    filenames = [
                "data-legacy/512t-2b-4k-4096.json",
                "data-legacy/512t-2b-2k-4096.json",
                "data-legacy/512t-2b-1k-4096.json",
                "data-legacy/512t-1b-1k-4096.json",
                "data-legacy/512t-1b-2k-4096.json",
                "data-legacy/1024t-1b-1k-4096.json",
                "prem-leg-comp/512t-2b-4k-1024-prem.json",
                "prem-leg-comp/512t-2b-4k-1024-legacy.json",
                "prem-barrier/512t-2b-4k-1024-prem.json",
    #             "out/data.json"
                ]
    for title, filename in zip(titles, filenames):
        drawScenario(filename, title)
    plt.show()
