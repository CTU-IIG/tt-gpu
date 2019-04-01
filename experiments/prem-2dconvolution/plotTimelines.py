#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np


filename = "out/data.json"

numberOfSM = 2

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

blockTimes = [float(i)/1e9 for i in blockTimes]
kernelTimes = [float(i) for i in kernelTimes]
smids = [int(i) for i in smids]


def getBlockTimeLine(kernel, nofBlocks, blockTimes, smids):

    # Copy out only kernel block times
    startindex = nofBlocks*kernel
    times = blockTimes[2*startindex:2*startindex+2*nofBlocks]
    smid = smids[startindex:startindex+nofBlocks]

    # Split into start end times
    start_times = []
    stop_times = []
    for i in range(len(times)):
        if (i%2) == 0:
            start_times.append(times[i])
        else:
            stop_times.append(times[i])

    timelines = {}
    for i in range(0,numberOfSM):
        # Sort SMid
        start_sm = [time for time, smid in zip(start_times, smids) if smid == i]
        end_sm = [time for time, smid in zip(stop_times, smids) if smid == i]

        start_sm.sort(reverse=True)
        end_sm.sort(reverse=True)

        timeline_times = []
        timeline_blockCount = []
        timeline_times.append(0.0)
        timeline_blockCount.append(0)

        current_block_count = 0
        while True:
            if (len(start_sm) == 0) and (len(end_sm) == 0):
                break
            if len(end_sm) == 0:
                print("Error! The last block end time was before a start time.")

            current_time = 0.0
            previous_block_count = current_block_count
            if len(start_sm) != 0:
                if start_sm[-1] == end_sm[-1]:
                    # A block started and ended at the same time, don't change the
                    # current thread count.
                    current_time = start_sm.pop()
                    end_sm.pop()
                elif start_sm[-1] <= end_sm[-1]:
                    # A block started, so increase the thread count
                    current_time = start_sm.pop()
                    current_block_count += 1
                else:
                    # A block ended, so decrease the thread count
                    current_time = end_sm.pop()
                    current_block_count -= 1
            else:
                current_time = end_sm.pop()
                current_block_count -= 1

            # Make sure that changes between numbers of running threads are abrupt.
            # Do this by only changing the number of blocks at the instant they
            # actually change rather than interpolating between two values.
            timeline_times.append(current_time)
            timeline_blockCount.append(previous_block_count)
            # Finally, append the new thread count.
            timeline_times.append(current_time)
            timeline_blockCount.append(current_block_count)
        timelines[i] = {}
        timelines[i]['times'] = timeline_times
        timelines[i]['blockcount'] = timeline_blockCount
    return timelines


def drawTimeline(times, blockCount, ax, minv, maxv, title):
        ax.plot(times, blockCount)
        ax.set_xlim(minv-0.01, maxv+0.01)
        ax.set_yticks(range(0, max(blockCount)+1,1))
        h = ax.set_ylabel(title+"\nBlock count", rotation=0, labelpad=40)
        ax.grid()

minv = min(blockTimes)
maxv = max(blockTimes)

fig = plt.figure()
fig.suptitle("Timelines of resident blocks per SM")
old_labels = []
for i in range(0, nofKernel):
    timelines = getBlockTimeLine(i, nofBlocks, blockTimes, smids)
    ax = fig.add_subplot(nofKernel*2, 1, 2*i+1)
    drawTimeline(timelines[0]['times'], timelines[0]['blockcount'],ax,minv,maxv,"SM0: Kernel "+str(i))
    ax = fig.add_subplot(nofKernel*2, 1, 2*i+2)
    drawTimeline(timelines[1]['times'], timelines[1]['blockcount'],ax,minv,maxv,"SM1: Kernel "+str(i))

ax.set_xlabel("Time [s]")
plt.show()
