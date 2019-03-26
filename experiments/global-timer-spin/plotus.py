#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
import itertools
import decimal

file1='out_us/jitter.json'

with open(file1) as f1:
    data = json.load(f1)

cockRateHz = int(data["clockRatekHz"])*1000.0
stepns = int(data["stepns"])


def plotGeneralns(data1):
    marker = itertools.cycle(('o', 'x', '+', '.')) 
    mylegend = []
    blocks_sm = {}
    blocks_sm['0'] = []
    blocks_sm['1'] = []
    fig1, ax1 = plt.subplots(1,2)

    for block in data1["blocks"]:
        mymarker = next(marker)
        mylegend.append("Block: "+block+" on SM "+data1['smid'][block])
        blocks_sm[data1['smid'][block]].append(block)
        x1 = list(range(1,len(data1['times'][block])+1))
        x2 = list(range(1,len(data1['differences'][block])+1))
        ax1[0].scatter(x1, data1['times'][block], marker=mymarker)
        ax1[1].scatter(x2, data1['differences'][block], marker=mymarker)

    # configure plot stuff
    ax1[0].grid(True, which="both")
    ax1[0].set_title('Global timer jitter TX2\n')
    ax1[0].set_ylabel('Nof Cycles')
    ax1[1].grid(True, which="both")
    ax1[1].set_title('Global timer jitter TX2\n')
    box = ax1[1].get_position()
    ax1[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1[1].legend(mylegend,loc='center left',bbox_to_anchor=(1, 0.5))


    #Get differences of block timestamps
    block0 = blocks_sm['0'][0]
    block1 = blocks_sm['0'][1]
    diff = [x1 - x2 for (x1, x2) in zip(data1['times'][block0], data1['times'][block1])]
    x1 = list(range(1,len(data1['times'][block])+1))

    fig2, ax2 = plt.subplots(1,2)
    ax2[0].scatter(x1, diff, marker=mymarker)
    ax2[0].grid(True, which="both")
    ax2[0].set_title('Diff SM 0 (block' + block0 +'-block'+block1+')')
    ax2[0].set_xlabel('Iteration')
    ax2[0].set_ylabel('Nof Cycles')

    #Get differences of block timestamps
    block0 = blocks_sm['1'][0]
    block1 = blocks_sm['1'][1]
    diff = [x1 - x2 for (x1, x2) in zip(data1['times'][block0], data1['times'][block1])]
    x1 = list(range(1,len(data1['times'][block])+1))

    ax2[1].scatter(x1, diff, marker=mymarker)
    ax2[1].grid(True, which="both")
    ax2[1].set_title('Diff SM 1 (block' + block0 +'-block'+block1+')')
    ax2[1].set_xlabel('Iteration')
    return fig1, ax1,fig2, ax2, blocks_sm

_, _,_,_, blocks_sm = plotGeneralns(data)
print("Blocks on SM: "+str(blocks_sm))
minDiff = min(data['differences']['0'])/cockRateHz*1000000
maxDiff = max(data['differences']['0'])/cockRateHz*1000000
print("Cycles: "+str(min(data['differences']['0']))+" equals " + str(minDiff) +"us")
print("Cycles: "+str(max(data['differences']['0']))+" equals " + str(maxDiff) +"us")
print("Difference " + str((maxDiff-minDiff)*1000) +"ns")
print("Configured Stepsize " + str(stepns/1000.0) +"us")
plt.show()
