#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
import itertools

file1='out_ns/jitter.json'

with open(file1) as f1:
    data = json.load(f1)

def plotGeneralns(data1):
    marker = itertools.cycle(('o', 'x', '+', '.')) 
    mylegend = []
    blocks_sm = {}
    blocks_sm['0'] = []
    blocks_sm['1'] = []
    fig1, ax1 = plt.subplots(1,2,figsize=[7,2])

    for block in data1["blocks"]:
        mymarker = next(marker)
        mylegend.append("B"+block+" SM:"+data1['smid'][block])
        blocks_sm[data1['smid'][block]].append(block)
        x1 = list(range(1,len(data1['times'][block])+1))
        x2 = list(range(1,len(data1['differences'][block])+1))
        times = np.array(data1['times'][block])
        times = times - np.min(times)
        data1['times'][block] = times
        ax1[0].scatter(x1, data1['times'][block], marker=mymarker)
        bins = np.linspace(0,170,21)
        ax1[1].hist(data1['differences'][block], alpha=0.5, color='b', bins=bins)

    # configure plot stuff
    fig1.suptitle("Globaltimer jitter")
    ax1[0].grid(True, which="both")
    ax1[0].set_ylabel('Time [ns]')
    ax1[1].grid(True, which="both")
    ax1[0].legend(mylegend, loc='lower right')


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
    ax2[0].set_ylabel('Time [ns]')

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

#_, _,_,_, blocks_sm = plotGeneralns(data)
#print("Blocks on SM: "+str(blocks_sm))
#plt.show()
