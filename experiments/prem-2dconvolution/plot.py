#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np


filename1 = "out/128kb-1ke-32th.json" 
filename2 = "out/128kb-1ke-128th.json"
filename3 = "out/128kb-1ke-256th.json"
filename4 = "out/128kb-1ke-512th.json"
filename5 = "out/128kb-1ke-1024th.json"

with open(filename1) as f1:
    data1 = json.load(f1)
with open(filename2) as f2:
    data2 = json.load(f2)
with open(filename3) as f3:
    data3 = json.load(f3)
with open(filename4) as f4:
    data4 = json.load(f4)
with open(filename5) as f5:
    data5 = json.load(f5)

def getData(data, name='blk'):
    meanv= np.array(data['mean'+name])
    minerr = meanv-np.array(data['min'+name])
    maxerr = np.array(data['max'+name])-meanv
    return (meanv, minerr, maxerr)

def plotData(data, marker, color, ax, ylabel='Cycles/Elem'):
    l = ax.errorbar(data1['nof_blocks'], data[0], yerr=[data[1], data[2]], fmt=color+marker+'--',linewidth=1,elinewidth=1,ecolor=color, capsize=5, capthick=0.5)
    ax.set_xticks(range(1,33,2))
    ax.grid(True, which="both")
    ax.set_xlabel('Nof Blocks')
    ax.set_ylabel(ylabel)
    return l

fig, ax = plt.subplots(3,2)
l1 = plotData(getData(data1,'blk'),'*','r',ax[0,0],'Cycles/Elem')
l2 = plotData(getData(data2,'blk'),'+','m',ax[1,0],'Cycles/Elem')
l3 = plotData(getData(data3,'blk'),'x','k',ax[1,1],'Cycles/Elem')
l4 = plotData(getData(data4,'blk'),'|','b',ax[2,0],'Cycles/Elem')
l5 = plotData(getData(data5,'blk'),'v','g',ax[2,1],'Cycles/Elem')
ax[0, 1].axis('off')

box = ax[0,1].get_position()
#ax[0,0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[0,1].legend((l1, l2, l3 ,l4, l5), (\
          '128kB, 32 threads',\
          '128kB, 128 threads',\
          '128kB, 256 threads',\
          '128kB, 512 threads',\
          '128kB, 1024 threads'\
          ),loc='center')
fig.suptitle('Coallocent sequential walk - multiple blocks')



fig, ax = plt.subplots(3,2)
l1 = plotData(getData(data1,'ker'),'*','r',ax[0,0],'Time [us]')
l2 = plotData(getData(data2,'ker'),'+','m',ax[1,0],'Time [us]')
l3 = plotData(getData(data3,'ker'),'x','k',ax[1,1],'Time [us]')
l4 = plotData(getData(data4,'ker'),'|','b',ax[2,0],'Time [us]')
l5 = plotData(getData(data5,'ker'),'v','g',ax[2,1],'Time [us]')
ax[0, 1].axis('off')

box = ax[0,1].get_position()
#ax[0,0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[0,1].legend((l1, l2, l3 ,l4, l5), (\
          '128kB, 32 threads',\
          '128kB, 128 threads',\
          '128kB, 256 threads',\
          '128kB, 512 threads',\
          '128kB, 1024 threads'\
          ),loc='center')
fig.suptitle('Coallocent sequential walk - multiple blocks - kerneltimes')



plt.show()
