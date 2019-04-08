#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np


filename1 = "out-warmup/128kb-1ke-32th.json" 
filename2 = "out-warmup/128kb-1ke-128th.json"
filename3 = "out-warmup/128kb-1ke-256th.json"
filename4 = "out-warmup/128kb-1ke-512th.json"
filename5 = "out-warmup/128kb-1ke-1024th.json"

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

def getData(data):
    meanv= np.array(data['mean'])
    minerr = meanv-np.array(data['min'])
    maxerr = np.array(data['max'])-meanv
    return (meanv, minerr, maxerr)

def plotData(data, marker, color, ax):
    l = ax.errorbar(data1['nof_blocks'], data[0], yerr=[data[1], data[2]], fmt=color+marker+'--',linewidth=1,elinewidth=1,ecolor=color, capsize=5, capthick=0.5)
    ax.set_xticks(range(1,33,2))
    ax.grid(True, which="both")
    ax.set_xlabel('Nof Blocks')
    ax.set_ylabel('Cycles/Elem')
    return l

fig, ax = plt.subplots(3,2)
l1 = plotData(getData(data1),'*','r',ax[0,0])
l2 = plotData(getData(data2),'+','m',ax[1,0])
l3 = plotData(getData(data3),'x','k',ax[1,1])
l4 = plotData(getData(data4),'|','b',ax[2,0])
l5 = plotData(getData(data5),'v','g',ax[2,1])
ax[0, 1].axis('off')

#fig.legend((l1, l2, l3, l4, l5), (\
#          '128kB, 32 threads',\
#          '128kB, 128 threadst',\
#          '128kB, 256 threadst',\
#          '128kB, 512 threads',\
#          '128kB, 1024 threads'\
#          ), 'best')

box = ax[0,1].get_position()
#ax[0,0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax[0,1].legend((l1, l2, l3 ,l4, l5), (\
          '128kB, 32 threads',\
          '128kB, 128 threadst',\
          '128kB, 256 threadst',\
          '128kB, 512 threads',\
          '128kB, 1024 threads'\
          ),loc='center')
fig.suptitle('Coallocent sequential walk - multiple blocks')
plt.xlabel('Nof Blocks')
plt.ylabel('Cycles/Elem')
plt.show()
