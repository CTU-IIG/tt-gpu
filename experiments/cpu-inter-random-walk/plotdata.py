#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np

outdir = 'out/'

file1=outdir+'nozc-no-randwalk.json'
file2=outdir+'nozc-rnd-randwalk.json'
file3=outdir+'nozc-seq-randwalk.json'
file4=outdir+'zc-no-randwalk.json' 
file5=outdir+'zc-rnd-randwalk.json'
file6=outdir+'zc-seq-randwalk.json'

with open(file1) as f1:
    data1 = json.load(f1)

with open(file2) as f2:
    data2 = json.load(f2)

with open(file3) as f3:
    data3 = json.load(f3)

with open(file4) as f4:
    data4 = json.load(f4)

with open(file5) as f5:
    data5 = json.load(f5)

with open(file6) as f6:
    data6 = json.load(f6)

ax = plt.subplot(111)

ax.semilogx(data1['size'], data1['mean'],'r',marker='x')
ax.semilogx(data2['size'], data2['mean'],'b',marker='+')
ax.semilogx(data3['size'], data3['mean'],'g',marker='1')
ax.semilogx(data4['size'], data4['mean'],'m',marker='2')
ax.semilogx(data5['size'], data5['mean'],'c',marker='3')
ax.semilogx(data6['size'], data6['mean'],'y',marker='|')
ax.fill_between(data1['size'], data1['min'], data1['max'], facecolor='red', alpha=0.5)
ax.fill_between(data2['size'], data2['min'], data2['max'], facecolor='blue', alpha=0.5)
ax.fill_between(data3['size'], data3['min'], data3['max'], facecolor='green', alpha=0.5)
ax.fill_between(data4['size'], data4['min'], data4['max'], facecolor='magenta', alpha=0.5)
ax.fill_between(data5['size'], data5['min'], data5['max'], facecolor='cyan', alpha=0.5)
ax.fill_between(data6['size'], data6['min'], data6['max'], facecolor='yellow', alpha=0.5)

ax.grid(True, which="both")
plt.title('Random walk on increasing data set with different CPU interference on TX2\n')
plt.xlabel('Dataset size [kB]')
plt.ylabel('Cycles/Elem')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend((\
    'Memcopy, no interference',\
    'Memcopy, random interference',\
    'Memcopy, sequential interference',\
    'Zerocopy, no interference',\
    'Zerocopy, random interference',\
    'Zerocopy, sequential interference',\
    ),loc='center left',bbox_to_anchor=(1, 0.5))
plt.show()
