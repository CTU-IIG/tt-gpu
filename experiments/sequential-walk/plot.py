#!/usr/bin/python3

file1='out/L2.json'
file2='out/L1_default.json'
file3='out/L1_smaller.json'
file4='out/L1_bigger.json'
file5='out/L1_equal.json'
file6='out/shm.json'

import json
import matplotlib.pyplot as plt
import numpy as np

with open(file1) as f1:
    datal1 = json.load(f1)

with open(file2) as f2:
    datal2 = json.load(f2)

with open(file3) as f3:
    datal3 = json.load(f3)

with open(file4) as f4:
    datal4 = json.load(f4)

with open(file5) as f5:
    datal5 = json.load(f5)

with open(file6) as f6:
    datal6 = json.load(f6)

# red dashes, blue squares and green triangles
plt.plot(datal1['size'], datal1['mean'],'r',marker='x')
plt.plot(datal2['size'], datal2['mean'],'b',marker='+')
plt.plot(datal3['size'], datal3['mean'],'g',marker='1')
plt.plot(datal4['size'], datal4['mean'],'m',marker='2')
plt.plot(datal5['size'], datal5['mean'],'c',marker='3')
plt.plot(datal6['size'], datal6['mean'],'y',marker='|')
plt.fill_between(datal1['size'], datal1['min'], datal1['max'], facecolor='red', alpha=0.5)
plt.fill_between(datal2['size'], datal2['min'], datal2['max'], facecolor='blue', alpha=0.5)
plt.fill_between(datal3['size'], datal3['min'], datal3['max'], facecolor='green', alpha=0.5)
plt.fill_between(datal4['size'], datal4['min'], datal4['max'], facecolor='magenta', alpha=0.5)
plt.fill_between(datal5['size'], datal5['min'], datal5['max'], facecolor='cyan', alpha=0.5)
plt.fill_between(datal6['size'], datal6['min'], datal6['max'], facecolor='yellow', alpha=0.5)
plt.axis([1,2500 , 0, 370])
plt.semilogx(datal1['size'], np.sin(2*np.pi*np.asarray(datal1['size'])))
plt.grid(True, which="both")
plt.title('Random walk on increasing data set\n')
plt.xlabel('Dataset size [kB]')
plt.ylabel('Cycles/Elem')
plt.gca().legend(('L2 (-Xptxas -dlcm=cg)',\
    'L1 (CachePreferNone)',\
    'L1 (CachePreferShared)',\
    'L1 (CachePreferL1)',\
    'L1 (CachePreferEqual)',\
    'L1 (Shared occ.)'))
plt.show()
