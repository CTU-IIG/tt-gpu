#!/usr/bin/python3
import json
import matplotlib.pyplot as plt
import numpy as np
import itertools

file1='out_us/jitter.json'
file2='out_ns/jitter.json'

interval = (0,50)

with open(file1) as f1:
    dataus = json.load(f1)

with open(file2) as f2:
    datans = json.load(f2)

def plot(dataus, datans, figsize=[7,2], labels=['Original', 'Nvprof B0', 'Nvprof B1']):
    fig1, ax1 = plt.subplots(1,2,figsize=figsize)

    timesus = np.array(dataus['times']['0'])
    timesus = timesus-timesus[0]
    timesns = np.array(datans['times']['0'])
    timesns = timesns-timesns[0]
    diffsus = np.array(dataus['differences']['0'])
    diffsns = np.array(datans['differences']['0'])

    timesnsb1 = np.array(datans['times']['1'])
    timesnsb1 = timesnsb1-timesnsb1[0]
    diffsnsb1 = np.array(datans['differences']['1'])

    x1 = list(range(1,len(timesus)+1))
    ax1[0].scatter(x1, timesus, color='r', marker='x')
    ax1[0].scatter(x1, timesns, color='b', marker='+')
    ax1[0].scatter(x1, timesnsb1, color='g', marker='1')

    bins = np.linspace(0,1100,50)
    ax1[1].hist(diffsus, alpha=0.5, color='r', hatch='//', bins=bins)
    ax1[1].hist(diffsns, alpha=0.5, color='b', hatch='\\\\', bins=bins)
    ax1[1].hist(diffsnsb1, alpha=0.5, color='g', hatch='\\\\////', bins=bins)

#    fig1.suptitle("Globaltimer resolution")
    ax1[0].grid(True, which="both")
    ax1[0].set_ylabel('Time [ns]')
    ax1[0].set_xlabel('Iterations')
    ax1[0].set_xlim(interval)
    ax1[0].set_ylim(timesus[interval[0]],timesus[interval[1]]+5)
    ax1[0].legend(labels, loc='upper left')

    ax1[1].grid(True, which="both")
    ax1[1].set_ylabel('Count')
    ax1[1].set_xlabel('Stepsize [ns]')
    ax1[1].set_yscale('log')
    ax1[1].legend(labels, loc='upper right')

    return fig1, ax1

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, ax = plot(dataus, datans, figsize=[7,2])
plt.subplots_adjust(wspace=0.4)
fig.savefig('globaltimer.pdf', format='pdf', bbox_inches='tight')
plt.show()
