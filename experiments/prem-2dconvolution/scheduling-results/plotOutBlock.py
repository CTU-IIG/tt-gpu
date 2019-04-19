from showBlocks import drawScenario
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

titles = [
#            "sched-0pfo-3000wbo.json",  
#            "sched-500pfo-3000wbo.json",
#            "sched-1000pfo-3000wbo.json",
#            "sched-1500pfo-3000wbo.json",
#            "sched-2000pfo-3000wbo.json",
#            "sched-2500pfo-3000wbo.json",
#            "sched-3000pfo-3000wbo.json",
             "Kernel wise tile-scheduling with an offset of $1.4\mu s$",
]

filenames= [
#            "data/512t-2b-4k-1024-prem-kernelsched-0pfo-3000wbo.json",  
#            "data/512t-2b-4k-1024-prem-kernelsched-500pfo-3000wbo.json",
#            "data/512t-2b-4k-1024-prem-kernelsched-1000pfo-3000wbo.json",
#            "data/512t-2b-4k-1024-prem-kernelsched-1500pfo-3000wbo.json",
#            "data/512t-2b-4k-1024-prem-kernelsched-2000pfo-3000wbo.json",
#            "data/512t-2b-4k-1024-prem-kernelsched-2500pfo-3000wbo.json",
#            "data/512t-2b-4k-1024-prem-kernelsched-3000pfo-3000wbo.json",
             "data/512t-2b-4k-1024-prem-kernelsched-1400pfo-prof.json",
]

for filename, title in zip(filenames, titles):
    fig = drawScenario(filename, title)
    fig.savefig('schedblocks.pdf', format='pdf')
#plt.show()
