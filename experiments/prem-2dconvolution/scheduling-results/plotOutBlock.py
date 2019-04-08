from showBlocks import drawScenario
import matplotlib.pyplot as plt

titles = [
            "sched-0pfo-3000wbo.json",  
            "sched-500pfo-3000wbo.json",
            "sched-1000pfo-3000wbo.json",
            "sched-1500pfo-3000wbo.json",
            "sched-2000pfo-3000wbo.json",
            "sched-2500pfo-3000wbo.json",
            "sched-3000pfo-3000wbo.json",
]

filenames= [
            "data/512t-2b-4k-1024-prem-kernelsched-0pfo-3000wbo.json",  
            "data/512t-2b-4k-1024-prem-kernelsched-500pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-1000pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-1500pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-2000pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-2500pfo-3000wbo.json",
            "data/512t-2b-4k-1024-prem-kernelsched-3000pfo-3000wbo.json",
]

for filename, title in zip(filenames, titles):
    drawScenario(filename, title)
plt.show()
