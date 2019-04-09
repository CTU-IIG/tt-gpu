from plotns import *

pltRange = (1465,1480)
fig1,ax1,fig2, ax2, blocks_sm = plotGeneralns(data)
ax1[0].set_xlim([pltRange[0], pltRange[1]])
ax1[0].set_ylim([data['times']['0'][pltRange[0]], data['times']['0'][pltRange[1]]])
ax1[1].set_xlim([pltRange[0], pltRange[1]])
beQuiet = ax1[1].set_yticks(range(0,192,32))

ax2[0].set_xlim([pltRange[0], pltRange[1]])
ax2[1].set_xlim([pltRange[0], pltRange[1]])
beQuiet = ax2[0].set_yticks(range(0,-160,-32))
beQuiet = ax2[1].set_yticks(range(0,-160,-32))
plt.show()
