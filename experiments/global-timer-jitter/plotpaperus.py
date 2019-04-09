from plotus import *

pltRange = (3800,3850)
fig1,ax1,fig2, ax2, blocks_sm = plotGeneralus(data)
ax1[0].set_xlim([pltRange[0], pltRange[1]])
ax1[0].set_ylim([data['times']['0'][pltRange[0]], data['times']['0'][pltRange[1]]])
ax1[1].set_xlim([pltRange[0], pltRange[1]])
#beQuiet = ax1[1].set_ylim(896,1192)
beQuiet = ax1[1].set_yticks(range(0,1192,160))

ax2[0].set_xlim([pltRange[0], pltRange[1]])
ax2[1].set_xlim([pltRange[0], pltRange[1]])
#beQuiet = ax2[0].set_ylim(-896,-1192)
#beQuiet = ax2[1].set_ylim(-896,-1192)
beQuiet = ax2[0].set_yticks(range(0,-1160,-160))
beQuiet = ax2[1].set_yticks(range(0,-1160,-160))
plt.show()
