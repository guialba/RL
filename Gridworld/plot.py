import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np 

def plotTrajectory(e, g, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        ax.invert_yaxis()

    e = [(g.S[s], g.ACTIONS[a]) for s, a in e]

    arrows = np.array([(0,1) for _ in g.S])
    arrows = arrows.reshape([g.size[0],g.size[1], 2])

    x = ax.quiver(range(g.size[0]), range(g.size[1]), arrows[:, :, 0], arrows[:, :, 1], pivot='mid', color=(0,0,0,0))
    for s, a in e: 
        angle = math.atan2(-a[1], a[0]) * 180 / math.pi 
        ax.quiverkey(x, s[0], s[1], 1, label=f'', angle=angle, color=(0,0,0,1), coordinates='data')
    for x in range(g.size[0]):
        for y in range(g.size[1]):
            rect = patches.Rectangle((x-.5, y-.5), 1, 1, fc=(0,0,0,0), linewidth=1, edgecolor='black', facecolor='w', zorder=2)
            ax.add_patch(rect)

    return ax


def plotPolicy(pi, g, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        ax.invert_yaxis()

    arrows = np.array([g.ACTIONS[np.argmax(pi[x,y])]*np.array([1,-1]) for x,y in g.S])
    arrows = arrows.reshape([g.size[0],g.size[1], 2])

    ax.quiver(range(g.size[0]), range(g.size[1]), arrows[:, :, 0], arrows[:, :, 1], pivot='mid')

    return ax