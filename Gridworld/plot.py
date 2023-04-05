import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np 

# def plotTrajectory(e, g, ax=None):
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.gca()
#         ax.invert_yaxis()

#     e = [(g.S[s], g.ACTIONS[a]) for s, a in e]

#     arrows = np.array([(0,1) for _ in g.S])
#     arrows = arrows.reshape([g.size[0],g.size[1], 2])

#     x = ax.quiver(range(g.size[0]), range(g.size[1]), arrows[:, :, 0], arrows[:, :, 1], pivot='mid', color=(0,0,0,0))
#     for s, a in e: 
#         angle = math.atan2(-a[1], a[0]) * 180 / math.pi 
#         ax.quiverkey(x, s[0], s[1], 1, label=f'', angle=angle, color=(0,0,0,1), coordinates='data')
#     for x in range(g.size[0]):
#         for y in range(g.size[1]):
#             color = 'b' if g.isBlocked(x,y) else 'w'
#             rect = patches.Rectangle((x-.5, y-.5), 1, 1, fc=(0,0,0,int(g.isBlocked(x,y))), linewidth=1, edgecolor='black', facecolor=color, zorder=2)
#             ax.add_patch(rect)

#     return ax

def plotTrajectory(e, g, ax=None, show_text=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        ax.invert_yaxis()

    arrows = np.array([(0,1) for _ in g.S])
    arrows = arrows.reshape([g.size[0],g.size[1], 2])

    q = ax.quiver(range(g.size[0]), range(g.size[1]), arrows[:, :, 0], arrows[:, :, 1], pivot='mid', color=(0,0,0,0))
    n = {}
    for i, ((xs,ys), (xa,ya)) in enumerate(e):
        count = sum(1 for s, _ in e if s == (xs,ys))
        n[(xs,ys)] = n.get((xs,ys),0)+1
        # n = (n_count+1, n_count)[n_count==(count-1)]
        angle = math.atan2(-ya, xa) * 180 / math.pi 
        text = f'{n[(xs,ys)]}' if show_text else ''
        ax.quiverkey(q, xs+xa*.1, ys+ya*.1, .5, label=text, angle=angle, color=(0,0,0,n[(xs,ys)]/count), coordinates='data')
    
    for x in range(g.size[0]):
        for y in range(g.size[1]):
            color = 'b' if g.isBlocked(x,y) else 'w'
            rect = patches.Rectangle((x-.5, y-.5), 1, 1, fc=(0,0,0,int(g.isBlocked(x,y))), linewidth=1, edgecolor='black', facecolor=color, zorder=2)
            ax.add_patch(rect)

    return ax


def plotPolicy(pi, g, ax=None, show_best_prob=False, show_best=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        ax.invert_yaxis()

    arrows = np.array([g.ACTIONS[np.argmax(pi[x,y])]*np.array([1,-1]) for x,y in g.S])
    arrows = arrows.reshape([g.size[0],g.size[1], 2])
    q = ax.quiver(range(g.size[0]), range(g.size[1]), arrows[:, :, 0], arrows[:, :, 1], pivot='mid', color=(1,0,0,0))

    for x,y in g.S:
        for a, p_a in enumerate(pi[x,y]):
            best = int(np.argmax(pi[x,y]) == a)
            xa, ya = g.ACTIONS[a]
            angle = math.atan2(-ya, xa) * 180 / math.pi 
            best_color = int(show_best and best)
            alpha = (p_a * 0.9 + 0.1, best)[best_color]
            text = ('', f'{round(p_a,2)}')[best and show_best_prob]
            ax.quiverkey(q, x+xa*.1, y+ya*.1, .5, label=text, angle=angle, color=(best_color,0,0,alpha), coordinates='data')
    
    for x in range(g.size[0]):
        for y in range(g.size[1]):
            color = 'b' if g.isBlocked(x,y) else 'w'
            rect = patches.Rectangle((x-.5, y-.5), 1, 1, fc=(0,0,0,int(g.isBlocked(x,y))), linewidth=1, edgecolor='black', facecolor=color, zorder=2)
            ax.add_patch(rect)
    return ax