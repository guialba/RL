import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np 

def plotEffects(g, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        ax.invert_yaxis()

    # Start Marker
    plt.scatter(0, 0, marker='o', s=1000, c='w', alpha=0.5, edgecolor='black',linewidth=3)
    for s, (x,y) in enumerate(g.S):
        rect = patches.Rectangle((x-.5, y-.5), 1, 1, fc=(0,0,0,0), linewidth=1, edgecolor='black', facecolor='w', zorder=2)
        ax.add_patch(rect)
        
        if (x, y) in g.effects:
            eff = g.effects[(x, y)]
            # Goals Markers
            if eff.get('terminal', False): 
                plt.scatter(x, y, marker='*', s=1000, c='w', alpha=0.5, edgecolor='black',linewidth=3)
            
            # Blockeds
            if eff.get('blocked', False): 
                rect = patches.Rectangle((x-.5, y-.5), 1, 1, fc=(0,0,0,1), linewidth=1, edgecolor='black', facecolor='black', zorder=2)
                ax.add_patch(rect)
    return ax

def plotStateValue(v, g, ax=None, show_text=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        ax.invert_yaxis()
    
    v = v.reshape([*g.size])
   
    ax.imshow(v)
    return ax

def plotActionStateValue(q, g, ax=None, show_text=False):    
    # q_Value to s_value
    v = np.array([sum(q[s]) for s,_ in enumerate(g.S)])
    return plotStateValue(v, g, ax, show_text)

def plotTrajectory(e, g, ax=None, show_text=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        ax.invert_yaxis()

    arrows = np.array([(0,1) for _ in g.S])
    arrows = arrows.reshape([g.size[0],g.size[1], 2])

    q = ax.quiver(range(g.size[0]), range(g.size[1]), arrows[:, :, 0], arrows[:, :, 1], pivot='mid', color=(0,0,0,0))
    n = {}
    for s, a in e:
        if a is not None:
            (xs, ys), (xa, ya) = g.S[s], g.A[a]
            count = sum(1 for s_, _ in e if s_ == s)
            n[s] = n.get(s,0)+1
            # n = (n_count+1, n_count)[n_count==(count-1)]
            angle = math.atan2(-ya, xa) * 180 / math.pi 
            text = f'{n[s]}' if show_text else ''
            ax.quiverkey(q, xs+xa*.1, ys+ya*.1, .5, label=text, angle=angle, color=(0,0,0,n[s]/count), coordinates='data')

    return ax


def plotPolicy(pi, g, ax=None, show_best_prob=False, show_best=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        ax.invert_yaxis()

    arrows = np.array([g.A[np.argmax(pi[s])]*np.array([1,-1]) for s,_ in enumerate(g.S)])
    arrows = arrows.reshape([g.size[0], g.size[1], 2])
    q = ax.quiver(range(g.size[0]), range(g.size[1]), arrows[:, :, 0], arrows[:, :, 1], pivot='mid', color=(1,0,0,0))

    for s, (x,y) in enumerate(g.S):
        for a, p_a in enumerate(pi[s]):
            best = int(np.argmax(pi[s]) == a)
            xa, ya = g.A[a]
            angle = math.atan2(-ya, xa) * 180 / math.pi 
            best_color = int(show_best and best)
            alpha = (p_a * 0.9 + 0.1, best)[best_color]
            text = ('', f'{round(p_a,2)}')[best and show_best_prob]
            ax.quiverkey(q, x+xa*.1, y+ya*.1, .5, label=text, angle=angle, color=(best_color,0,0,alpha), coordinates='data')

    return ax