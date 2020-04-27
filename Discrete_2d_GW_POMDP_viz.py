import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def pol2cart(rho, phi):
    """Return cartesian coords from polar coords.
    Utility fxn.
    """
    x = rho * np.cos(phi) 
    y = rho * np.sin(phi)
    return(x, y)

class visual():
    """Class for visualizing this experiment."""
    
    def __init__(self):
        # setup plot 
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        self.viz = ax


    def update(self, state):
        plt.cla()

        radius = 3

        self.viz.axis('square')
        self.viz.set_xlim((-3.5,3.5))
        self.viz.set_ylim((-3.5,3.5))
        
        # show FOV limits
        circle = plt.Circle((0,0), radius, color='k',fill=False)
        self.viz.add_artist(circle)

        # center point (rig)
        self.viz.plot((0), (0), 'o', color='k',MarkerSize=3)

        # target
        tx, ty = pol2cart(state.target_r, np.deg2rad(state.target_theta*10))
        self.viz.plot((tx), (ty), 'o', color='r',MarkerSize=3)
        
##        for k in range(particles.shape[1]):
##            d = np.deg2rad(particles[1,k])
##            r = particles[0,k]
##            x , y = pol2cart(r, d)
##            x += cent[0]
##            y += cent[1]
####            print(x,y)
##            self.viz.plot(x,y,'kx')
##        
        # create and plot camera FOV patch
        c = state.cam_theta*10
        low = (c - state.FOV_width*10/2) % 360
        high = (c + state.FOV_width*10/2) % 360
        w = Wedge((0,0), state.focal_pt+1, low, high, color='g',alpha=0.5)
        self.viz.add_artist(w)
##        
##        if incre != None:
####            plt.savefig('./videos/imgs_01/'+str(incre))
##            plt.pause(0.05)
##        else: 
        plt.pause(0.05)

    def close(self):
##        plt.close()
        pass
