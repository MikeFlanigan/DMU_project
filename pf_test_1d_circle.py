import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

rigx = 0.5
rigy = 0.5
cent = (rigx, rigy)

radius = 0.5

cam_info = {"FOV center": 0, # degrees
            "FOV width": 20, # degrees
            }

discrits = 15
world = np.linspace(0,360-360/discrits,discrits)

N_particles = 100

def cart2pol(x, y):
    """Return polar coords from cartesian coords.
    Utility fxn.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    """Return cartesian coords from polar coords.
    Utility fxn.
    """
    x = rho * np.cos(phi) 
    y = rho * np.sin(phi)
    return(x, y)

def get_FOV_ends(cent, FOV):
    """Take center of FOV and FOV in degrees [0-360]
    and return low and high endpoints in degrees.
    """
    l = (cent - FOV) % 360
    h = (cent + FOV) % 360
    return [l,h]

class visual():
    """Class for visualizing this experiment."""
    
    def __init__(self):
        # setup plot 
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        self.viz = ax


    def update(self, cam, particles, target, incre = None):
        plt.cla()

        self.viz.axis('square')
        self.viz.set_xlim((-0.1,1.1))
        self.viz.set_ylim((-0.1,1.1))
        
        # FOV of the camera
        circle = plt.Circle(cent, radius, color='k',fill=False)
        self.viz.add_artist(circle)

        # center point (rig)
        self.viz.plot((rigx), (rigy), 'o', color='k',MarkerSize=2)

##        print(' ')
##        print(particles[0])
        for p in particles[0]:
            d = np.deg2rad(world[p] + np.random.randint(10))
            r = np.random.rand()*0.3 + 0.2
            x , y = pol2cart(r, d)
            x += cent[0]
            y += cent[1]
##            print(x,y)
            self.viz.plot(x,y,'kx')
        
        # create and plot camera FOV patch
        low, high = get_FOV_ends(cam["FOV center"],cam["FOV width"])
        w = Wedge(cent, radius, low, high, color='g',alpha=0.5)
        self.viz.add_artist(w)
        
        if incre != None:
##            plt.savefig('./videos/imgs_01/'+str(incre))
            plt.pause(0.05)
        else: 
            plt.pause(0.05)

def low_variance_resample(b, w):
    num_particles = b.shape[1]
    
    bnew = np.zeros(b.shape)
    r = np.random.rand()
    c = w[0]
    i = 0
    for m in range(num_particles):
        U = r + (m - 1)/num_particles
        while U > c:
            i += 1
            i = i % num_particles
            c += w[i]
        bnew[0,m] = b[0,i]
##        print(m,i)
    return bnew

def update_belief(b, cam_pos, observation):
    weights = [1/len(b[0])]*len(b[0])
    weights = np.asarray(weights)

    domain_resample = True
    if domain_resample:
        # reject the bottom z% likelihood particles and resample
        # from an analytical distribution. Assign mean weight for
        # these new particles
        # TODO: this is very inefficient computationally...
        cut = int(0.01*b.shape[1])
        if cut <= 0: cut = 1
    ##    print(cut)
        fresh_partics = np.random.randint(discrits, size=(1,cut))
    ##    print(b[0,np.where(np.argsort(weights)>=cut)].shape,fresh_partics.shape)
##        b = np.concatenate((b[0,np.where(np.argsort(weights)>=cut)],fresh_partics),axis = 1)
        
        # hm, just resampling z% of all particles seems to work well.
        # noteably this cuts down heavier on particle dense regions.
        # however since in this filtering case, particle dense regions
        # are actually an artifact, this turns out to be good and rational.
        b = np.concatenate((b[:,cut:],fresh_partics),axis = 1)
        
    i = 0
    for p in b[0]:
        # if contained generative model would update here
        # based on states like velocity

        # assign weight to every particle based on observation
        if cam_pos == p:
            if observation:
                weights[i] = 10*weights[i]
            else:
                weights[i] = 0.10*weights[i]
        else:
            pass # no observation on this particle

        i += 1 # increment index
        
    # normalize the weights to a valid probability distribution
    weights = weights/weights.sum()
    
    # sample a new set of particles and return that as the new belief
##    b_new = np.random.choice(b[0], replace=True, size=np.shape(b),p=weights)
    
    # use the low variance resampling algorithm from the Probabilistic Robotics Book
    b_new = low_variance_resample(b, weights)
    b_new = b_new.astype(int)
    return b_new
    
# initialize belief to a random distribution of particles
belief = np.random.randint(discrits,size=(1,N_particles)) 

viz = visual()
for i in range(250):
    # display what's happening
    plt.figure(1)
    viz.update(cam_info,belief,0, incre = i)
    
    plt.figure(2)
    plt.cla()
    plt.hist(belief[0],bins=range(discrits))
    plt.ylim([0,35])
    plt.pause(0.001)

    # put the continuous camera center into a discrete world bin
    cam_ind = np.where(abs(cam_info["FOV center"] - world) == min(abs(cam_info["FOV center"] - world)))[0][0]

    # update belief based on current state
    belief = update_belief(belief, cam_ind, False)
    
    cam_info["FOV center"] += 10
    cam_info["FOV center"] = cam_info["FOV center"] % 360
        
plt.close()



