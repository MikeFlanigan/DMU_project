import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

rigx = 0.5
rigy = 0.5
cent = (rigx, rigy)

radius = 0.5

cam_info = {"FOV center": 0, # degrees
            "FOV width": 40, # degrees
            "Zoom": 0.0, # percent max
            "Focal min": 0.2,  # unitless for now
            "Focal max": 0.6,  # unitless for now
            "FOV min_width": 5, # degrees FOV at max zoom
            "FOV max_width": 90, # degrees FOV at min zoom
            }

# camera control in degrees CCW
ctrl = 10

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

def get_FOV_width(cam):
    m = (cam["FOV max_width"] - cam["FOV min_width"])
    width = -m*cam["Zoom"] + cam["FOV max_width"]
    return width

def get_focal_dist(cam):
    m = (cam["Focal max"] - cam["Focal min"])
    f_dist = m*cam["Zoom"] + cam["Focal min"]
    return f_dist

def get_FOV_ends(cent, FOV):
    """Take center of FOV and FOV in degrees [0-360]
    and return low and high endpoints in degrees.
    """
    l = (cent - FOV/2) % 360
    h = (cent + FOV/2) % 360
    return [l,h]

def in_FOV(p, cam):
    if (abs(p[1]-cam["FOV center"]) % 360) <= cam["FOV width"]/2 and p[0] <= get_focal_dist(cam):
##        print("In FOV!")
        return True
##    # check if the particle is within the focal distance of the camera
##    if p[0] <= get_focal_dist(cam):
##        # check if the particle is within the pan FOV of the camera
##        if ((abs(p[1]-cam["FOV pan"]) % 360) <= cam["FOV width"]/2):
##            # check if the particle is within the tilt FOV of the camera
##            if ((abs(p[2]-cam["FOV pan"]) % 360) <= cam["FOV width"]/2):
##                return True
    # if not in the FOV then:
    return False



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

        for k in range(particles.shape[1]):
            d = np.deg2rad(particles[1,k])
            r = particles[0,k]
            x , y = pol2cart(r, d)
            x += cent[0]
            y += cent[1]
##            print(x,y)
            self.viz.plot(x,y,'kx')
        
        # create and plot camera FOV patch
        low, high = get_FOV_ends(cam["FOV center"],cam["FOV width"])
        w = Wedge(cent, get_focal_dist(cam), low, high, color='g',alpha=0.5)
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
        noise = np.asarray([np.random.rand()*radius/50-(radius/50)/2,np.random.randint(-1,2)])
        bnew[:,m] = b[:,i] + noise
        bnew[0,m] = np.clip(bnew[0,m],0.01*radius,radius) # clipping the particle distances
##        print(m,i)
    return bnew

def domain_resample(b, percent):
    """Reject a small % of particles and replace them from a known distribution."""
    cut = int(percent*b.shape[1])
    if cut <= 0: cut = 1
    fresh_degs = np.random.rand(1,cut)*360
    fresh_radis = np.random.rand(1,cut)*radius
    fresh_partics = np.concatenate((fresh_radis,fresh_degs),0)
    
    # hm, just resampling z% of all particles seems to work well.
    # noteably this cuts down heaviest on particle dense regions.
    # however since in this filtering case, particle dense regions
    # are actually an artifact, this turns out to be good and rational.
    b = np.concatenate((b[:,cut:],fresh_partics),axis = 1)
    return b


def update_belief(b, cam, observation):
    weights = [1/len(b[0])]*len(b[0])
    weights = np.asarray(weights)

    b = domain_resample(b,0.01)
        
    i = 0
    for k in range(b.shape[1]):
        # if contained generative model would update here
        # based on states like velocity

        # TODO: write actual camera observation likelihoods based on zoom percents
        # assign weight to every particle based on observation
        if in_FOV(b[:,k], cam):
            if observation:
                weights[i] = 10*weights[i]
            else:
                weights[i] = 0.05*weights[i]
        else:
            pass # no observation on this particle

        i += 1 # increment index
        
    # normalize the weights to a valid probability distribution
    weights = weights/weights.sum()
    
    # use the low variance resampling algorithm from the Probabilistic Robotics Book
    b_new = low_variance_resample(b, weights)

    # TODO: maybe add a flag condition, if variance ever does get super low, resample
    # uniformly?
    
    return b_new
    
# initialize belief to a random distribution of particles
degs = np.random.rand(1,N_particles)*360
radis = np.random.rand(1,N_particles)*radius
belief = np.concatenate((radis,degs),0) # a belief is now a radius and a distance

viz = visual()
for i in range(550):

    # camera control
    cam_info["FOV center"] += ctrl
    cam_info["FOV center"] = cam_info["FOV center"] % 360
    cam_info["FOV width"] = get_FOV_width(cam_info)
    
    # update belief based on current state
    belief = update_belief(belief, cam_info, False)

    cam_info["Zoom"] = 0.4
##    s = input("command: ")
##    if s == '':
##        pass
##    elif s == 'a':
##        ctrl = -10
##    elif s == 's':
##        ctrl = 0
##    elif s == 'd':
##        ctrl = 10
##    elif s == 'r':
##        cam_info["Zoom"] += 0.10
##    elif s == 'f':
##        cam_info["Zoom"] -= 0.10

##    print(cam_info["Zoom"],cam_info["FOV width"])

    # display what's happening
    plt.figure(1)
    viz.update(cam_info,belief,0, incre = i)
        
plt.close()



