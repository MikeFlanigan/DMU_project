import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp
import math


rigx = 0.5
rigy = 0.5
rigz = 0.0
cent = (rigx, rigy, rigz)

radius = 0.5

cam_info = {"FOV pan": 0, # degrees
            "FOV tilt": 0, # degrees (off of horizon, positive up)
            "FOV width": 40, # degrees
            "FOV height": 30, # degrees
            "Zoom": 0.2, # percent max
            "Focal min": 0.2,  # unitless for now
            "Focal max": 0.6,  # unitless for now
            "FOV min_width": 5, # degrees FOV at max zoom
            "FOV max_width": 90, # degrees FOV at min zoom
            "FOV min_height": 4, # degrees FOV at max zoom
            "FOV max_height": 90*4/5, # degrees FOV at min zoom
            }

# hard core testing
cam_info["FOV min_width"] = 120
cam_info["FOV max_width"] = 120
cam_info["FOV min_height"] = 120
cam_info["FOV max_height"] = 120
cam_info["Focal max"] = 0.8
cam_info["Zoom"] = 1.0



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

def polar2cart(r, theta, phi):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(90-phi)
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return [x, y, z]

def sphere2cart(r, theta, phi):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(90-phi)
    
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return [x, y, z]

def get_FOV_width(cam):
    m = (cam["FOV max_width"] - cam["FOV min_width"])
    width = -m*cam["Zoom"] + cam["FOV max_width"]
    return width

def get_FOV_height(cam):
    m = (cam["FOV max_height"] - cam["FOV min_height"])
    height = -m*cam["Zoom"] + cam["FOV max_height"]
    return height

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

def get_verts(r,theta,phi,cam):
    verts = [] # list of 3x3 patches of vertices
    rt = r # approximation for now
    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 = polar2cart(r, theta+cam["FOV width"]/2, phi+cam["FOV height"]/2)
    x3, y3, z3 = polar2cart(r, theta-cam["FOV width"]/2, phi+cam["FOV height"]/2)
    x4, y4, z4 = polar2cart(r, theta+cam["FOV width"]/2, phi-cam["FOV height"]/2)
    x5, y5, z5 = polar2cart(r, theta-cam["FOV width"]/2, phi-cam["FOV height"]/2)
    
    verts.append(np.asarray([[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]])) # top
    verts.append(np.asarray([[x1,y1,z1],[x3,y3,z3],[x5,y5,z5]])) # right 
    verts.append(np.asarray([[x1,y1,z1],[x5,y5,z5],[x4,y4,z4]])) # bottom
    verts.append(np.asarray([[x1,y1,z1],[x4,y4,z4],[x2,y2,z2]])) # left 
    verts.append(np.asarray([[x2,y2,z2],[x3,y3,z3],[x5,y5,z5],[x4,y4,z4]])) # face

    # apply rig translation
    for v in verts:
        v[:,0] += cent[0] # x
        v[:,1] += cent[1] # y
        v[:,2] += cent[2] # z
    return verts

def in_FOV(p, cam):
    # check if the particle is within the focal distance of the camera
    if p[0] <= get_focal_dist(cam):
        # check if the particle is within the pan FOV of the camera
        if ((abs(p[1]-cam["FOV pan"]) % 360) <= cam["FOV width"]/2):
            # check if the particle is within the tilt FOV of the camera
            if ((abs(p[2]-cam["FOV tilt"]) % 360) <= cam["FOV height"]/2):
                return True
    # if not in the FOV then:
    return False

class visual():
    """Class for visualizing this experiment."""
    
    def __init__(self):
        # setup plot 
        ax = a3.Axes3D(pl.figure())

        # set axis planes to be white for visibility
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        
        self.viz = ax


    def update(self, cam, particles, target, incre = None):
        pl.cla()

        self.viz.set_xlim([-0.1,1.1])
        self.viz.set_ylim([-0.1,1.1])
        self.viz.set_zlim([0,1.5])

        self.viz.set_xlabel('X')
        self.viz.set_ylabel('Y')
        self.viz.set_zlabel('Z')
        
        # center point (rig)
        self.viz.plot([cent[0]],[cent[1]],[cent[2]],'ro')

        for k in range(particles.shape[1]):
            r = particles[0,k]
            th = particles[1,k]
            phi = particles[2,k]
            
            x , y, z = sphere2cart(r, th, phi)
            x += cent[0]
            y += cent[1]
            if in_FOV(particles[:,k], cam):
                self.viz.plot([x],[y],[z],'rx')
            else:
                self.viz.plot([x],[y],[z],'kx')
        
        # create and plot camera FOV patches
        vtxs = get_verts(get_focal_dist(cam), cam["FOV pan"], cam["FOV tilt"], cam)
        for v in vtxs:
            tri = a3.art3d.Poly3DCollection([v])
            tri.set_alpha(0.2)
            tri.set_facecolor('g')
            tri.set_edgecolor('k')
            self.viz.add_collection3d(tri)
    
        if incre != None:
##            plt.savefig('./videos/imgs_01/'+str(incre))
            pl.pause(0.05)
        else: 
            pl.pause(0.05)

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
        noise = np.asarray([np.random.rand()*radius/50-(radius/50)/2,
                            np.random.randint(-1,2),
                            np.random.randint(-1,2)])
        bnew[:,m] = b[:,i] + noise
        bnew[0,m] = np.clip(bnew[0,m],0.01*radius,radius) # clipping the particle distances
##        print(m,i)
    return bnew

def domain_resample(b, percent):
    """Reject a small % of particles and replace them from a known distribution."""
    cut = int(percent*b.shape[1])
    if cut <= 0: cut = 1
    fresh_radis = np.random.rand(1,cut)*radius
    fresh_thetas = np.random.rand(1,cut)*360
    fresh_phis = np.random.rand(1,cut)*90
    fresh_partics = np.concatenate((fresh_radis,fresh_thetas),0)
    fresh_partics = np.concatenate((fresh_partics,fresh_phis),0)
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
        # TODO: add tilt component to this detection check
        # check pan, check radius, check tilt
        if in_FOV(b[:,k], cam):
            if observation:
                weights[i] = 10*weights[i]
            else:
                weights[i] = 0.005*weights[i]
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
radis = np.random.rand(1,N_particles)*radius
thetas = np.random.rand(1,N_particles)*360
phis = np.random.rand(1,N_particles)*90
belief = np.concatenate((radis,thetas),0) 
belief = np.concatenate((belief,phis),0) # belief is now 3d state, r, theta, phi

viz = visual()
for i in range(550):

    # camera control
    # rand control for testing
    if i % 2 == 0:
        cam_info["FOV pan"] = np.random.randint(0,360)
        cam_info["FOV tilt"] = np.random.randint(0,90)
        cam_info["Zoom"] = np.random.rand()
        
##    cam_info["FOV pan"] += ctrl
##    cam_info["FOV pan"] = cam_info["FOV pan"] % 360
    cam_info["FOV width"] = get_FOV_width(cam_info)
    cam_info["FOV height"] = get_FOV_height(cam_info)
    
    # update belief based on current state
    belief = update_belief(belief, cam_info, False)

##    cam_info["Zoom"] = 0.2

    # display what's happening
    plt.figure(1)
    viz.update(cam_info, belief, 0, incre = i)
        
plt.close()



