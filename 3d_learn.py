import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp
import math
import numpy as np

cam_info = {"FOV pan": 0, # degrees
            "FOV tilt": 0, # degrees (off of horizon, positive up)
            "FOV width": 40, # degrees
            "FOV height": 30, # degrees
            "Zoom": 0.0, # percent max
            "Focal min": 0.2,  # unitless for now
            "Focal max": 0.6,  # unitless for now
            "FOV min_width": 5, # degrees FOV at max zoom
            "FOV max_width": 90, # degrees FOV at min zoom
            "FOV min_height": 4, # degrees FOV at max zoom
            "FOV max_height": 90*4/5, # degrees FOV at min zoom
            }


def get_focal_dist(cam):
    m = (cam["Focal max"] - cam["Focal min"])
    f_dist = m*cam["Zoom"] + cam["Focal min"]
    return f_dist

def polar2cart(r, theta, phi):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(90-phi)
    
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    return [x, y, z]

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
    return verts

vtx = get_verts(get_focal_dist(cam_info), cam_info["FOV pan"], cam_info["FOV tilt"], cam_info)
ax = a3.Axes3D(pl.figure())
for v in vtx:
    tri = a3.art3d.Poly3DCollection([v])
    tri.set_alpha(0.2)
    tri.set_facecolor('g')
##    tri.set_color(colors.rgb2hex(sp.rand(3)))
    tri.set_edgecolor('k')
    ax.add_collection3d(tri)

# plot belief particles
ax.plot([2],[2],[2],'kx')

# plot hemisphere of range
##ax.plot_wireframe(X, Y, Z, color='black')
# set the hemisphere lines to semi-transparent

# set axis planes to be white for visibility
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# set axis limits
ax.set_xlim([-2,3])
ax.set_ylim([-2,3])
ax.set_zlim([0,3])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

pl.show()

##pl.pause(1)
##pl.cla()
##pl.pause(1)

