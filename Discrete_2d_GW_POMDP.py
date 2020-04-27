import random
import math
import copy
from Discrete_2d_GW_POMDP_viz import visual

class GW_State:
    """State representation for 2D version
    of the camera search problem.
    """
    
    def __init__(self):
        # camera parameters 
        self.cam_theta = random.randint(1,36) # 10 degree discritized bins
        self.zoom = random.randint(1,3) # 3 zoom bins, 1 = zoomed out, 3 = zoomed in
        self.focal_pt, self.FOV_width = self.zoom_to_FOV()

        # target parameters
        self.target_theta = random.randint(1,36)
        self.target_r = random.randint(1,3)
        self.theta_dot = 0 # better for noise to update the velocity terms
        self.r_dot = 0 # better for noise to update the velocity terms

    def zoom_to_FOV(self):
        """Convert zoom to FOV width and focal distance.
        """
        # for now having it on a simple 1-1 relationship
        # farther zoomed in, the narrower the view
        return self.zoom, 4-self.zoom

# Define the action space
# maintaining single action choices per time step for now
GW_Actions = {
    "1":"Left",
    "2":"Stay",
    "3":"Right",
    "4":"ZoomIn",
    "5":"NoZoom",
    "6":"ZoomOut",
    "7":"Track",
    }

# Should return transition probabilities rather than sp?
def GW_Transitions(s, a):
    sp = copy.deepcopy(s) # init return state

    # camera transitions
    # need to be a function of zoom to account for speed decrease
    # at deep zoom. write up include pix / meter / second at zooms
    if a == "Left":
        if s.zoom <= 2:
            sp.cam_theta += 2
        elif s.zoom > 2:
            sp.cam_theta += 1
    elif a == "Stay": pass
    elif a == "Right":
        if s.zoom <= 2:
            sp.cam_theta -= 2
        elif s.zoom > 2:
            sp.cam_theta -= 1

    # handle wrap around
    if sp.cam_theta > 36:
        sp.cam_theta = 1
    if sp.cam_theta < 1:
        sp.cam_theta = 36

    if a == "ZoomIn":
        sp.zoom += 1
    elif a == "NoZoom": pass
    elif a == "ZoomOut":
        sp.zoom -= 1

    sp.zoom = clamp(sp.zoom, 1, 3)

    sp.focal_pt, sp.FOV_width = sp.zoom_to_FOV()

    # target transitions
##    sp.theta_dot += random.random() - 0.5
##    sp.theta_dot = clamp(sp.theta_dot, -1.9, 1.9)
##    sp.r_dot += random.random() - 0.5
##    sp.r_dot = clamp(sp.r_dot, -1, 1)
    sp.theta_dot = 0
    sp.r_dot = 0
    
    sp.target_theta += int(sp.theta_dot)
    sp.target_theta = sp.target_theta % 36
    sp.target_r += int(sp.r_dot)
    sp.target_r = clamp(sp.target_r, 1,3)

    return sp


# should return observation probabilities rather than O?
def GW_Observations(s):
    # if in FOV
    # TODO: not handling wrap around
    if abs(s.cam_theta - s.target_theta) <= math.ceil(s.FOV_width/2):
        # observation based on range
        if s.target_r == s.focal_pt: # in FOV and in focal range
            if random.random() < 0.9: # 90% TP
                return True
            else:
                return False
        else: # in FOV and not at correct focal range
            if random.random() < 0.4: # 40% TP
                return True
            else:
                return False
    else: # if not in FOV
        if random.random() < 0.05: # 5% FP 
            return True
        else:
            return False


def GW_Rewards(s, a):
    if a == "Track":
        # if in FOV
        # TODO: not handling wrap around
        if abs(s.cam_theta - s.target_theta) <= math.ceil(s.FOV_width/2):
            return [10.0, True]
        else:
            return [-1.0, False]
    else:
        return [0.0, False]
    
def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def step_update(s, policy):

    # observe environment
    obs = GW_Observations(s)
    
    # choose action with policy
    act = policy.choose_action(s, obs)

    # interact with environment
    s = GW_Transitions(s,act) # sp

    # observe environment
    obs = GW_Observations(s)
    
    # collect rewards
    r, done = GW_Rewards(s,act)
    
    return obs, act, s, r, done


class rand_policy:
    def choose_action(self, s, obs):
        act = random.randint(1,len(GW_Actions))
        act = GW_Actions[str(act)]
        return act


class simple_policy:
    def choose_action(self, s, obs):
        if obs == True:
            act = "Track"
        elif random.random() < 0.8:
            act = "Left"
        else:
            act = "ZoomOut"
        return act


visualize = True
if visualize:
    viz = visual()
    
p = rand_policy()
p = simple_policy()

gamma = 0.99
episodes = 1
ep_rewards = 0
for ep in range(episodes):
    s = GW_State() # initialize
    # might need an initial observation here

    r_tot = 0
    for t in range(200):
        obs, a, s, r, done = step_update(s,p)
        r_tot = r_tot + gamma**t*r
        print(obs, a, r, done, s.target_r, s.focal_pt,s.target_theta,s.cam_theta)

        if visualize:
            viz.update(s)
            
        if done:
            print('Found target, r_tot:',r_tot)
            viz.close()
            break

    ep_rewards += r_tot

print('average reward: ',ep_rewards/episodes)
    
    

