#from klampt.model import trajectory
#
#milestones = [[0,0,0],[0.02,0,0],[1,0,0],[2,0,1],[2.2,0,1.5],[3,0,1],[4,0,-0.3]]
#
#
#traj = trajectory.Trajectory(milestones=milestones)
#from klampt import vis
#
#vis.add("point",[0,0,0])
#vis.animate("point",traj)
#vis.add("traj",traj)
#vis.spin(float('inf'))   #show the window until you close it#setting up steps for simulating 2Ddims = 2
#import numpy as np
#import matplotlib.pyplot as plt
#dims = 2
#step_n = 10
#step_set = [0, 1.e-3]
#origin = np.zeros((1,dims))# Simulate steps in 2D
#step_shape = (step_n,dims)
#steps = np.random.choice(a=step_set, size=step_shape)
#path = np.concatenate([origin, steps]).cumsum(0)
#start = path[:1]
#stop = path[-1:]# Plot the path
#fig = plt.figure(figsize=(8,8))
#ax = fig.add_subplot(111)
#ax.scatter(path[:,0], path[:,1],c='blue',alpha=0.25,s=0.05);
#ax.plot(path[:,0], path[:,1],c='blue',alpha=0.5,lw=0.25);
#ax.plot(start[:,0], start[:,1],c='red', marker='+')
#ax.plot(stop[:,0], stop[:,1],c='black', marker='o')
#plt.title('2D Random Walk')
#plt.tight_layout(pad=0)
#plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
np.random.seed(24)
def dist_points(point1, point2):
    return np.linalg.norm(point1-point2)

#def propose(current, L):
#    return  np.hstack((np.random.uniform(current[0]-L, current[0]+L),\
#                np.random.uniform(current[1]-L, current[1]+L)))

def propose(current, L):
    return  np.random.uniform(current-L, current+L)
           
    

def generate_path2D(start=np.array([0]),end=np.array([1]),L=1e-2,d=1e-4):
    path = []
    current = start
    path.append(current)
    current_dist = dist_points(current,end)
    while current_dist >= d:
        ghost_dist = 1000
        while ghost_dist >= current_dist:
        #ghost_dist = 100000 
        #while ghost_dist >= current_dist:
            ghost = propose(current,L)
            ghost_dist = dist_points(ghost, end)
            if ghost_dist < current_dist:
                current = ghost
                current_dist = ghost_dist
                path.append(current)
                break
        if current_dist <= d:
            break
    path.append(end)
    return np.vstack(path)

#for i in range(100):
#    origin = np.zeros(1)
#    rand_options= [np.random.normal(1,0.1), np.random.normal(-1,0.1)]
#    rand_point = np.random.choice(rand_options,1)
#    path = generate_path2D(origin,rand_point)
#    print(path)
#    paths = interpolate.interp1d(np.arange(0,len(path)),path,kind='cubic')
#    x = np.linspace(0,len(path)-1,100)
#    plt.plot(x, paths(x))
#    #plt.plot(path)

end_point = np.ones(1)*0.1
for i in range(100):
    origin = np.zeros(1)
    #rand_options= [np.random.normal(1,0.1), np.random.normal(-1,0.1)]
    #rand_point = np.random.choice(rand_options,1)
    path = generate_path2D(origin,end_point)
    paths = interpolate.interp1d(np.arange(0,len(path)),path.flatten(),kind='cubic')
    x = np.linspace(0,len(path)-1,100)
    plt.plot(x, paths(x))

#end_point = 0.4
#N = 1 
#path = []
#for i in range(N): 
#    for j in range(data.shape[0]):
#        path.append(generate_path2D(np.zeros(1),np.array(0.2))
#
#for i in range(N*data.shape[0]):
#    plt.plot(np.arange(0, path[i].shape[0]), path[i][:,0], label='ex')
#    plt.plot(np.arange(0, path[i].shape[0]), path[i][:,1], label='ey')
#plt.legend()
plt.show()

#curr = np.zeros(2)
#print(propose(curr, 1e-3))
