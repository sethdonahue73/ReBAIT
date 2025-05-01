
from matplotlib import pyplot as plt
import numpy as np
import quaternion
from scipy import integrate
from sklearn.decomposition import PCA
from scipy import signal

from utils import plot_3


with open('data/llte_test2.npy','rb') as f:
    pelvis = np.load(f)
    thigh = np.load(f)
    shank = np.load(f)
    foot = np.load(f)
    contra = np.load(f)
    rt_fs = np.load(f)
    joint_ang = np.load(f)
    cal_time = np.load(f)

# plot_3(shank[:,0],shank[:,1:4])

q_ana = []
q_f_ = []
t=foot[:,0]-foot[0,0]
q_imu = quaternion.as_quat_array(foot[:,7:11])
a_world = quaternion.as_vector_part(q_imu * quaternion.from_vector_part(foot[:,18:21]) * q_imu.conj())
w_world = quaternion.as_vector_part(q_imu * quaternion.from_vector_part(foot[:,4:7]) * q_imu.conj())

g_avg = np.mean(a_world[t<5,:],axis=0)
g_ana = [0,9.81,0]

n = np.cross(g_avg,g_ana)
n = n/np.linalg.norm(n)
theta = np.arccos(np.dot(g_avg,g_ana)/(np.linalg.norm(g_avg)*np.linalg.norm(g_ana)))
q_g = quaternion.as_quat_array([np.cos(theta/2),np.sin(theta/2)*n[0],np.sin(theta/2)*n[1],np.sin(theta/2)*n[2]]).normalized()

w_world = quaternion.rotate_vectors(q_g,w_world)

pca = PCA(n_components=1)
pca.fit(w_world[t<cal_time-foot[0,0]])
z_pca = pca.components_[0]
rot_ax = np.cross(z_pca,[0,0,1])
theta = np.arccos(np.dot(z_pca,[0,0,1]))
q_pca = quaternion.as_quat_array([np.cos(theta/2),0,np.sin(theta/2)*np.sign(rot_ax[1]),0])

q_f = q_pca * q_g  # q_f is the rotation quaternion from imu global frame to global lab-fixed anatomical frame
q_init = q_imu[t<5].conj() * q_f.conj()  # equivalent to conj(q_f*q_imu); gives the quaternions from global lab-fixed anatomical frame to local imu frame at each timestep
q_0 = quaternion.from_float_array(np.mean(quaternion.as_float_array(q_init),axis=0))  # left mutliplication by q_0 gives orientation relative to starting orientation
q_f_.append(q_f)
q_ana.append(q_f * q_imu * q_0)  #  q_f * q_imu * q_0 gives the orientation quaternion between the starting orientation and the current orientation for any timestep of q_imu

w_ft = quaternion.as_vector_part(q_f * q_imu *quaternion.from_vector_part(foot[:,4:7])*q_imu.conj() * q_f.conj())[:,2]

cur_search = 1  # keep track of what event we are searching for. 1:initial contact, 2: mid-stance, 3:toe-off, 4: mid-swing
ic = []
mst = [0]
to = []
msw = []
temp = 0
n=len(foot)
for i in range(n):
# gait event detection - for post-processing, technically we can use i-1, i, i+1 since we can look forward in time (i+1); however to mimic real-time processing all time steps need to be current or past, so I have adjusted the below code to use i-2, i-1, i
# if we implement a 2nd order median filter as the paper with the gait event detection did, we will need to increase this delay by one to get a filtered value for the last time step
    if cur_search == 1:
        if w_ft[i] > 1.0:
            if w_ft[i] > w_ft[temp]:
                temp = i
        elif temp > 0:
            ic.append(temp)
            temp = 0
            cur_search = 2

    elif cur_search == 2:
        if foot[i,11] == 0:
            mst.append(i)
            cur_search = 3

    elif cur_search == 3:
        if w_ft[i] > 2:
            if w_ft[i] > w_ft[temp]:
                temp = i
        elif temp > 0:
            to.append(temp)
            temp = 0
            cur_search = 4

    elif cur_search == 4:
        if w_ft[i] < -0.5:
            if w_ft[i] < w_ft[temp]:
                temp = i
        elif temp > 0:
            msw.append(temp)
            temp = 0
            cur_search = 1

# plt.plot(t,w_ft)
# plt.scatter(t[ic],w_ft[ic],color='blue')
# plt.scatter(t[mst],w_ft[mst],color='green')
# plt.scatter(t[to],w_ft[to],color='yellow')
# plt.scatter(t[msw],w_ft[msw],color='red')
# plt.show()

a_ft = quaternion.as_vector_part(q_f * quaternion.from_vector_part(foot[:,1:4]) * q_f.conj())
filt = signal.butter(1,25,'lp',fs=120,output='sos')
# a_ft = signal.sosfilt(filt,a_ft)



mst.append(i)

acc = [[0,0,0]]
vel = [[0,0,0]]
pos = [[0,0,0]]
scat_ind = []

for i in range(len(mst)-1):
    time_samp = t[mst[i]:mst[i+1]]
    fa_samp = a_ft[mst[i]:mst[i+1]]
    int_time = time_samp[-1] - time_samp[0]

    samp_vel = integrate.cumulative_trapezoid(fa_samp,time_samp,initial=0,axis=0)

    v_slope = (samp_vel[-1]-samp_vel[0])/int_time
    v_drift = np.multiply(v_slope,(np.array(time_samp)-time_samp[0])[:,np.newaxis])
    samp_vel=samp_vel-v_drift

    if any(abs(samp_vel[0]) >= 1e5) or any(abs(samp_vel[-1]) >= 1e5):
        raise RuntimeError(f'Issue with drift correction. First and last values should be zero, but are {samp_vel[0]} and {samp_vel[-1]}')

    samp_pos = integrate.cumulative_trapezoid(samp_vel,time_samp,initial=0,axis=0)
    samp_pos += np.array(pos[-1])

    acc += list(fa_samp)
    vel += list(samp_vel)
    pos += list(samp_pos)
print(len(mst))

mst_filtered = mst[2:-3]


for i in range(len(mst_filtered)-1):
    plt.plot(t[mst_filtered[i]:mst_filtered[i+1]]-t[mst_filtered[i]],np.array(pos)[mst_filtered[i]:mst_filtered[i+1],0]-pos[mst_filtered[i]][0])

plt.show()
# plot_3(t,a_ft,scat_ind=to)

# plot_3(t,np.array(acc),scat_ind = mst)
# plot_3(t,np.array(vel),scat_ind = mst)
# plot_3(t,np.array(pos),scat_ind = mst)
    
# plot_3(t,np.array(foot[:,1:4]),ylim=(-50,150))
# plot_3(t,a_ft,ylim=(-50,150))
# plot_3(t,np.array(foot[:,4:7]),ylim=(-7.5,11))
# plot_3(t,quaternion.as_vector_part(q_f * q_imu *quaternion.from_vector_part(foot[:,4:7])*q_imu.conj() * q_f.conj()),ylim=(-7.5,11))
# plt.show()
