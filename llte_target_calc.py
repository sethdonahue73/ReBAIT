from matplotlib import pyplot as plt
import numpy as np
import quaternion
from mtw_postproc_gait_analysis import replay_func
from new_utils import my_resample


replay_file = 'data/080_sk_1.nc'
# llte_target = 'targets/archive/080_wn.pkl'
llte_target = 'targets/target0.pkl'
dc = replay_func(replay_file,llte_target)
plt.close('all')

t = dc.time[dc.devIds[2]]

xtarget = []
ytarget = []
ttarget = []
if len(dc.fs) >= 7:
    pos = np.array(dc.pos[dc.devIds[2]])
    leg_theta_anatomical = []
    samp_theta = quaternion.as_float_array(dc.q_cal[dc.devIds[2]][0] * quaternion.from_float_array(dc.qutSer[dc.devIds[2]]) * dc.q_cal[dc.devIds[2]][1])
    for quat in samp_theta:
        w,x,y,z = quat
        shank_ang = np.arctan2(2.0* ( w*z + x*y), 1.0-2.0*(y**2+z**2))
        leg_theta_anatomical.append(shank_ang)
    for i in range(0,len(dc.fo)-7): # remove first and last several gait cycles from plot; typically very noisy as subject is speeding up and slowing down
        if dc.fs[i+1]-dc.fs[i] > 130:
            print(dc.fs[i+1]-dc.fs[i])
            continue
        y_x = -np.array(pos[dc.fs[i]:dc.fo[i],0])
        y_x += dc.leg_length*np.sin(leg_theta_anatomical[dc.fs[i]]) - y_x[0]
        y_x *= dc.leg_length/(dc.leg_length-dc.imu2knee)
        y_y=np.array(pos[dc.fs[i]:dc.fo[i],1])
        y_t=np.array(leg_theta_anatomical[dc.fs[i]:dc.fo[i]])*180/np.pi
        x=np.array(t[dc.fs[i]:dc.fo[i]])
        x=x-x[0]
        y_x = my_resample(101,y_x,x)
        y_y = my_resample(101,y_y,x)
        y_t = my_resample(101,y_t,x)
        xtarget.append(y_x)
        ytarget.append(y_y)
        ttarget.append(y_t)
        
xtarget = np.average(xtarget,axis=0)/dc.leg_length
ytarget = np.average(ytarget,axis=0)/dc.leg_length
ttarget = np.average(ttarget,axis=0)/(np.arctan(dc.leg_length/dc.foot_length)*180/np.pi)
llte_tar = np.vstack([xtarget,ytarget,ttarget])

with open(llte_target, 'rb') as file:
    llte_targets = np.load(file)
x=range(101)
plt.figure()
plt.subplot(211)
plt.plot(x, xtarget)
plt.plot(x, ytarget)
plt.plot(x, ttarget)
plt.subplot(212)
plt.plot(x,np.array(llte_targets).T)
plt.show()
with open('targets/archive/sk_targets.pkl', 'wb') as f:
    np.save(f, llte_tar)