# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:56:36 2024

@author: srd8236
"""

# New Utils Files...Simplified 
import numpy as np
import quaternion
import pandas as pd
import pickle
import tkinter as tk

from tkinter import filedialog
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from scipy import integrate, signal

def select_Calibration_Region(data):
    """
    Function to interactively select a calibration region from a plot.
    
    Parameters:
    - w: 2D numpy array, shape (n_samples, n_features)
        ANGULAR VELOCITY Time series data to be plotted.
    
    Returns:
    - x: 1D numpy array
        Selected indices representing the calibration region.
    """
    
    t=data[:,0]-data[0,0]
    q_imu = quaternion.as_quat_array(data[:,7:11])
    a_world = quaternion.as_vector_part(q_imu * quaternion.from_vector_part(data[:,18:21]) * q_imu.conj())
    w_world = quaternion.as_vector_part(q_imu * quaternion.from_vector_part(data[:,4:7]) * q_imu.conj())

    g_avg = np.mean(a_world[t<5,:],axis=0)
    g_ana = [0,9.81,0]

    n = np.cross(g_avg,g_ana)
    n = n/np.linalg.norm(n)
    theta = np.arccos(np.dot(g_avg,g_ana)/(np.linalg.norm(g_avg)*np.linalg.norm(g_ana)))
    q_g = quaternion.as_quat_array([np.cos(theta/2),np.sin(theta/2)*n[0],np.sin(theta/2)*n[1],np.sin(theta/2)*n[2]]).normalized()
    
    w_world = quaternion.rotate_vectors(q_g,w_world)
    
    # Generate time indices
    t1 = np.arange(0, len(w_world), 1)

    # Create a plot for the time series data
    plt.figure(figsize=(10, 15))
    plt.plot(t1, w_world[:, 0])

    # Use ginput to interactively select points on the plot
    x = plt.ginput(3)
    # First Click is to zoom in 
    # Close all open plots
    plt.close('all')

    # Round the selected x-coordinates and convert to integer
    x = np.round(x, 0)
    x = x[1:, 0].astype(np.int64)

    return x

def ana_Calibration(segment_list,  cal_time, idx_pelvis = None, idx_lower = None):
    # Idx variables are used from 
    q_ana = []
    q_f_ = []
    fa_ana = []
    omega_world = []

    for data in segment_list:
        # if not q_ana:
        #     idx = idx_pelvis
        # else:
        #     idx = idx_lower
            
        t=data[:,0]-data[0,0]
        q_imu = quaternion.as_quat_array(data[:,7:11])
        a_world = quaternion.as_vector_part(q_imu * quaternion.from_vector_part(data[:,18:21]) * q_imu.conj())
        w_world = quaternion.as_vector_part(q_imu * quaternion.from_vector_part(data[:,4:7]) * q_imu.conj())

        g_avg = np.mean(a_world[t<5,:],axis=0)
        # g_avg = np.array([0,0,9.81])
        g_ana = [0,9.81,0]

        
        pca = PCA(n_components=1)
        pca.fit(a_world[t<cal_time-data[0,0]])
        x_pca = pca.components_[0]
        x_pca = x_pca - np.dot(x_pca,g_avg)/np.dot(g_avg,g_avg)*g_avg
        pca = PCA(n_components=1)
        pca.fit(w_world[t<cal_time-data[0,0]])
        z_pca = pca.components_[0]
        z_pca = z_pca - np.dot(z_pca,g_avg)/np.dot(g_avg,g_avg)*g_avg

        # print(g_avg/np.linalg.norm(g_avg),x_pca/np.linalg.norm(x_pca),z_pca/np.linalg.norm(z_pca))
        # print(np.degrees(np.arccos(np.dot(x_pca,z_pca)/(np.linalg.norm(x_pca)*np.linalg.norm(z_pca)))))

        n = np.cross(g_avg,g_ana)
        n = n/np.linalg.norm(n)
        theta = np.arccos(np.dot(g_avg,g_ana)/(np.linalg.norm(g_avg)*np.linalg.norm(g_ana)))
        q_g = quaternion.as_quat_array([np.cos(theta/2),np.sin(theta/2)*n[0],np.sin(theta/2)*n[1],np.sin(theta/2)*n[2]]).normalized()
        
        a_world = quaternion.rotate_vectors(q_g,a_world)
        w_world = quaternion.rotate_vectors(q_g,w_world)

        # pca = PCA(n_components=1)
        # pca.fit(a_world[t<cal_time-data[0,0]])
        # x_pca = pca.components_[0]
        # rot_ax = np.cross(x_pca,[1,0,0])
        # theta = np.arccos(np.dot(x_pca,[1,0,0]))
        pca = PCA(n_components=1)
        pca.fit(w_world[t<cal_time-data[0,0]])
        z_pca = pca.components_[0]
        rot_ax = np.cross(z_pca,[0,0,1])
        theta = np.arccos(np.dot(z_pca,[0,0,1]))
        q_pca = quaternion.as_quat_array([np.cos(theta/2),0,np.sin(theta/2)*np.sign(rot_ax[1]),0])
        
        q_f = q_pca * q_g  # q_f is the rotation from imu global frame to global lab-fixed anatomical frame
        q_init = q_imu[t<5].conj() * q_f.conj()  # equivalent to conj(q_f*q_imu); gives the quaternions from global lab-fixed anatomical frame to local imu frame at each timestep
        q_0 = quaternion.from_float_array(np.mean(quaternion.as_float_array(q_init),axis=0))  # left mutliplication by q_0 gives orientation relative to starting orientation
        q_f_.append(q_f)
        q_ana.append(q_f * q_imu * q_0)  #  q_f * q_imu * q_0 gives the orientation quaternion between the starting orientation and the current orientation for any timestep of q_imu
        
        w_world = quaternion.rotate_vectors(q_pca,w_world)
        a_ft = quaternion.as_vector_part(q_f* quaternion.from_vector_part(data[:,1:4]) * q_f.conj())
        fa_ana.append(a_ft)
        omega_world.append(w_world)
    
    return q_f_, q_ana, fa_ana, omega_world

def foot_contact_id_foot(w_ft, foot, shank):

    
    cur_search = 1  # keep track of what event we are searching for. 1:initial contact, 2: mid-stance, 3:toe-off, 4: mid-swing
    ic = []
    mst = [0]
    to = []
    msw = []
    temp = 0
    n=len(shank)
    c1 = 0
    for i in range(n):
        t = max(shank[i,0],foot[0,0])  # timestamp of current time step or maximum of first timestamps
        j = max(np.nonzero(foot[:,0]<=t)[0][-1],0) # the nonzero function will return the index of the data point with the timestamp closest to matching the shank timestamp without going over (simulates real-time behavior if there are dropped packets)
        
    #for i in range(n):
    # gait event detection - for post-processing, technically we can use i-1, i, i+1 since we can look forward in time (i+1); however to mimic real-time processing all time steps need to be current or past, so I have adjusted the below code to use i-2, i-1, i
    # if we implement a 2nd order median filter as the paper with the gait event detection did, we will need to increase this delay by one to get a filtered value for the last time step
        if j != 0: 
            if cur_search == 1:
                if w_ft[j] > 1.0:
                    if w_ft[j] > w_ft[temp]:
                        temp = j
                elif temp > 1:
                    ic.append(temp)
                    temp = 0
                    cur_search = 2
        
            elif cur_search == 2:
                if foot[j,11] == 0:
                    mst.append(j)
                    cur_search = 3
        
            elif cur_search == 3:
                if w_ft[j] > 2:
                    if w_ft[i] > w_ft[temp]:
                        temp = j
                elif temp > 0:
                    to.append(temp)
                    temp = 0
                    cur_search = 4
        
            elif cur_search == 4:
                if w_ft[j] < -0.5:
                    if w_ft[j] < w_ft[temp]:
                        temp = j
                elif temp > 0:
                    msw.append(temp)
                    temp = 0
                    cur_search = 1
            c1 +=1
            print(c1)

    # mst.append(i)
    gait_events= [ic, mst, to, msw]
    return  gait_events

def foot_contact_id_shank(w_sh):
    cur_search = 1  # keep track of what event we are searching for. 1:midswing, 2:FS, 3:FO
    flag = False  # flag used in various aspects of event detection
    temp = -1
    msw = []
    mst = []
    fo = []
    fs = []
    n=len(w_sh)
    # m=len(shank)-len(shank[cal_time>=shank[:,0]])
    for i in range(n):
    # gait event detection - for post-processing, technically we can use i-1, i, i+1 since we can look forward in time (i+1); however to mimic real-time processing all time steps need to be current or past, so I have adjusted the below code to use i-2, i-1, i
    # if we implement a 2nd order median filter as the paper with the gait event detection did, we will need to increase this delay by one to get a filtered value for the last time step
        if i > 1:
            # print(i,cur_search,flag)
            if cur_search == 1:  # mid-swing search
                if flag:  # wait for shank angular velocity falling edge zero crossing
                    if w_sh[i-1] < w_sh[i-2] and w_sh[i-1] <= w_sh[i] and w_sh[i-1] < -2:  # look for local minima with vel < -2 rad/s
                        msw.append(i-1)
                        cur_search = 2  # move on to FS search
                        flag = False  # reset for next search segment
                elif w_sh[i-2] >= 0 and w_sh[i-1] < 0:  # identify falling edge zero crossing
                    flag = True
            elif cur_search == 2:  # FS search
                if flag:  # wait for shank ang vel rising edge zero crossing
                    if w_sh[i-1] > w_sh[i-2] and w_sh[i-1] > w_sh[i]:  # look for local maxima
                        fs.append(i-1)
                        wait_time=w_sh[i-1,0]
                        cur_search = 3  # move on the FS search
                        flag = False  # reset flag for next search segment
                elif w_sh[i-1] > 0:  # identify rising edge zero crossing; we already know from midswing segment that angular velocity is less than zero, so we only need to identify a positive value
                    flag = True
            elif cur_search == 3:  # FO search
                if flag:  # wait at least 200 ms
                    if w_sh[i-1] > w_sh[i-2] and w_sh[i-1] >= w_sh[i]:  # look for local maxima
                        temp = i-1
                    if w_sh[i-2] >= 0 and w_sh[i-1] < 0:  # identify falling edge
                        if temp > 0:
                            fo.append(temp)  # FO occurs at last local maxima
                            mst.append(int(np.round((temp - fs[-1])/2))+fs[-1])
                        temp = -1  # reset temp for next gait cycle
                        cur_search = 1  # move on to MSW search
                        #  DO NOT reset flag since we have already identified a falling edge zero crossing
                elif w_sh[i-1,0] - wait_time >= .2:
                    flag = True
            else:
                raise RuntimeError(f'unexpected value of cur_search: {cur_search}')
    
    # gait_events = [fs, fo, msw]
    return fs, fo, mst, msw   

def joint_angle_calc(pelvis, thigh, shank, foot, q_ana):
    # ###  JOINT ANGLES  ###
    n = len(shank)  # use shank as basis, since we use shank angular velocity for gait event detection
    q_h = quaternion.as_quat_array(np.zeros((n,4)))
    q_k = quaternion.as_quat_array(np.zeros((n,4)))
    q_a = quaternion.as_quat_array(np.zeros((n,4)))
    
    for i in range(n):
        t=max(shank[i,0],pelvis[0,0],thigh[0,0],foot[0,0])  # timestamp of current time step or maximum of first timestamps
        q_pv_t = q_ana[0][max(np.nonzero(pelvis[:,0]<=t)[0][-1],0)]  # the nonzero function will return the index of the data point with the timestamp closest to matching the shank timestamp without going over (simulates real-time behavior if there are dropped packets)
        q_th_t = q_ana[1][max(np.nonzero(thigh[:,0]<=t)[0][-1],0)].conj()
        q_sh_t = q_ana[2][i].conj()
        q_ft_t = q_ana[3][max(np.nonzero(foot[:,0]<=t)[0][-1],0)]
        
        q_h[i]=q_pv_t.conj()*q_th_t
        q_k[i]=q_sh_t.conj()*q_th_t
        q_a[i]=q_sh_t.conj()*q_ft_t
        print(i)
    
    plt.close('all')
    
    q_h = quaternion.as_float_array(q_h)
    q_k = quaternion.as_float_array(q_k)
    q_a = quaternion.as_float_array(q_a)
    
    theta_h = 2*np.arccos(q_h[:,0]/np.sqrt(np.power(q_h[:,0],2)+np.power(q_h[:,3],2)))*np.sign(q_h[:,3])*180/np.pi
    theta_k = 2*np.arccos(q_k[:,0]/np.sqrt(np.power(q_k[:,0],2)+np.power(q_k[:,3],2)))*np.sign(q_k[:,3])*180/np.pi
    theta_a = 2*np.arccos(q_a[:,0]/np.sqrt(np.power(q_a[:,0],2)+np.power(q_a[:,3],2)))*np.sign(q_a[:,3])*180/np.pi

    return theta_h, theta_k, theta_a

def vertical_correction(shank, pos, imu_shank_theta, fs, fo, leg_length):
    zero_cross = []
    ap_pos = []
    t=shank[:,0]-shank[0,0]
    flag = False
    j = 1
    for i in range(len(shank)):
        if i > 5*120: 
            # If a foot strike has been detected, set flag to true 
            if fs[j] == i:
                flag = True
             
            if fo[j] == i and flag == True: # If the next foot off is passed and flag is still true
                # Find the minimum anglular excursion 
                zero_cross.append(np.argmax(imu_shank_theta[fs[j]:fo[j],2]))
                if len(zero_cross) > 2:
                    time_samp = t[zero_cross[-2]:zero_cross[-1]]
                    time_samp = time_samp.reshape(-1,1)
                    int_time = time_samp[-1] - time_samp[0]      
                
                    x1 = pos[zero_cross[-2],1]
                    x2 = pos[zero_cross[-2],1] + np.cos(np.radians(imu_shank_theta[i,2])) # CLosest the shank comes to vertical 
                    pos_slope =  (x2 - x1)/int_time
                    
                    pos_drift = np.multiply(pos_slope,(np.array(time_samp)-time_samp[0]).T)
                    pos[zero_cross[-2]:zero_cross[-1],1] = pos[zero_cross[-2]:zero_cross[-1],1] - pos_drift - x1
                    flag = False
                    j +=1
                    
      
            if imu_shank_theta[i-1,2] < 0 and imu_shank_theta[i,2]>=0 and flag == True:
                zero_cross.append(i)
                if len(zero_cross) > 2:
                    time_samp = t[zero_cross[-2]:zero_cross[-1]]
                    time_samp = time_samp.reshape(-1,1)
                    int_time = time_samp[-1] - time_samp[0]      
                    
                    x1 = pos[zero_cross[-2],1]
                    x2 = pos[zero_cross[-1],1]
                    pos_slope =  (x2 - x1)/int_time
                    
                    pos_drift = np.multiply(pos_slope,(np.array(time_samp)-time_samp[0]).T)
                    pos[zero_cross[-2]:zero_cross[-1],1] = pos[zero_cross[-2]:zero_cross[-1],1] - pos_drift - x1
                    flag = False
                    j +=1
                    
    # ap_position
    j=1
    new_pos = [0]
    flag = False
    for i in range(len(shank)):
        # Set new position equal to 0 
        if fs[j] == i and flag == False :
            # new_pos.append[]
            flag = True 
        if i < fo[j] and flag == True:
            new_pos.append((leg_length*np.sin(imu_shank_theta[i,2])))
        if i == fo[j]:
            flag = True
            j += 1
            ap_pos.append(np.array(my_resample(new_pos,101)))    

def quaternions_to_euler_angles(quaternions):
    # Normalize quaternions
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)

    # Extract scalar and vector parts
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # ZYX convention
    roll_x = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x**2 + y**2))
    pitch_y = np.arcsin(2.0 * (w*y - z*x))
    yaw_z = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y**2 + z**2))

    # Convert angles to degrees
    roll_x = np.degrees(roll_x)
    pitch_y = np.degrees(pitch_y)
    yaw_z = np.degrees(yaw_z)

    return np.column_stack([roll_x, pitch_y, yaw_z])

def my_resample(n, x, t=None):
    # Create new sample points 'xx' linearly spaced between 0 and 100.
    xx = np.linspace(0, 100, n)

    if t is None:
        # Create a set of percentage values 'percent' linearly spaced between 0 and 100,
        # based on the length of the input data 'x'.
        percent = np.linspace(0, 100, len(x))
    else:
        percent = (t - t[0]) * 100/(t[-1]-t[0])

    # Compute the B-spline representation of the input data 'x' using cubic splines (k=3).
    tck = splrep(percent, x, k=3)

    # Evaluate the B-spline at the new sample points 'xx' to obtain the resampled data.
    y = splev(xx, tck)

    return y

def my_resample_multi(X, n):
    # Assuming all input arrays have the same length
    length = len(X)
    # Create new sample points 'xx' linearly spaced between 0 and 100.
    xx = np.linspace(0, 100, n)
    # create a set of percentage values 'percent' linearly spaced between 0 and 100,
    # based on the length of the first input data.
    percent = np.linspace(0, 100, length)
    # Initialize a list to store resampled arrays
    resampled_arrays = []

    for x in X.T:
        # Compute the B-spline representation of the input data 'x' using cubic splines (k=3).
        tck = splrep(percent, x, k=3)

        # Evaluate the B-spline at the new sample points 'xx' to obtain the resampled data.
        y = splev(xx, tck)

        # Append the resampled array to the list
        resampled_arrays.append(y)
    
    return resampled_arrays

def Cut_trial(shank, shank_q, shank_a, foot):
    
    # Extracts a portion of the 'seg' array, removing the first 21 columns.
    # Creates two plots to interactively select a region of interest on the second plot.
    # Rounds the selected coordinates to the nearest integer.
    # Extracts a portion of the 'seg' array based on the selected coordinates.
    # Adjusts the 'fs' vector based on the selected region of interest.
    # Returns the modified 'seg_new' data and the adjusted 'fs_new' vector.
 
    t1 =  np.arange(0, len(shank), 1)
    
    plt.figure(figsize=(10,15))
    plt.plot(t1,shank[:,2])
    x = plt.ginput(3)     
    x = np.round(x,0)
    plt.close('all')
    x = x[1:,0].astype(np.int64)
    # print(x)
    
    
    seg = shank[x[0]:x[1],:]
    seg_q = quaternion.as_float_array(shank_q[x[0]:x[1]])
    seg_a = shank_a[x[0]:x[1],:]
    seg_theta = quaternions_to_euler_angles(seg_q)
    t1 =  np.arange(0, len(seg), 1)
    w_sh = shank[x[0]:x[1],5]

    fs, fo, mst, msw = foot_contact_id_shank(w_sh)

    # fs, fo, mst, msw = foot_contact_id_foot(w_ft, foot, shank)
        
    return seg, seg_q, seg_a, seg_theta, fs, fo, mst

def UI_get_pkl():
    '''    
    We create a GUI window using tkinter and hide it with root.withdraw() so that it doesn't appear on the screen.
    We use filedialog.askopenfilename() to open a file dialog that allows the user to select a .pkl file. The filetypes argument specifies that only pickle files should be shown in the file dialog.
    We check if the user selected a file. If a file is selected, we open and read the pickle file using pickle.load() and store the content in the loaded_data variable.
    If no file is selected, we print a message indicating that no file was selected.
    Remember to replace ("Pickle files", "*.pkl") with the specific file filter you want to use, or you can allow all files by omitting the filetypes parameter. Additionally, you may need to install the tkinter library if it's not already available on your system.
    Returns
    -------
    loaded_data : TYPE
        DESCRIPTION.

    '''
    # Create a GUI window (you can skip this if you have a tkinter window)
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    # Ask the user to select a .pkl file
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    
    # Check if the user selected a file
    if file_path:
        # Extract the original file name
        # Open and read the selected pickle file
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)

        # Now, 'loaded_data' contains the content of the pickle file
    else:
        print("No file selected")
    return loaded_data

def main_func():
    with open('s01_alpha_wn_cg.npy','rb') as f:
        pelvis = np.load(f)
        thigh = np.load(f)
        shank = np.load(f)
        foot = np.load(f)
        contra = np.load(f)
        rt_fs = np.load(f)
        joint_ang = np.load(f)
        cal_time = np.load(f)

    leg_length = 0.50
    np.set_printoptions(precision=3)
    plt.close()
        
    # idx_pelvis = select_Calibration_Region(pelvis)
    # idx_lower = select_Calibration_Region(shank)
    sf = 120
    segment_list = [pelvis, thigh, shank, foot, contra]
        
    q_f_, q_ana, fa_ana, omega_world = ana_Calibration(segment_list,  cal_time)

    shank_q = q_ana[2]
    shank_a = fa_ana[2]
    shank_w = shank[:,5]
    shank_t = shank[:,0]-shank[0,0]

    seg, seg_q, seg_a, seg_theta, fs, fo, mst = Cut_trial(shank, shank_q, shank_a, foot)

    w_ft = foot[:,5]
    # fs_foot, fo_foot, mst, msw = foot_contact_id_foot(w_ft, foot, shank)
    # fs, fo, mst, msw = foot_contact_id_shank(shank_w)

    # len(fs_foot) - len(fs_shank)

    # plt.plot(seg_theta[:,2])

    zupInd = [0,-1]
    acc = [[0,0,0]]
    vel = [[0,0,0]]
    pos = [[0,0,0]]
    new_pos = []
    scat_ind = []


    t = np.arange(0,len(seg)/sf,1/sf)
    pos_llte_vert = []
    zero_cross = [0]
    j = 0
    fs_real = []
    k = 0
    fo_real = []
    ii = 0 
    mst_real = []
    contact_flag = False
    mst_flag = False
    fo_flag = False
    llte_list = []

    for i in range(len(seg[:,0])):
        if seg[i,11] == 0:
            if zupInd[1] == 0:
                zupInd[1] = i
        else:
            zupInd[1] = 0

        if i == len(seg[:,0])-1:
            zupInd[1] = i
            
        # If a foot strike has been detected, set flag to true
        try:
            if fs[j] == i and j < len(fs):
                contact_flag = True
                fs_real.append(fs[j])
                j += 1
        except:
            pass
    
        try:
            if fo[k] == i and k < len(fo):
                fo_flag = True
                fo_real.append(fo[k])
                k += 1
        except:
            pass
        try:
            if mst[ii] == i and ii < len(mst):
                mst_flag = True
                mst_real.append(mst[ii])
                ii += 1
        except:
            pass
            
            # print('true')
            
        # process a chunk of data whenever we have a new zero-velocity update
        if zupInd[1] > 0:
            scat_ind.append(i)
            time_samp=t[zupInd[0]:zupInd[1]]
            # fa_samp=np.array(shank[zupInd[0]:zupInd[1],1:4])
            # q_samp= q_f_[2] #* q_imu[zupInd[0]:zupInd[1]]
            # fa_samp = quaternion.as_vector_part(q_samp * quaternion.from_vector_part(fa_samp) * q_samp.conj())  # rotate acc to align with initial anatomical frame
            fa_samp = seg_a[zupInd[0]:zupInd[1],:]
            int_time = time_samp[-1] - time_samp[0]
            samp_vel = integrate.cumulative_trapezoid(fa_samp,time_samp,initial=0,axis=0)
            v_slope = (samp_vel[-1]-samp_vel[0])/int_time
            v_drift = np.multiply(v_slope,(np.array(time_samp)-time_samp[0])[:,np.newaxis])
            samp_vel=samp_vel-v_drift

            if any(abs(samp_vel[0]) >= 1e5) or any(abs(samp_vel[-1]) >= 1e5):
                raise RuntimeError(f'Issue with drift correction. First and last values should be zero, but are {samp_vel[0]} and {samp_vel[-1]}')

            samp_pos = integrate.cumulative_trapezoid(samp_vel,time_samp,initial=0,axis=0)
            
            
            try:
                samp_pos+=np.array(pos[-1])
            except IndexError:
                pass
                    
            acc += list(fa_samp)
            vel += list(samp_vel)
            pos += list(samp_pos)
            
            samp_theta = seg_theta[zupInd[0]:zupInd[1],2]
            crossing = np.where(np.diff(np.sign(samp_theta)))[0] # find zero crossings 
            if contact_flag == True and len(crossing) > 1 :# or (mst_flag == True and zupInd[0] < mst_real[-1])):
            
                if len(crossing) > 1:
                    zero_cross.append(crossing[-1]+zupInd[0]) 
                    est_h = leg_length

                else:
                    zero_cross.append(mst_real[-1])

                    est_h = leg_length*np.cos(np.radians(samp_theta[mst_real[-1] - zupInd[0]]))        

                
                if len(zero_cross) >= 2:
                    pos_samp  = np.array(pos)[zero_cross[-2]:zero_cross[-1],1]
                    
                    time_samp = t[zero_cross[-2]:zero_cross[-1]]
                    time_samp = time_samp.reshape(-1,1)
                    int_time  = time_samp[-1] - time_samp[0]      
                    
                    x1 = pos_samp[0]
                    x2 = pos_samp[-1]
                    pos_slope =  (x2 - x1)/int_time
                    pos_drift = np.multiply(pos_slope,(np.array(time_samp)-time_samp[0]).T)
                    pos_samp  = pos_samp - pos_drift - x1 + est_h
                    pos_samp = pos_samp.reshape(1,-1)

                    c1 = 0
                    for c in range(zero_cross[-2],zero_cross[-1]):
                        pos[c][1] = pos_samp[0][c1]
                        c1 += 1
                    contact_flag = False
                    mst_flag = False
                    
                    
                # if fo_flag:
                #     theta_llte = seg_theta[fs_real[-1]:fo_real[-1],2]
                #     pos_ap = np.sin(np.radians(theta_llte))
                #     vert_pos = np.array(pos)[fs_real[-1]:fo_real[-1],1]
                #     llte_stance = np.array([theta_llte, pos_ap, vert_pos]).T
                #     llte_stance = my_resample_multi(llte_stance, 101)
                #     llte_list.append(llte_stance)
                #     fo_flag = False   
                
            zupInd = [zupInd[1], -1]  # reset indeces for the next chunk       
    c1 = 0
    llte_list=[]
    if len(fo) <= len(fs):
        for i in range(len(fo)):
            j = max(np.nonzero(np.array(fs)<=np.array(fo)[i])[0][-1],0)
            print(fo[i] - fs[j])
            theta_llte = seg_theta[fs[j]:fo[i],2]
            if theta_llte[0] < 0:
                plt.plot(theta_llte)
                pos_ap = np.sin(np.radians(theta_llte))
                vert_pos = np.array(pos)[fs[j]:fo[i],1]
                llte_stance = np.vstack((theta_llte, pos_ap, vert_pos)).T
                # llte_stance = my_resample_multi(llte_stance, 101)
                llte_list.append(my_resample_multi(llte_stance, 101))
                
    theta_llte, ap_llte, vert_llte= [], [],[]
    for cur_list in llte_list: 
        theta_llte.append(cur_list[0])   
        ap_llte.append(cur_list[1])  
        vert_llte.append(cur_list[2])

    theta_llte = np.mean(np.array(theta_llte), axis = 0) 
    ap_llte =  np.mean(np.array(ap_llte), axis = 0)  
    vert_llte =  np.mean(np.array(vert_llte), axis = 0)  
        
    LLTE_targets = np.vstack((theta_llte, ap_llte, vert_llte))

    pickle_path = 'C:\\Users\\srd8236\\Desktop\\Dynamics and Power Encoders\\RT-IMU-Py\\Stance_Phase_sk_LLTE_data.pkl'
    with open(pickle_path, 'wb') as file:
        pd.to_pickle(LLTE_targets, file)
        
        # if fo_flag == True and contact_flag == True:
        #     print(i)
        #     if len(zero_cross) > 1:
        #         zero_cross.append(np.argmax(seg_theta[fs[j]:fo[k],2])+fs[j])
        #         print(zero_cross[-1])
        #         pos_samp = np.array(pos)[zero_cross[-2]:zero_cross[-1],1]
        #         x1 = pos_samp[0]
        #         x2 = pos_samp[-1] + np.cos(np.radians(seg_theta[i,2])) # Closest the shank comes to vertical 
                
            # contact_flag = False
            # fo_flag = False            
        

        # if len(zero_cross) > 2 and len(pos)>zero_cross[-1]:
        #     time_samp = shank_t[zero_cross[-2]:zero_cross[-1]]
        #     time_samp = time_samp.reshape(-1,1)
        #     int_time = time_samp[-1] - time_samp[0]      
                    
        
        #     x1 = pos_array[zero_cross[-2],1]
        #     x2 = pos_array[zero_cross[-2],1] + np.cos(np.radians(seg_theta[i,2])) # CLosest the shank comes to vertical 
        #     pos_slope =  (x2 - x1)/int_time
            
        #     pos_drift = np.multiply(pos_slope,(np.array(time_samp)-time_samp[0]).T)
        #     pos_array[zero_cross[-2]:zero_cross[-1],1] = pos_array[zero_cross[-2]:zero_cross[-1],1] - pos_drift - x1
        #     flag = False
        #     j += 1 
            



    # plt.plot(seg_theta[:,2])
    # plt.plot(seg[:,11]*15)
    # plt.figure()
    # plt.plot(np.cumsum(np.array(pos)[:,1]))
    # plt.plot(pos_array[:,1])
    # Calculate veritcal position and from acceleration and oreintation


    # pos_array = np.concatenate(new_pos, axis = 1).T
    plt.plot(np.array(pos)[:,1])

    # foot_omega = omega_world[3]

    plt.figure()
    plt.plot(t, seg_theta[:,2])
    plt.scatter(t[fs], seg_theta[fs,2], color = 'red')
    plt.scatter(t[fo], seg_theta[fo,2], color = 'green')
    plt.scatter(t[zero_cross], seg_theta[zero_cross, 2], color = 'black')
    plt.scatter(t[mst], seg_theta[mst, 2], color = 'orange')

    # plt.figure()
    # shank_q_y = quaternion.as_float_array(shank_q)[:,3]
    # plt.plot(shank_t, quaternion.as_float_array(shank_q)[:,3])
    # plt.scatter(shank_t[fs_foot], shank_q_y[fs_foot], color = 'red')
    # plt.scatter(shank_t[fo_foot], shank_q_y[fo_foot], color = 'green')
    # plt.scatter(shank_t[zero_cross], shank_q_y[zero_cross], color = 'green')
    # # fs, msw, fo = foot_contact_id_shank(shank_w)

    # plt.figure()
    # shank_q_y = quaternion.as_float_array(shank_q)[:,3]
    # plt.plot(shank_t, quaternion.as_float_array(shank_q)[:,3])
    # plt.scatter(shank_t[fs_shank], shank_q_y[fs_shank], color = 'red')
    # plt.scatter(shank_t[fo_shank], shank_q_y[fo_shank], color = 'green')
    # plt.scatter(shank_t[zero_cross], shank_q_y[zero_cross], color = 'green')

    # flag = False
    # j = 1
    # for i in range(shank_t):
    #     if i > 5*120: 
    #         # If a foot strike has been detected, set flag to true 
    #         if fs[j] == i:
    #             flag = True
    #             if fo[j] == i and flag == True: # If the next foot off is passed and flag is still true
    #                 # Find the minimum anglular excursion 
    #                 zero_cross.append(np.argmax(imu_shank_theta[fs[j]:fo[j],2]))
    #                 if len(zero_cross) > 2:
    #                     time_samp = t[zero_cross[-2]:zero_cross[-1]]
    #                     time_samp = time_samp.reshape(-1,1)
    #                     int_time = time_samp[-1] - time_samp[0]      
                    
    #                     x1 = pos[zero_cross[-2],1]
    #                     x2 = pos[zero_cross[-2],1] + np.cos(np.radians(imu_shank_theta[i,2])) # CLosest the shank comes to vertical 
    #                     pos_slope =  (x2 - x1)/int_time
                        
    #                     pos_drift = np.multiply(pos_slope,(np.array(time_samp)-time_samp[0]).T)
    #                     pos[zero_cross[-2]:zero_cross[-1],1] = pos[zero_cross[-2]:zero_cross[-1],1] - pos_drift - x1
    #                     flag = False
    #                     j +=1
                        
        
    #             if imu_shank_theta[i-1,2] < 0 and imu_shank_theta[i,2]>=0 and flag == True:
    #                 zero_cross.append(i)
    #                 if len(zero_cross) > 2:
    #                     time_samp = t[zero_cross[-2]:zero_cross[-1]]
    #                     time_samp = time_samp.reshape(-1,1)
    #                     int_time = time_samp[-1] - time_samp[0]      
                        
    #                     x1 = pos[zero_cross[-2],1]
    #                     x2 = pos[zero_cross[-1],1]
    #                     pos_slope =  (x2 - x1)/int_time
                        
    #                     pos_drift = np.multiply(pos_slope,(np.array(time_samp)-time_samp[0]).T)
    #                     pos[zero_cross[-2]:zero_cross[-1],1] = pos[zero_cross[-2]:zero_cross[-1],1] - pos_drift - x1
    #                     flag = False
    #                     j +=1








    # theta_h, theta_k, theta_a = joint_angle_calc(pelvis, thigh, shank, foot, q_ana)

    # plt.figure(figsize=(10,15))
    # plt.plot(t1,LLTE_ser[:,2])
    # x = plt.ginput(2)     
    # x = np.round(x,0)
    # plt.close('all')
    # x = x[:,0].astype(np.int64)
    # # print(x)
    # plt.figure(figsize=(10,15))
    # seg_new = LLTE_ser[x[0]:x[1],:]
    # angle_new = angles[x[0]:x[1],:]
    # t1 =  np.arange(0, len(seg_new), 1)
    


if __name__ == '__main__':
    main_func()