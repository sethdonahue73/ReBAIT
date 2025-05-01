# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 21:31:48 2023

@author: srd8236
"""
from statistics import mean, stdev
import numpy as np
import quaternion
import math
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from sklearn.decomposition import PCA
import pandas as pd
import pickle
import tkinter as tk
from tkinter import filedialog
from scipy import integrate
#from matplotlib.widgets import RectangleSelector

def my_resample(x, n):
    # Create new sample points 'xx' linearly spaced between 0 and 100.
    xx = np.linspace(0, 100, n)

    # Create a set of percentage values 'percent' linearly spaced between 0 and 100,
    # based on the length of the input data 'x'.
    percent = np.linspace(0, 100, len(x))

    # Compute the B-spline representation of the input data 'x' using cubic splines (k=3).
    tck = splrep(percent, x, k=3)

    # Evaluate the B-spline at the new sample points 'xx' to obtain the resampled data.
    y = splev(xx, tck)

    return y

def ang_vel_bias_correction(seg):
    bias = np.mean(seg[:500,4:7], axis=0) #mean of the first 500 samples 
    # print(mean)
    seg[:,4:7] = seg[:,4:7] - bias
    
    return seg

def acc_bias_correction(seg):
    bias = np.mean(seg[:500,1:4], axis=0) #mean of the first 500 samples 
    # print(mean)
    seg[:,1:4] = seg[:,1:4] - bias
    
    return seg


def zupt_position_ASB(aworld, t, zupt_index):
    # Force the first and last sample to be zero points if they aren't already
    if zupt_index[0] != 0:
        zupt_index = [0] + zupt_index
    if zupt_index[-1] != len(t) - 1:
        zupt_index = zupt_index + [len(t) - 1]

    # Initialize variables
    v = np.zeros((len(t), 3))
    d = np.zeros((len(t), 3))

    # Integrate between zero velocity points and apply drift corrections
    for uu in range(len(zupt_index) - 1):
        acc = aworld[zupt_index[uu]:zupt_index[uu + 1] + 1, :]
        time = t[zupt_index[uu]:zupt_index[uu + 1] + 1]
        int_time = t[zupt_index[uu + 1]] - t[zupt_index[uu]]

        vx = integrate.cumtrapz(time, acc[:, 0])
        vy = integrate.cumtrapz(time, acc[:, 1])
        vz = integrate.cumtrapz(time, acc[:, 2])

        # Assume linear drift velocities and correct
        vx_slope = (vx[-1] - vx[0]) / int_time
        vx_drift = vx_slope * (time - time[0])
        vx -= vx_drift

        vy_slope = (vy[-1] - vy[0]) / int_time
        vy_drift = vy_slope * (time - time[0])
        vy -= vy_drift

        vz_slope = (vz[-1] - vz[0]) / int_time
        vz_drift = vz_slope * (time - time[0])
        vz -= vz_drift

        # Store velocity values
        v[zupt_index[uu]:zupt_index[uu + 1] + 1, :] = np.column_stack((vx, vy, vz))

        # Integrate to obtain position
        dx = integrate.cumtrapz(time, vx)
        dy = integrate.cumtrapz(time, vy)
        dz = integrate.cumtrapz(time, vz)

        # Assume zero-velocity locations are at the same z position and apply
        # linear drift correction
        dz_slope = (dz[-1] - dz[0]) / int_time
        dz_drift = dz_slope * (time - time[0])
        dz -= dz_drift

        # Add initial position to integrated values
        if uu == 0:
            dx += d[zupt_index[uu], 0]
            dy += d[zupt_index[uu], 1]
            dz += d[zupt_index[uu], 2]

        # Store position values
        d[zupt_index[uu]:zupt_index[uu + 1] + 1, :] = np.column_stack((dx, dy, dz))

    return v, d

def orientation_est(a, w, t):
    # Define an initial direction cosine matrix based on initial orientation
    # Assume at rest for the first 2 seconds
    rest = t < 2
    ma = np.mean(a[rest, :], axis=0)
    Z = ma / np.linalg.norm(ma)
    Y = np.cross(Z, np.array([1, 0, 0]))
    Y = Y / np.linalg.norm(Y)
    X = np.cross(Y, Z)

    DCMstart = np.vstack((X, Y, Z))

    # Apply rotations using the method found in McGinnis and Perkins 2012
    R = [DCMstart]
    for ii in range(1, len(w)):
        time_step = t[ii] - t[ii - 1]
        theta = 0.5 * (w[ii - 1, :] + w[ii, :]) * time_step
        skew_theta = skewsym(theta)
        R_last = R[ii - 1]
        eye_plus_theta = (np.eye(3) + (2 / (1 + 0.5 * np.dot(theta, theta.T)))) * (0.5 * skew_theta + 0.25 * np.dot(skew_theta, skew_theta))
        #eye_plus_theta = eye_plus_theta / np.linalg.norm(eye_plus_theta, axis=0)
        R_new = np.dot(R_last, eye_plus_theta)
        R_new = R_new /  np.linalg.norm(R_new , axis=0)
        #R_new = R_last @ eye_plus_theta
        R.append(R_new)

    DCM = np.array(R)

    return DCM

def estimate_orientation(a, w, t):
    # Initialize orientation quaternion
    q_orientation = [quaternion.one]

    for ii in range(1, len(w)):
        time_step = t[ii] - t[ii - 1]
        
        # Calculate quaternion rate from angular velocity
        omega_quat_rate = 0.5 * quaternion.quaternion(0, *w[ii, :])

        # Integrate quaternion rate to get quaternion change
        delta_q = np.exp(0.5 * quaternion.as_float_array(omega_quat_rate))
        delta_q = quaternion.as_quat_array(delta_q)

        # Update orientation quaternion
        q_last = q_orientation[-1]
        q_new = q_last * delta_q
        q_orientation.append(q_new)

    return q_orientation

def skewsym(x):
    # Put a vector in skew symmetric form
    scm = np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])
    
    return scm


def zero_velocity_finder(a, w, a_limit=1.0, w_limit=10 * np.pi / 180):
    grav = 9.80665  # m/s^2 (gravity)

    # Calculate angular velocity magnitude and acceleration magnitude
    wmag = np.linalg.norm(w, axis=1)
    amag = np.linalg.norm(a, axis=1) - grav

    # Identify zero points that meet threshold conditions
    zupt_index = np.where((wmag < w_limit) & (amag < a_limit) & (amag > -a_limit))[0]

    return zupt_index

    
def Cut_trial(seg, fs):
    
    # Extracts a portion of the 'seg' array, removing the first 21 columns.
    # Creates two plots to interactively select a region of interest on the second plot.
    # Rounds the selected coordinates to the nearest integer.
    # Extracts a portion of the 'seg' array based on the selected coordinates.
    # Adjusts the 'fs' vector based on the selected region of interest.
    # Returns the modified 'seg_new' data and the adjusted 'fs_new' vector.
 
    y = seg[:,21:]
    t1 =  np.arange(0, len(y), 1)
    
    plt.figure(figsize=(10,15))
    plt.plot(t1,seg[:,21])
    x = plt.ginput(2)     
    x = np.round(x,0)
    plt.close('all')
    x = x[:,0].astype(np.int64)
    # print(x)
    plt.figure(figsize=(10,15))
    seg_new = seg[x[0]:x[1],:]
    t1 =  np.arange(0, len(seg_new), 1)
    contact_1 = np.argmax(fs>x[0])
    if np.argmax(fs>x[1]) != 0:
        contact_2 = np.argmax(fs>x[1])
    else: 
        contact_2 = fs[-1]
        
    # print(contact_2)
    fs_new = fs[contact_1:contact_2]
    fs_new = fs_new - x[0]
    # print(contact_1, contact_2)
   # plt.plot(t1,seg[x[0]:x[1],5])
    return seg_new, fs_new
    

def calculate_knee_position(height, foot_length, shank_seg):
    shank_ml = shank_seg[10000:10120,1]
    shank_vert = shank_seg[:,3]
    shank_sag = shank_seg[:,21]
    ml_norm = my_resample(shank_ml, 100)
    return


def calc_LLTE_Components(seg, fs):
    x = np.arange(0, 100, 1)
    fig = plt.figure(figsize=(6.4,9.6))
    ax1 = fig.add_subplot(311)
    ax1.set_xlim(0,100)
    ax1.set_title('Knee Vertical Position')
    ax1.tick_params(bottom=False,labelbottom=False)
    ax1.set_ylabel('Knee Vertical (m)')
    ax2 = fig.add_subplot(312)
    ax2.set_xlim(0,100)
    ax2.set_title('Knee Anterior Posterior')
    ax2.tick_params(bottom=False,labelbottom=False)
    ax2.set_ylabel('Knee A/P (m)')
    ax3 = fig.add_subplot(313)
    ax3.set_xlim(0,100)
    ax3.set_title('Shank Angle')
    ax3.set_xlabel('Gait Cycle (%)')
    ax3.set_ylabel('Angle (deg)')
    
    y_pos = seg[:,[12,13,21]]
    x = np.arange(0, 100, 1)
    norm_data = [[[] for i in range(100)] for i in range(3)]
    print(len(fs))
    y_norm = np.zeros((3, 100, len(fs)-20))
    c1 = 0
    for i in range(10,len(fs)-10):
        y_v=y_pos[fs[i]:fs[i+1],0]
        y_a=y_pos[fs[i]:fs[i+1],1]
        y_theta=y_pos[fs[i]:fs[i+1],2]*-1
 
        y_v = y_v[0] - y_v
        y_a = y_a[0] - y_a
        y_theta = y_theta[0] - y_theta
 
 
        y_v = my_resample(y_v, 101)
        y_a = my_resample(y_a, 101)
        y_theta = my_resample(y_theta, 101)
        
        y_norm[:,:,c1] = [y_v, y_a, y_theta]
        c1 +=1
        for j in range(len(x)):
            norm_data[0][int(x[j])%100].append(y_v[j])
            norm_data[1][int(x[j])%100].append(y_a[j])
            norm_data[2][int(x[j])%100].append(y_theta[j])
        ax1.plot(x,y_v,color='orange',linewidth=.5)
        ax2.plot(x,y_a,color='orange',linewidth=.5)
        ax3.plot(x,y_theta,color='orange',linewidth=.5)
  
    for i in range(len(norm_data[0])):
        norm_data[0][i] = np.mean(norm_data[0][i])
        norm_data[1][i] = np.mean(norm_data[1][i])
        norm_data[2][i] = np.mean(norm_data[2][i])

    norm_data[0]=[norm_data[0][-1]] + norm_data[0] + [norm_data[0][0]]
    norm_data[1]=[norm_data[1][-1]] + norm_data[1] + [norm_data[1][0]]
    norm_data[2]=[norm_data[2][-1]] + norm_data[2] + [norm_data[2][0]]
    
    norm_gc = np.array(norm_data)
    LLTE_Vals = norm_gc[:,15:55]
       
       
    norm_x=np.array(range(len(norm_data[0])))-0.5
    ax1.plot(norm_x,norm_data[0],color='black',linewidth=2)
    ax2.plot(norm_x,norm_data[1],color='black',linewidth=2)
    ax3.plot(norm_x,norm_data[2],color='black',linewidth=2)
    plt.show()

    
    return y_norm, norm_gc, LLTE_Vals
    
def LLTE_calc(LLTE_tar,input_data):
    # input_data 3x100x#footconacts
    x_tar = LLTE_tar[:,1].T
    y_tar = LLTE_tar[:,0].T
    theta_tar = LLTE_tar[:,2].T
    LLTE_val = np.zeros(input_data.shape[2])
    l_foot = 0.28
    l_leg = 0.48
    for i in range(0, input_data.shape[2]):
        y = input_data[0,15:55,i]
        x = input_data[1,15:55,i]
        theta = input_data[2,15:55,i]
        LLTE_val[i] = np.sqrt(np.mean(np.square((y-y_tar)/l_leg) + np.square(x-x_tar/l_leg) + np.square((theta-theta_tar)/math.atan(l_leg/l_foot))))
        # np.sqrt(np.mean(np.square(y-y_tar)))    
    return LLTE_val

def plot_LLTE_Components(gait_LLTE, LLTE_vals, y_norm):
    fig = plt.figure(figsize=(6.4,9.6))
    #plt.title('Average')
    ax1 = fig.add_subplot(311)
    ax1.set_title('Vertical Position Change')
    ax1.tick_params(bottom=False,labelbottom=False)
    ax1.set_ylabel('Vertical Position (m)')
    ax2 = fig.add_subplot(312)
    ax2.set_title('AP Position Change')
    ax2.tick_params(bottom=False,labelbottom=False)
    ax2.set_ylabel('A/P Position (m)')
    ax3 = fig.add_subplot(313)
    ax3.set_title('Shank Angle (degrees)')
    ax3.set_xlabel('Gait Cycle (%)')
    ax3.set_ylabel('Shank Angle (degrees')
    y1 = gait_LLTE
    y2 = LLTE_vals.T
    x=np.arange(15, 55, 1)
    ax1.plot(x,y1[:,0],color='red')
    ax1.plot(x,y2[:,0],color='green')
    ax2.plot(x,y1[:,1],color='red')
    ax2.plot(x,y2[:,1],color='green')
    ax3.plot(x,y1[:,2],color='red', label = 'crouch')
    ax3.plot(x,y2[:,2],color='green', label = 'Normal Walking')
    ax3.legend(loc='upper left')

    fig = plt.figure(figsize=(6.4,9.6))
    #plt.title('Target in Red')
    ax1 = fig.add_subplot(311)
    ax1.set_title('Vertical Position Change')
    ax1.tick_params(bottom=False,labelbottom=False)
    ax1.set_ylabel('Vertical Position (m)')
    ax2 = fig.add_subplot(312)
    ax2.set_title('AP Position Change')
    ax2.tick_params(bottom=False,labelbottom=False)
    ax2.set_ylabel('A/P Position (m)')
    ax3 = fig.add_subplot(313)
    ax3.set_title('Shank Angle (degrees)')
    ax3.set_xlabel('Gait Cycle (%)')
    ax3.set_ylabel('Shank Angle (degrees')
    y1 = gait_LLTE
    y2 = y_norm
    x=np.arange(15, 55, 1)

    # ax3.plot(x,y2[:,2],color='green', label = 'Normal Walking')
    for i in range(0, y_norm.shape[2]):
        ax1.plot(x,y2[0,15:55,i],color='green',linewidth=.5)
        ax2.plot(x,y2[1,15:55,i],color='green',linewidth=.5)
        if i == 1:
            ax3.plot(x,y2[2,15:55,i],color='green',label = 'Stride-by-stride',linewidth=.5)
        else:
            ax3.plot(x,y2[2,15:55,i],color='green',linewidth=.5)

    ax1.plot(x,y1[:,0],color='red',linewidth=2.5)
    # ax1.plot(x,y2[:,0],color='green')
    ax2.plot(x,y1[:,1],color='red',linewidth=2.5)
    # ax2.plot(x,y2[:,1],color='green')
    ax3.plot(x,y1[:,2],color='red', label = 'Average',linewidth=2.5)
    ax3.legend(loc='upper left')

    fig = plt.figure(figsize=(6.4,9.6))
    ax1 = fig.add_subplot(311)
    ax1.set_title('Vertical Position Change')
    ax1.tick_params(bottom=False,labelbottom=False)
    ax1.set_ylabel('Vertical Position (m)')
    ax2 = fig.add_subplot(312)
    ax2.set_title('AP Position Change')
    ax2.tick_params(bottom=False,labelbottom=False)
    ax2.set_ylabel('A/P Position (m)')
    ax3 = fig.add_subplot(313)
    ax3.set_title('Shank Angle (degrees)')
    ax3.set_xlabel('Gait Cycle (%)')
    ax3.set_ylabel('Shank Angle (degrees')
    y1 = LLTE_vals.T
    y2 = y_norm
    x=np.arange(15, 55, 1)

    # ax3.plot(x,y2[:,2],color='green', label = 'Normal Walking')
    for i in range(0, y_norm.shape[2]):
        ax1.plot(x,y2[0,15:55,i],color='green',linewidth=.5)
        ax2.plot(x,y2[1,15:55,i],color='green',linewidth=.5)
        if i == 1:
            ax3.plot(x,y2[2,15:55,i],color='green',label = 'Normal Walking',linewidth=.5)
        else:
            ax3.plot(x,y2[2,15:55,i],color='green',linewidth=.5)

    ax1.plot(x,y1[:,0],color='red',linewidth=2.5)
    # ax1.plot(x,y2[:,0],color='green')
    ax2.plot(x,y1[:,1],color='red',linewidth=2.5)
    # ax2.plot(x,y2[:,1],color='green')
    ax3.plot(x,y1[:,2],color='red', label = 'crouch',linewidth=2.5)
    ax3.legend(loc='upper left')
    # # Create the RectangleSelector and connect it to the axis
    # rect_selector = RectangleSelector(ax1, onselect, drawtype='box', useblit=True, button=[1])

    # plt.show()


def Barplot_groups(data, num_groups):
    # Calculate the means and standard deviations
    means = [np.mean(data[group]) for group in data]
    std_devs = [np.std(data[group], ddof=1) for group in data]

    # Number of groups
    
    # Bar width
    bar_width = 0.35
    # X-axis positions for bars
    index = np.arange(num_groups)
    # Create the bar plot
    fig = plt.figure(figsize=(9.6,9.6))
    plt.bar(index, means, bar_width, label='Mean', yerr=std_devs, capsize=5)

    # Add labels, legend, and title
    plt.xlabel('Trials')
    plt.ylabel('LLTE Values')
    plt.title('Bar Plot with Mean and SD LLTE Values')
    plt.xticks(index, data.keys())
    plt.legend()

    # Show the plot
    plt.show()

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

# You can use 'loaded_data' as needed
def plot_3(x,y1,y2 = None,title=None,ylim=None,scat_ind=None,scatter=False,xlabel = None):
    fig = plt.figure(figsize=(6.4,9.6))
    ax1 = fig.add_subplot(311)
    # ax1.set_title('Knee Anterior/Posterior Position Error')
    # ax1.set_ylabel('Normalized RMSE')
    ax1.tick_params(bottom=False,labelbottom=False)
    ax2 = fig.add_subplot(312)
    # ax2.set_title('Knee Superior/Inferior Position Error')
    # ax2.set_ylabel('Normalized RMSE')
    ax2.tick_params(bottom=False,labelbottom=False)
    ax3 = fig.add_subplot(313)
    # ax3.set_title('Shank Angle Error')
    # ax3.set_ylabel('Normalized RMSE')
    if xlabel is not None:
        ax3.set_xlabel(xlabel)
    if scatter == False:
        ax1.plot(x,y1[:,0])
        ax2.plot(x,y1[:,1])
        ax3.plot(x,y1[:,2])
        if y2 is not None:
            ax1.plot(x,y2[:,0],color='green')
            ax2.plot(x,y2[:,1],color='green')
            ax3.plot(x,y2[:,2],color='green')
    else:
        ax1.scatter(x,y1[:,0])
        ax2.scatter(x,y1[:,1])
        ax3.scatter(x,y1[:,2])
        if y2 is not None:
            ax1.scatter(x,y2[:,0],color='green')
            ax2.scatter(x,y2[:,1],color='green')
            ax3.scatter(x,y2[:,2],color='green')
    if title is not None:
        fig.suptitle(title)
    if ylim is not None:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax3.set_ylim(ylim)
    if scat_ind is not None:
        ax1.scatter(x[scat_ind],y1[scat_ind,0])
        ax2.scatter(x[scat_ind],y1[scat_ind,1])
        ax3.scatter(x[scat_ind],y1[scat_ind,2])
        if y2 is not None:
            ax1.scatter(x[scat_ind],y2[scat_ind,0])
            ax2.scatter(x[scat_ind],y2[scat_ind,1])
            ax3.scatter(x[scat_ind],y2[scat_ind,2])
    return fig

def plot_4(x,y1,y2=None,scat_ind=None):
    y1 = quaternion.as_float_array(y1)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.set_title('qw')
    ax1.set_ylim(-1,1)
    ax2 = fig.add_subplot(222)
    ax2.set_title('qx')
    ax2.set_ylim(-1,1)
    ax3 = fig.add_subplot(223)
    ax3.set_title('qy')
    ax3.set_ylim(-1,1)
    ax4 = fig.add_subplot(224)
    ax4.set_title('qz')
    ax4.set_ylim(-1,1)
    ax1.plot(x,y1[:,0],color='red',linewidth=2)
    ax2.plot(x,y1[:,1],color='red',linewidth=2)
    ax3.plot(x,y1[:,2],color='red',linewidth=2)
    ax4.plot(x,y1[:,3],color='red',linewidth=2)
    if y2 is not None:
        y2 = quaternion.as_float_array(y2)
        ax1.plot(x,y2[:,0],color='green')
        ax2.plot(x,y2[:,1],color='green')
        ax3.plot(x,y2[:,2],color='green')
        ax4.plot(x,y2[:,3],color='green')
    if scat_ind is not None:
        ax1.scatter(x[scat_ind],y1[scat_ind,0])
        ax2.scatter(x[scat_ind],y1[scat_ind,1])
        ax3.scatter(x[scat_ind],y1[scat_ind,2])
        ax4.scatter(x[scat_ind],y1[scat_ind,3])

# Define a function to handle the selected region
def onselect(eclick, erelease):
    x1, x2 = eclick.xdata, erelease.xdata
    y1, y2 = eclick.ydata, erelease.ydata
    print(f'Selected region: x1={x1}, x2={x2}, y1={y1}, y2={y2}')
    return x1, x2

# with open('data/kiley_run1_left.npy','rb') as f:
#     pelvis = np.load(f)
#     thigh = np.load(f)
#     shank = np.load(f)
#     foot = np.load(f)
#     contra = np.load(f)
#     rt_fs = np.load(f)
#     joint_ang = np.load(f)


# data=[pelvis,thigh,shank,foot]
# foot = ang_vel_bias_correction(foot)
# shank = ang_vel_bias_correction(shank)
# new_shank, fs = Cut_trial(shank, rt_fs)
# pca= PCA(n_components=3)
# pca_comp1 = pca.components_
# pca.fit(new_shank[:,4:7])
# transformed_shank = pca.transform(new_shank[:,4:7])
# t1 = np.arange(0, len(new_shank)/100, 0.01)
# y1 = new_shank[:,21:24]
# y2 = transformed_shank


# theta = np.zeros((len(y2)-1, 3))

# theta[:,0] = integrate.cumtrapz(y2[:,0], t1)
# theta[:,1] = integrate.cumtrapz(y2[:,1], t1)
# theta[:,2] = integrate.cumtrapz(y2[:,2], t1)
# theta = np.degrees(theta)
# plt.close('all')
# # Angular Velocity FIgure 
# fig = plt.figure(figsize=(15,15))
# ax1 = fig.add_subplot(621)
# ax1.set_title('x-axis')
# ax1.tick_params(bottom=False,labelbottom=False)
# ax1.set_ylabel('Vertical Position (m)')
# ax2 = fig.add_subplot(623)
# ax2.set_title('y-axis')
# ax2.tick_params(bottom=False,labelbottom=False)
# ax2.set_ylabel('A/P Position (m)')
# ax3 = fig.add_subplot(625)
# ax3.set_title('z-axis')
# #ax3.tick_params(bottom=False,labelbottom=False)
# ax3.set_ylabel('Shank Angle (degrees')

# ax4 = fig.add_subplot(622)
# ax4.set_title('PC 1')
# ax4.tick_params(bottom=False,labelbottom=False)
# ax4.set_ylabel('Shank Angular Velocity')
# ax5 = fig.add_subplot(624)
# ax5.set_title('PC 2')
# ax5.tick_params(bottom=False,labelbottom=False)
# ax5.set_ylabel('Shank Angular Velocity')
# ax6 = fig.add_subplot(626)
# ax6.set_title('PC 3')
# #ax3.tick_params(bottom=False,labelbottom=False)
# ax6.set_ylabel('Shank Angular Velocity')


# ax1.plot(t1,y1[:,0],color='red')
# ax2.plot(t1,y1[:,1],color='red')
# ax3.plot(t1,y1[:,2],color='red')

# ax4.plot(t1,y2[:,0],color='green')
# ax5.plot(t1,y2[:,1],color='green')
# ax6.plot(t1,y2[:,2],color='green')


# # Angle from PCA Figure 
# fig = plt.figure(figsize=(15,15))
# ax1 = fig.add_subplot(311)
# ax1.set_title('x-axis')
# ax1.tick_params(bottom=False,labelbottom=False)
# ax1.set_ylabel('Vertical Position (m)')
# ax2 = fig.add_subplot(312)
# ax2.set_title('y-axis')
# ax2.tick_params(bottom=False,labelbottom=False)
# ax2.set_ylabel('A/P Position (m)')
# ax3 = fig.add_subplot(313)
# ax3.set_title('z-axis')
# #ax3.tick_params(bottom=False,labelbottom=False)
# ax3.set_ylabel('Shank Angle (degrees')
# t1 = np.arange(0, (len(new_shank)-1)/100, 0.01)
# ax1.plot(t1,theta[:,0],color='red')
# ax2.plot(t1,theta[:,1],color='red')
# ax3.plot(t1,theta[:,2],color='red')




# new_seg, fs = Cut_trial(shank, rt_fs)
# new_seg, fs = Cut_trial(shank, rt_fs)
# y_pos, norm_gc, LLTE_Vals = calc_LLTE_Components(new_seg, fs)
# data=[pelvis,thigh,shank,foot]
# foot = ang_vel_bias_correction(foot)
# shank = ang_vel_bias_correction(shank)

# while True:
#     user_input = input("Enter 'yes' to continue or 'no' to exit: ").lower()  # Convert input to lowercase
    
#     if user_input == 'yes':

#         new_seg, fs = Cut_trial(shank, rt_fs)
#         y_pos, norm_gc, LLTE_Vals = calc_LLTE_Components(new_seg, fs)

#         df_gc = pd.DataFrame({'y_vertical':norm_gc[0,:],'y_ap':norm_gc[1,:], 'y_theta':norm_gc[2,:]})
#         df_LLTE = pd.DataFrame({'y_vertical':LLTE_Vals[0,:],'y_ap':LLTE_Vals[1,:], 'y_theta':LLTE_Vals[2,:]})
         
#         my_dict = {
#             'y_pos': y_pos, 
#             'norm_gc': norm_gc,
#             'LLTE_values': LLTE_Vals
#             }
        
#         fname = input('Enter name for file: ')
#         fname.replace('/','-')
#         fname.replace('\\','-')
#         # save data to file
#         fname1 = 'data/' + fname +'_norm_gc.csv'
#         df_gc.to_csv(fname1, index = False) 
#         fname2 = 'data/' + fname + '_LLTE.csv'
#         df_LLTE.to_csv(fname2, index = False) 
        
#         fname3 = 'data/' + fname + '_LLTE.pkl'
#         with open(fname3, 'wb') as file:
#             pickle.dump(my_dict, file)
            
#         print("Processing...")
#         plt.close('all')
    
#     elif user_input == 'no':
#         print("Exiting the loop.")
#         plt.close('all')
#         break  # Exit the loop

#     else:
#         print("Invalid input. Please enter 'yes' or 'no'.")

# data_dict = UI_get_pkl()
# y_pos = data_dict['y_pos']
# norm_gait_cycle= data_dict['norm_gc']
# LLTE_vals = data_dict['LLTE_values']

# # df_crouch = pd.read_csv('data/sd_crouch_LLTE.csv')
# # df_stiff = pd.read_csv('data/sd_stiff_leg_LLTE.csv')
# # df_0_50 = pd.read_csv('data/sd_walk_0_50_LLTE.csv')
# # df_1_00 = pd.read_csv('data/sd_walk_1_00_LLTE.csv')
# # df_1_25 = pd.read_csv('data/sd_walk_1_25_LLTE.csv')
# # df_1_50 = pd.read_csv('data/sd_walk_1_50_LLTE.csv')


# df_crouch = pd.read_csv('data/ka_crouch_LLTE.csv')
# df_stiff = pd.read_csv('data/ka_stiff_leg_LLTE.csv')
# df_0_50 = pd.read_csv('data/ka_085_walking_LLTE.csv')
# df_1_00 = pd.read_csv('data/ka_100_walking_LLTE.csv')
# df_1_25 = pd.read_csv('data/ka_125_walking_LLTE.csv')
# df_1_50 = pd.read_csv('data/ka_150_walking_LLTE.csv')

# Crouch_LLTE = df_crouch.to_numpy()
# Stiff_LLTE = df_stiff.to_numpy()
# walk_050_LLTE = df_0_50.to_numpy()
# walk_100_LLTE = df_1_00.to_numpy()
# walk_125_LLTE = df_1_25.to_numpy()
# walk_150_LLTE = df_1_50.to_numpy()


# # Calculate LLTE 
# crouch_LLTE_val = LLTE_calc(Crouch_LLTE, y_pos)
# stiff_LLTE_val = LLTE_calc(Stiff_LLTE, y_pos)
# walking_LLTE_val = LLTE_calc(LLTE_vals.T, y_pos)

# walk_050_LLTE_val = LLTE_calc(walk_050_LLTE, y_pos)
# walk_100_LLTE_val = LLTE_calc(walk_100_LLTE, y_pos)
# walk_125_LLTE_val = LLTE_calc(walk_125_LLTE, y_pos)
# walk_150_LLTE_val = LLTE_calc(walk_150_LLTE, y_pos)

        
# plt.close('all')
# plot_LLTE_Components(Crouch_LLTE, LLTE_vals, y_pos)
# plot_LLTE_Components(Stiff_LLTE, LLTE_vals, y_pos)
# plot_LLTE_Components(walk_050_LLTE, LLTE_vals, y_pos)
# plot_LLTE_Components(walk_100_LLTE, LLTE_vals, y_pos)
# plot_LLTE_Components(walk_125_LLTE, LLTE_vals, y_pos)
# plot_LLTE_Components(walk_150_LLTE, LLTE_vals, y_pos)



# num_groups = 7
#  # Sample data
# data = {
#      'Average Walking LLTE ': walking_LLTE_val,
#      'Average Crouch Gait LLTE': crouch_LLTE_val,
#      'Average Stiff Leg LLTE': stiff_LLTE_val,
#      'Average Walking 050 LLTE ': walk_050_LLTE_val,
#      'Average Walking 100 LLTE': walk_100_LLTE_val,
#      'Average Walking 125 LLTE': walk_125_LLTE_val,
#      'Average Walking 150 LLTE': walk_150_LLTE_val
# }

# Barplot_groups(data, num_groups)


 

# with open('data/seth_part1_right.npy','rb') as f:
#     pelvis = np.load(f)
#     thigh = np.load(f)
#     shank = np.load(f)
#     foot = np.load(f)
#     contra = np.load(f)
#     rt_fs = np.load(f)
#     joint_ang = np.load(f)

# data=[pelvis,thigh,shank,foot]
# foot = ang_vel_bias_correction(foot)
# shank = ang_vel_bias_correction(shank)
# # new_seg, fs = Cut_trial(shank, rt_fs)
# new_seg, fs = Cut_trial(shank, rt_fs)
# # Identify Gait Cycles and norm_gc
# y_norm, norm_gc, LLTE_Vals = calc_LLTE_Components(new_seg, fs)

