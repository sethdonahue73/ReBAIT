import pickle
from time import sleep
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from DataCollector import DataCollector
from playback import playback
import tkinter as tk
from tkinter import filedialog

def select_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory()  # Open the file dialog
    return directory


# user select directory where the pickle files are located



loc = ['Pelvis','RThigh','RShank','RFoot', 'LThigh', 'LShank', 'LFoot', 'Torso']
pickle_directory = select_directory()
print(f"Selected directory: {pickle_directory}")
# recovery_file = 'recovery/trial_10_target_01.pkl'
llte_target_file = 'targets/archive/seth_080_llte_target.pkl'
# save_file = 'data/ab_imu_05_recovery.nc'
file_list = os.listdir(pickle_directory)
pickle_files = [file for file in file_list if file.endswith(".pkl") and "trial" in file]

trial_names = {i: file.split('.')[0] for i, file in enumerate(pickle_files)}
for recovery_file in pickle_files:
    with open(os.path.join(pickle_directory, pickle_files[-1]), 'rb') as f:
        data = pickle.load(f)
    devIds = list(data[0].keys())
    time = data[0]
    accSer = data[1]
    gyrSer = data[2]
    qutSer = data[3]
    zupSer = data[4]
    calAcc = data[5]
    eulAng = data[6]

    data_list = []
    for i in range(min(len(devIds), 8)):
        did = devIds[i]
        data = np.hstack([accSer[did], gyrSer[did], qutSer[did], np.array(zupSer[did])[:, None], calAcc[did], eulAng[did]])
        time_samp = time[did]
        da = xr.DataArray(data, coords=[time_samp, ['FreeAccX', 'FreeAccY', 'FreeAccZ', 'AngVelX', 'AngVelY', 'AngVelZ', 'QuaternionW', 'QuaternionX', 'QuaternionY', 'QuaternionZ', 'ZeroVelUpdate', 'CalAccX', 'CalAccY', 'CalAccZ', 'Pitch', 'Yaw', 'Roll']], dims=['time', 'Data'], name=loc[i])
        data_list.append(da)
    ds = xr.merge(data_list)

    plt.plot(time_samp, data[:, 0])
    timestamps = plt.ginput(-1)
    plt.close('all')
    cal_time = timestamps[0][0]
    trial_stamps = []
    for i in range(1, len(timestamps)):
        trial_stamps.append([trial_names[i - 1], timestamps[i - 1][0], timestamps[i][0]])

    P = playback(ds, loc, cal_time)

    foot_len = 0.31
    leg_len = 0.49
    imu_to_knee = 0.13
    anthroData = [foot_len, leg_len, imu_to_knee]

    with open(llte_target_file, 'rb') as file:
        llte_tar = np.load(file)

    dc = DataCollector(P, P.devIds, anthroData, llte_tar)

    dc.startCalibration()

    while P.packets[0].estimatedTimeOfSampling().secTime() < cal_time:
        sleep(1)

    dc.stopCalibration()

    dc.startCollection()

    while len(P.packets) > 0:
        print('X', end='')
        sleep(10)

    dc.stopCollection()

    data_list = []
    for i in range(min(len(P.devIds), 8)):
        did = P.devIds[i]
        data = np.hstack([dc.accSer[did], dc.gyrSer[did], dc.qutSer[did], np.array(dc.zupSer[did])[:, None], dc.pos[did], dc.vel[did], dc.calAcc[did], dc.eulAng[did]])
        time = dc.time[did]
        da = xr.DataArray(data, coords=[time, ['FreeAccX', 'FreeAccY', 'FreeAccZ', 'AngVelX', 'AngVelY', 'AngVelZ', 'QuaternionW', 'QuaternionX', 'QuaternionY', 'QuaternionZ', 'ZeroVelUpdate', 'PosX', 'PosY', 'PosZ', 'VelX', 'VelY', 'VelZ', 'CalAccX', 'CalAccY', 'CalAccZ', 'Pitch', 'Yaw', 'Roll']], dims=['time', 'Data'], name=loc[i])
        data_list.append(da)
    time = dc.time[P.devIds[2]]
    fs = np.zeros(len(time))
    fs[dc.fs] = 1
    fo = np.zeros(len(time))
    fo[dc.fo] = 1
    data = np.hstack([np.array(dc.joint_ang).T, np.array(fs)[:, None], np.array(fo)[:, None]])
    da = xr.DataArray(data, coords=[time, ['Hip', 'Knee', 'Ankle', 'IC', 'TO']], dims=['time', 'Data2'], name='CalculatedValues')
    data_list.append(da)
    ds = xr.merge(data_list)
    ds.attrs["cal_time"] = cal_time
    ds.attrs["trials"] = []
    for trial in trial_stamps:
        ds.attrs["trials"].append(trial[0])
        ds.attrs[trial[0]] = [trial[1], trial[2]]
    ds.attrs["LLTE_Vals"] = dc.llte_val
    ds.attrs["foot_len"] = dc.foot_length
    ds.attrs["leg_len"] = dc.leg_length
    ds.attrs["imu_to_knee"] = dc.imu2knee
    ds.to_netcdf(save_file)
# with open(recovery_file,'rb') as f:
#     data = pickle.load(f)
# devIds = list(data[0].keys())
# time = data[0]
# accSer = data[1]
# gyrSer = data[2]
# qutSer = data[3]
# zupSer = data[4]
# calAcc = data[5]
# eulAng = data[6]

# data_list = []
# for i in range(min(len(devIds),8)):
#     did = devIds[i]
#     data = np.hstack([accSer[did],gyrSer[did],qutSer[did],np.array(zupSer[did])[:,None],calAcc[did],eulAng[did]])
#     time_samp = time[did]
#     da = xr.DataArray(data,coords=[time_samp,['FreeAccX', 'FreeAccY', 'FreeAccZ','AngVelX', 'AngVelY', 'AngVelZ','QuaternionW', 'QuaternionX', 'QuaternionY', 'QuaternionZ','ZeroVelUpdate','CalAccX', 'CalAccY', 'CalAccZ','Pitch', 'Yaw', 'Roll']],dims=['time','Data'],name=loc[i])
#     data_list.append(da)
# ds = xr.merge(data_list)

# trial_names = {0:'trial06_target00',1:'trial07_target01',2:'Do Not Use',3:'trial08_target00',4:'trial09_target00',5:'trial10_target01'}

# plt.plot(time_samp,data[:,0])
# timestamps = plt.ginput(-1)
# plt.close('all')
# cal_time = timestamps[0][0]
# trial_stamps = []
# for i in range(1,len(timestamps)):
#      trial_stamps.append([trial_names[i-1],timestamps[i-1][0],timestamps[i][0]])

# P = playback(ds,loc,cal_time)

# foot_len = 0.31
# leg_len = 0.49
# imu_to_knee = 0.13
# anthroData = [foot_len,leg_len,imu_to_knee]

# with open(llte_target_file, 'rb') as file:
#         llte_tar = np.load(file)

# dc = dataCollector(P,P.devIds,anthroData,llte_tar)

# dc.startCalibration()

# while P.packets[0].estimatedTimeOfSampling().secTime() < cal_time:
#     sleep(1)

# dc.stopCalibration()

# dc.startCollection()

# while len(P.packets) > 0:
#     print('X',end='')
#     sleep(10)

# dc.stopCollection()

# data_list = []
# for i in range(min(len(P.devIds),8)):
#     did = P.devIds[i]
#     data = np.hstack([dc.accSer[did],dc.gyrSer[did],dc.qutSer[did],np.array(dc.zupSer[did])[:,None],dc.pos[did],dc.vel[did],dc.calAcc[did],dc.eulAng[did]])
#     time = dc.time[did]
#     da = xr.DataArray(data,coords=[time,['FreeAccX', 'FreeAccY', 'FreeAccZ','AngVelX', 'AngVelY', 'AngVelZ','QuaternionW', 'QuaternionX', 'QuaternionY', 'QuaternionZ','ZeroVelUpdate','PosX', 'PosY', 'PosZ','VelX', 'VelY', 'VelZ','CalAccX', 'CalAccY', 'CalAccZ','Pitch', 'Yaw', 'Roll']],dims=['time','Data'],name=loc[i])
#     data_list.append(da)
# time = dc.time[P.devIds[2]]
# fs = np.zeros(len(time))
# fs[dc.fs] = 1
# fo = np.zeros(len(time))
# fo[dc.fo] = 1
# data = np.hstack([np.array(dc.joint_ang).T,np.array(fs)[:,None],np.array(fo)[:,None]])
# da = xr.DataArray(data,coords=[time,['Hip', 'Knee', 'Ankle','IC','TO']],dims=['time','Data2'],name='CalculatedValues')
# data_list.append(da)
# ds = xr.merge(data_list)
# ds.attrs["cal_time"] = cal_time
# ds.attrs["trials"]=[]
# for trial in trial_stamps:
#     ds.attrs["trials"].append(trial[0])
#     ds.attrs[trial[0]] = [trial[1],trial[2]]
# ds.attrs["LLTE_Vals"] = dc.llte_val
# ds.attrs["foot_len"] = dc.foot_length
# ds.attrs["leg_len"] = dc.leg_length
# ds.attrs["imu_to_knee"] = dc.imu2knee
# ds.to_netcdf(save_file)