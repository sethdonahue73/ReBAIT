from time import sleep
import numpy as np
import quaternion
import xarray as xr
from DataCollector import DataCollector
from new_utils import my_resample
from playback import playback
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from utils import plot_3

replay_file = 'data/prealloc-test.nc'
llte_target = 'targets/Speeds_wn_080.pkl'
def replay_func(replay_file,llte_target_file):
    plt.close('all')
    ds = xr.open_dataset(replay_file)
    devIds = [id for id in list(ds.data_vars) if id != 'CalculatedValues']
    cal_time = ds.attrs['cal_time']
    P = playback(ds,devIds,cal_time)

    if "foot_len" in ds.attrs:
        foot_len = ds.attrs['foot_len']
    else:
        foot_len = 0.35
    if "leg_len" in ds.attrs:
        leg_len = ds.attrs['leg_len']
    else:
        leg_len = 0.5
    if "imu_to_knee" in ds.attrs:
        imu_to_knee = ds.attrs['imu_to_knee']
    else:
        imu_to_knee = 0.15
    anthroData = [foot_len,leg_len,imu_to_knee]
        
    with open(llte_target_file, 'rb') as file:
            llte_tar = np.load(file)
            # t= np.arange(0,101,1)
            # plot_3(t, llte_tar.T)

    
    dc = DataCollector(P,devIds,anthroData,llte_tar)

    dc.startCalibration()

    while P.packets[0].estimatedTimeOfSampling().secTime() < cal_time:
        sleep(1)

    dc.stopCalibration()

    
    class HighlightArea(pg.GraphicsWidget):
        def __init__(self, y1, y2, color):
            super().__init__()
            self.y1 = y1
            self.y2 = y2
            self.color = color

        def paint(self, p, *args):
            p.setPen(pg.mkPen(None))
            p.setBrush(pg.mkBrush(self.color))
            vr = self.viewRect()
            p.drawRect(QtCore.QRectF(vr.left(), self.y1, vr.width(), self.y2 - self.y1))


    class MainWindow(QtWidgets.QMainWindow):

        def __init__(self, *args, **kwargs):
            super(MainWindow, self).__init__(*args, **kwargs)

            self.graphWidget = pg.PlotWidget()
            self.setCentralWidget(self.graphWidget)

            self.x = [0,100]
            self.y = [0 for _ in range(5)]  # 5 data points
            
            self.color_1 = 'green'
            self.color_2 = 'yellow'
            self.color_3 = 'gray'

            highlight_1 = HighlightArea(-1.5, 0.1,  self.color_1)
            highlight_2 = HighlightArea(-4.0, -1.5,  self.color_2)
            highlight_3 = HighlightArea(-8.0,-4.0,  self.color_3)

            self.graphWidget.setBackground('w')
            # self.graphWidget.setYRange(-8,0.0, padding = 0.0)
            self.graphWidget.setXRange(0,100, padding = 0.0)
            # Add the highlight area
            # self.graphWidget.addItem(highlight_1, ignoreBounds = False)
            # highlight_1.setZValue(-3)
            # self.graphWidget.addItem(highlight_2, ignoreBounds = False)
            # highlight_2.setZValue(-2)
            # self.graphWidget.addItem(highlight_3, ignoreBounds = False)
            # highlight_3.setZValue(-1)


            pen1 = pg.mkPen(color=(0, 0, 255,  25), width=6)
            pen2 = pg.mkPen(color=(0, 0, 255,  50), width=6)
            pen3 = pg.mkPen(color=(0, 0, 255,  100), width=6)
            pen4 = pg.mkPen(color=(0, 0, 255, 150), width=6)
            pen5 = pg.mkPen(color=(0, 0, 255, 255), width=6)
            pen_tar = pg.mkPen(color=(0, 0, 0), width=32)
            
            self.data_line1 =  self.graphWidget.plot([self.y[0]], pen=pen1)
            self.data_line2 =  self.graphWidget.plot([self.y[1]], pen=pen2)
            self.data_line3 =  self.graphWidget.plot([self.y[2]], pen=pen3)
            self.data_line4 =  self.graphWidget.plot([self.y[3]], pen=pen4)
            self.data_line5 =  self.graphWidget.plot([self.y[4]], pen=pen5)
            self.data_line_tar = self.graphWidget.plot([0], pen=pen_tar)

            self.timer = QtCore.QTimer()
            self.timer.setInterval(50)
            self.timer.timeout.connect(self.update_plot_data)
            self.timer.start()

        def update_plot_data(self):
            if len(dc.llte_val) >= 5:
                try: # sometimes a foot strike is detected before position integration calc is completed, so there is no value to access and an index exception is raised. Just skip this frame update and try again next frame
                    self.y = dc.llte_val[-5:] # plots total distance walked
                    # min_y = min(self.y)
                    # if min_y < -4.0:
                    #     self.graphWidget.setYRange(-8,0, padding = 0.0)
                    # elif min_y < -1.5:
                    #     self.graphWidget.setYRange(-4,0, padding = 0.0)
                    # else: 
                    #     self.graphWidget.setYRange(-1.5,0, padding = 0.0)
                    
                    self.data_line1.setData(self.x, [-self.y[0],-self.y[0]])  # Update the data.
                    self.data_line2.setData(self.x, [-self.y[1],-self.y[1]])  # Update the data.
                    self.data_line3.setData(self.x, [-self.y[2],-self.y[2]])  # Update the data.
                    self.data_line4.setData(self.x, [-self.y[3],-self.y[3]])  # Update the data.
                    self.data_line5.setData(self.x, [-self.y[4],-self.y[4]])  # Update the data.
                    self.update()
                except:
                    pass

        def reset_plot(self):
            self.y = [0 for _ in self.y]
            self.data_line1.setData(self.x, [self.y[0],self.y[0]])  # Reset the data.
            self.data_line2.setData(self.x, [self.y[1],self.y[1]])  # Reset the data.
            self.data_line3.setData(self.x, [self.y[2],self.y[2]])  # Reset the data.
            self.data_line4.setData(self.x, [self.y[3],self.y[3]])  # Reset the data.
            self.data_line5.setData(self.x, [self.y[4],self.y[4]])  # Reset the data.
            self.data_line_tar.setData(self.x, [0,0])
        
    app = QtWidgets.QApplication([])
    w = MainWindow()

    w.reset_plot()

    dc.startCollection()

    w.show()
    app.exec_()

    while len(P.packets) > 0:
        sleep(1)

    dc.stopCollection()

    data_list = []
    loc = ['Pelvis','RThigh','RShank','RFoot', 'LThigh', 'LShank', 'LFoot', 'Torso']
    for i in range(min(len(devIds),8)):
        did = devIds[i]
        data = np.hstack([dc.accSer[did],dc.gyrSer[did],dc.qutSer[did],dc.zupSer[did][:,None],dc.pos[did],dc.vel[did],dc.calAcc[did],dc.eulAng[did]])
        time = dc.time[did][:dc.data_idx[did]]
        da = xr.DataArray(data[:dc.data_idx[did]],coords=[time,['FreeAccX', 'FreeAccY', 'FreeAccZ','AngVelX', 'AngVelY', 'AngVelZ','QuaternionW', 'QuaternionX', 'QuaternionY', 'QuaternionZ','ZeroVelUpdate','PosX', 'PosY', 'PosZ','VelX', 'VelY', 'VelZ','CalAccX', 'CalAccY', 'CalAccZ','Pitch', 'Yaw', 'Roll']],dims=['time','Data'],name=loc[i])
        data_list.append(da)

        # totalPackets = round((dc.time[did][-1] - dc.time[did][0]) * 120) + 1
        # dropped = (totalPackets - len(dc.time[did])) / totalPackets
        # if dropped > .01:
        #     print(f'device {did} dropped {dropped:.0%} of packets')
    time = dc.time[devIds[2]][:dc.data_idx[devIds[2]]]
    fs = np.zeros(len(time))
    fs[dc.fs] = 1
    fo = np.zeros(len(time))
    fo[dc.fo] = 1
    data = np.hstack([np.array(dc.joint_ang).T,np.array(fs)[:,None],np.array(fo)[:,None]])
    da = xr.DataArray(data,coords=[time,['Hip', 'Knee', 'Ankle','IC','TO']],dims=['time','Data2'],name='CalculatedValues')
    data_list.append(da)
    ds_new = xr.merge(data_list)
    ds_new.attrs["cal_time"] = cal_time
    ds_new.attrs["trials"]=[]
    if type(ds.attrs['trials']) == list:
        trial_stamps = [[trial, ds.attrs[trial][0], ds.attrs[trial][1]] for trial in ds.attrs["trials"]]
    else:
        trial_stamps = [[ds.attrs["trials"], ds.attrs[ds.attrs["trials"]][0], ds.attrs[ds.attrs["trials"]][1]]]
    for trial in trial_stamps:
        ds_new.attrs["trials"].append(trial[0])
        ds_new.attrs[trial[0]] = [trial[1],trial[2]]
    ds_new.attrs["LLTE_Vals"] = dc.llte_val
    ds_new.attrs["foot_len"] = dc.foot_length
    ds_new.attrs["leg_len"] = dc.leg_length
    ds_new.attrs["imu_to_knee"] = dc.imu2knee
    user_input = input("Press Enter to save temp file and confirm data format. Type file name to save replayed data.")
    if user_input:
        user_input.replace('/','-')
        user_input.replace('\\','-')
        # save data to file
        user_input = 'data/' + user_input + '.nc'
        print(f'Saving data to file {user_input}')
        ds_new.to_netcdf(user_input)
    else:
        ds_new.to_netcdf('temp.nc')

    plt.figure()
    plt.plot(dc.llte_val)

    fig = plot_3(range(len(dc.llte_parts[0])),np.array(dc.llte_parts).T,xlabel='Step Count',scatter=True,ylim=(0,0.5))
    plt.tight_layout()
    fig.savefig('img/080_150_llte.tif')

    print(f'{np.mean(dc.llte_val[:-7])}, {np.std(dc.llte_val[::-7])}')

    # Transform time_stamp to an index 

    hip = []
    knee = []
    ankle = []
    for i in range(2, len(dc.fs)-2):
        if dc.fs[i+1]-dc.fs[i] > 200:
            continue
        t = dc.time[devIds[2]][dc.fs[i]:dc.fs[i+1]]
        # print(len(t))
        hip.append(my_resample(101,dc.joint_ang[0][dc.fs[i]:dc.fs[i+1]],t))
        knee.append(my_resample(101,dc.joint_ang[1][dc.fs[i]:dc.fs[i+1]],t))
        ankle.append(my_resample(101,dc.joint_ang[2][dc.fs[i]:dc.fs[i+1]],t))

    hip = np.mean(np.array(hip),axis=0)
    knee = np.mean(np.array(knee),axis=0)
    ankle = np.mean(np.array(ankle),axis=0)

    # ang = np.array([hip,knee,ankle]).T

    # with open('080_sk_avg_ang.npy','wb') as f:
    #     np.save(f,ang)

    # plot_3(range(101),ang)
    # plt.show()

    # print('target', llte_tar[0,0],llte_tar[0,-1],llte_tar[1,0],llte_tar[1,-1],llte_tar[2,0],llte_tar[2,-1])

    # print(ds_new)
    with open('080_avg_ang.npy','rb') as f:
        ang_080 = np.load(f)
    fs_title = 24
    fs_ylabel = 20
    fs_tick = 14
    t = dc.time[devIds[2]]
    if len(dc.fs) >= 7:
        # plot joint angles during each gait cycle, normalized from 0%-100%
        fig = plt.figure(figsize=(6.4,9.6))
        # plt.suptitle('0.80 m/s Stiff Knee Gait')
        ax1 = fig.add_subplot(311)
        ax1.set_xlim(0,100)
        ax1.set_ylim(-35,100)
        ax1.set_title('Hip', fontsize= fs_title)
        ax1.set_ylabel('Angle (deg)', fontsize= fs_ylabel)
        ax1.set_xticks([])
        ax1.tick_params(axis='both', which='major', labelsize=fs_tick)
        ax2 = fig.add_subplot(312)
        ax2.set_xlim(0,100)
        ax2.set_ylim(-35,100)
        ax2.set_title('Knee', fontsize= fs_title)
        ax2.set_ylabel('Angle (deg)', fontsize= fs_ylabel)
        ax2.set_xticks([])
        ax2.tick_params(axis='both', which='major', labelsize=fs_tick)
        ax3 = fig.add_subplot(313)
        ax3.set_xlim(0,100)
        ax3.set_ylim(-35,100)
        ax3.set_title('Ankle', fontsize= fs_title)
        ax3.set_ylabel('Angle (deg)', fontsize= fs_ylabel)
        ax3.set_xlabel('Gait Cycle (%)', fontsize= fs_ylabel)
        ax3.tick_params(axis='both', which='major', labelsize=fs_tick)
        for i in range(0,len(dc.fs)-7): # remove first 4 and last 2 gait cycles from plot; typically very noisy as subject is speeding up and slowing down
            if dc.fs[i+1]-dc.fs[i] > 160:
                print(dc.fs[i+1]-dc.fs[i])
                continue
            y_h=np.array(dc.joint_ang[0][dc.fs[i]:dc.fs[i+1]])
            y_k=np.array(dc.joint_ang[1][dc.fs[i]:dc.fs[i+1]])
            y_a=np.array(dc.joint_ang[2][dc.fs[i]:dc.fs[i+1]])
            x=np.array(t[dc.fs[i]:dc.fs[i+1]])
            x=x-x[0]
            x=x*100/(x[-1])
            ax1.plot(x,y_h, color='black', zorder = -1, alpha = 0.1)
            ax2.plot(x,y_k, color='black', zorder = -1, alpha = 0.1)
            ax3.plot(x,y_a, color='black', zorder = -1, alpha = 0.1)
            if i == len(dc.fs)-10:
                ax1.plot(x,y_h, zorder = -1, alpha = 0.5, label = 'Gait Cycle Trajectory')
        x = range(101)
        ax1.plot(x, hip, linewidth = 3, color = 'red', zorder = 1, label = 'Average Trajectory')
        ax2.plot(x, knee, linewidth = 3, color = 'red', zorder = 1)
        ax3.plot(x, ankle, linewidth = 3, color = 'red', zorder = 1)
        ax1.plot(x, ang_080[:,0], linewidth = 5, color = 'orange', label='0.80 m/s Typical Gait')
        ax2.plot(x, ang_080[:,1], linewidth = 5, color = 'orange')
        ax3.plot(x, ang_080[:,2], linewidth = 5, color = 'orange')
        ax1.legend()
        plt.tight_layout()
        fig.savefig('img/080_150_joint_angles.tif')
   
    if len(dc.fs) >= 7:
        # plot joint angles during each gait cycle, normalized from 0%-100%
        fig = plt.figure(figsize=(6.4,9.6))
        # plt.suptitle('0.80 m/s Crouch gait', fontsize= fs_title )
        ax1 = fig.add_subplot(311)
        ax1.tick_params(axis='both', which='major', labelsize=fs_tick)
        ax1.set_xticks([])
        ax1.set_xlim(0,100)#(0,100)
        ax1.set_ylabel('Position (m)', fontsize= fs_ylabel)
        # ax1.set_ylim(-0.1,0.6)
        ax1.set_title('Knee Anterior/Posterior Position', fontsize= fs_title)
        ax2 = fig.add_subplot(312)
        ax2.tick_params(axis='both', which='major', labelsize=fs_tick)
        ax2.set_xticks([])
        ax2.set_xlim(0,100)#(0,100)
        ax2.set_ylim(0.8,1.10)
        ax2.set_ylabel('Position (m)', fontsize= fs_ylabel)
        # ax2.set_ylim(-0.1,0.6)
        ax2.set_title('Knee Superior/Inferior Position', fontsize= fs_title)
        ax3 = fig.add_subplot(313)
        ax3.tick_params(axis='both', which='major', labelsize=fs_tick)
        ax3.set_xlim(0,100)#(0,100)
        ax3.set_xlabel('Stance Phase (%)', fontsize= fs_ylabel)
        ax3.set_ylabel('Angle (deg)', fontsize= fs_ylabel)
        # ax3.set_ylim(-10,40)
        ax3.set_title('Shank Angle', fontsize= fs_title)
        pos = np.array(dc.pos[devIds[2]])
        leg_theta_anatomical = []
        samp_theta = quaternion.as_float_array(dc.q_cal[devIds[2]][0] * quaternion.from_float_array(dc.qutSer[devIds[2]]) * dc.q_cal[devIds[2]][1])
        for quat in samp_theta:
            w,x,y,z = quat
            shank_ang = np.arctan2(2.0* ( w*z + x*y), 1.0-2.0*(y**2+z**2))
            leg_theta_anatomical.append(shank_ang)
        for i in range(5,len(dc.fo)-7): # remove first and last several gait cycles from plot; typically very noisy as subject is speeding up and slowing down
            if dc.fs[i+1]-dc.fs[i] > 130:
                print(dc.fs[i+1]-dc.fs[i])
                continue
            y_x = -np.array(pos[dc.fs[i]:dc.fo[i],0])
            y_x += dc.leg_length*np.sin(leg_theta_anatomical[dc.fs[i]]) - y_x[0]
            y_x *= dc.leg_length/(dc.leg_length-dc.imu2knee)
            y_y = np.array(pos[dc.fs[i]:dc.fo[i],1])
            y_t = np.array(leg_theta_anatomical[dc.fs[i]:dc.fo[i]])*180/np.pi
            x=np.array(t[dc.fs[i]:dc.fo[i]])
            x=x-x[0]
            x=x*100/(x[-1])
            ax1.plot(x,y_x/dc.leg_length,color='black',zorder=-1, alpha = 0.1)
            ax2.plot(x,y_y/dc.leg_length,color='black',zorder=-1, alpha = 0.1)
            ax3.plot(x,y_t/(np.arctan(dc.leg_length/dc.foot_length)*180/np.pi),color='black',zorder=-1, alpha = 0.1)
            if i == len(dc.fo)-10:
                ax1.plot(x,y_x/dc.leg_length,zorder=-1,color='black', alpha = 0.5, label = 'Gait Cylce Trajectory')

        x=range(101)
        ax1.plot(x,llte_tar.T[:,0],color='orange',linewidth=5,zorder=1,label='LLTE Target')
        ax2.plot(x,llte_tar.T[:,1],color='orange',linewidth=5, zorder=1)
        ax3.plot(x,llte_tar.T[:,2],color='orange',linewidth=5,zorder=1)
        ax1.legend()
        plt.tight_layout()
        plt.savefig('img/080_150_stance_llte.tif')
        # plt.show()
            

    # plot_3(np.array(t),np.array(dc.pos['RShank']),scat_ind=dc.mst)
    # plot_3(ds['time'],ds['RShank'].loc[:,['PosX','PosY','PosZ']])
    # plt.figure()
    # plt.plot(t,np.array(dc.pos['RShank'])[:,1])
    # plt.scatter(np.array(t)[dc.fs],np.array(dc.pos['RShank'])[dc.fs,1])
    # plt.scatter(np.array(t)[dc.fo],np.array(dc.pos['RShank'])[dc.fo,1])

    return dc

if __name__ == '__main__':
    replay_func(replay_file,llte_target)
    plt.show()
