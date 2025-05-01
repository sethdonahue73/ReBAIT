import threading
import time
from warnings import warn
import numpy as np
import quaternion
from scipy import integrate
from sklearn.decomposition import PCA
import xarray as xr

from new_utils import my_resample

class DataCollector:
    def __init__(self,callback,devIds,imu_locs,anthroData,llte_tar,A_LIM=1,W_LIM=45*np.pi/180):
        self.callback=callback
        self.devIds=devIds
        self.imu_locs = imu_locs
        self.foot_length = anthroData['Foot Length']
        self.leg_length = anthroData['Leg Length']
        self.imu2knee = anthroData['IMU to KJC']
        self.llte_tar = llte_tar
        self.A_LIM=A_LIM
        self.W_LIM=W_LIM
        self.prealloc_secs = 60 * 60  # number of seconds of data collection to preallocate array size
        self.samp_freq = callback.samp_freq  # Hz
        self.reset()

    def reset(self):
        # raw data from IMUs
        self.time = {devId: np.ones(self.prealloc_secs*self.samp_freq) * -1 for devId in self.devIds}
        self.accSer = {devId: np.zeros(((self.prealloc_secs*self.samp_freq), 3)) for devId in self.devIds}
        self.gyrSer = {devId: np.zeros(((self.prealloc_secs*self.samp_freq), 3)) for devId in self.devIds}
        self.qutSer = {devId: np.zeros(((self.prealloc_secs*self.samp_freq), 4)) for devId in self.devIds}
        self.zupSer = {devId: np.ones(self.prealloc_secs*self.samp_freq) * -1 for devId in self.devIds} # Boolean for is this index a ZUPT undex 1 when non-zero, 0 when there is a zero-velocity
        self.calAcc = {devId: np.zeros(((self.prealloc_secs*self.samp_freq), 3)) for devId in self.devIds}
        self.eulAng = {devId: np.zeros(((self.prealloc_secs*self.samp_freq), 3)) for devId in self.devIds}
        self.data_idx = {devId: 0 for devId in self.devIds}

        # calculated values
        self.zupInd = {devId:[0,-1] for devId in self.devIds} # Indecies of the current chunk to be processed 
        self.pos = {devId: np.zeros(((self.prealloc_secs*self.samp_freq), 3)) for devId in self.devIds}
        self.vel = {devId: np.zeros(((self.prealloc_secs*self.samp_freq), 3)) for devId in self.devIds}
        self.calc_idx = {devId: 0 for devId in self.devIds}
        self.q_cal = {devId:[quaternion.as_quat_array([1,0,0,0]),quaternion.as_quat_array([1,0,0,0])] for devId in self.devIds}
        self.integration_flag = {devId:True for devId in self.devIds}
        self.llte_val = []  # stores overall error value for each step
        self.llte_parts = [[],[],[]]
        self.llte_mean = []
        self.joint_ang = [[],[],[]]
        self.msw = []
        self.fs = []
        self.fo = []
        self.mst = []
        self.mst_flag = False
        self.fs_flag = False
        self.zero_cross = []
        self.cur_search = 1  # keep track of what event we are searching for. 1:midswing, 2:FS, 3:FO
        self.flag = False  # flag used in various aspects of event detection
        self.temp = -1
        self.w_temp = [0,0]

    def staticCalibrate(self,seconds=5):
        self.reset()
        startTime = time.time()
        while time.time() - startTime <= seconds:
            if self.callback.packetAvailable():
                # Retrieve a packet
                did,data = self.callback.getNextPacket()

                if data[0] not in self.time[did]:
                    self.time[did][self.data_idx[did]] = data[0]
                    self.accSer[did][self.data_idx[did]] = data[1]
                    self.gyrSer[did][self.data_idx[did]] = data[2]
                    self.qutSer[did][self.data_idx[did]] = data[3]
                    self.calAcc[did][self.data_idx[did]] = data[4]
                    self.eulAng[did][self.data_idx[did]] = data[5]

                    a_mag = np.linalg.norm(data[1])
                    w_mag = np.linalg.norm(data[2])
                    if w_mag < self.W_LIM and a_mag < self.A_LIM:
                        zup = 0
                    else:
                        zup = 1
                    self.zupSer[did][self.data_idx[did]] = zup
                    
                    self.data_idx[did] += 1

                else:
                    warn(f'Device {did} received packet with duplicate timestamp {data[0]}')

    def startCalibration(self):
        self.pill = threading.Event()  # create a poison pill that will be used to gracefully end data collection
        self.calibrate_thread = threading.Thread(target = self.dynamicCalibrate, args=(self.pill, ), name='Calibration')  # create Thread object, pass in pill
        self.calibrate_thread.start()  # start real-time data collection

    def stopCalibration(self):
        self.pill.set()  # trigger poison pill to signal to data collection thread to process any remaining data then exit
        self.calibrate_thread.join()  # wait for thread to finish processing before continuing
        del self.pill
        del self.calibrate_thread
        self.cal_time = self.time[self.devIds[2]][-1]

    def dynamicCalibrate(self,pill):
        while not pill.is_set():
            if self.callback.packetAvailable():
                # Retrieve a packet
                did,data = self.callback.getNextPacket()

                if data[0] not in self.time[did]:
                    self.time[did][self.data_idx[did]] = data[0]
                    self.accSer[did][self.data_idx[did]] = data[1]
                    self.gyrSer[did][self.data_idx[did]] = data[2]
                    self.qutSer[did][self.data_idx[did]] = data[3]
                    self.calAcc[did][self.data_idx[did]] = data[4]
                    self.eulAng[did][self.data_idx[did]] = data[5]

                    a_mag = np.linalg.norm(data[1])
                    w_mag = np.linalg.norm(data[2])
                    if w_mag < self.W_LIM and a_mag < self.A_LIM:
                        zup = 0
                    else:
                        zup = 1
                    self.zupSer[did][self.data_idx[did]] = zup
                    
                    self.data_idx[did] += 1

                    # update zup index for identifying chunks of data between zero-velocity occurances
                    if data[4] == 0:
                        if self.zupInd[did][1] >= 0:
                            self.zupInd[did][1] = self.data_idx[did]
                    elif self.zupInd[did][1] == -1:
                        self.zupInd[did][1] = 0

                else:
                    warn(f'Device {did} received packet with duplicate timestamp {data[0]}')

        for did in self.devIds:
            t=self.time[did][:self.data_idx[did]]-self.time[did][0]
            q_imu = quaternion.as_quat_array(self.qutSer[did][:self.data_idx[did]])
            a_world = quaternion.as_vector_part(q_imu * quaternion.from_vector_part(self.calAcc[did][:self.data_idx[did]]) * q_imu.conj())
            w_world = quaternion.as_vector_part(q_imu * quaternion.from_vector_part(self.gyrSer[did][:self.data_idx[did]]) * q_imu.conj())

            g_avg = np.mean(a_world[t<5,:],axis=0)
            g_ana = [0,9.81,0]

            n = np.cross(g_avg,g_ana)
            n = n/np.linalg.norm(n)
            theta = np.arccos(np.dot(g_avg,g_ana)/(np.linalg.norm(g_avg)*np.linalg.norm(g_ana)))
            q_g = quaternion.as_quat_array([np.cos(theta/2),np.sin(theta/2)*n[0],np.sin(theta/2)*n[1],np.sin(theta/2)*n[2]]).normalized()
            
            w_world = quaternion.rotate_vectors(q_g,w_world)

            pca = PCA(n_components=1)
            pca.fit(w_world)
            z_pca = pca.components_[0]
            rot_ax = np.cross(z_pca,[0,0,1])
            theta = np.arccos(np.dot(z_pca,[0,0,1]))
            q_pca = quaternion.as_quat_array([np.cos(theta/2),0,np.sin(theta/2)*np.sign(rot_ax[1]),0])

            q_f = q_pca * q_g  # q_f converts from imu global frame to global lab-fixed anatomical frame

            w_world = quaternion.rotate_vectors(q_pca, w_world)

            if max(w_world[:,2]) > -min(w_world[:,2]):
                q_f = quaternion.as_quat_array([0,0,1,0]) * q_f
                print('reversing x-axis')

            q_init = q_imu[t<5].conj() * q_f.conj()  # equivalent to conj(q_f*q_imu); gives the quaternions from global lab-fixed anatomical frame to local imu frame at each timestep
            q_0 = quaternion.from_float_array(np.mean(quaternion.as_float_array(q_init),axis=0))  # left mutliplication by q_0 gives orientation relative to starting orientation
            self.q_cal[did] = [q_f, q_0]  #  q_f * q_imu * q_0 gives the quaternion between the starting orientation and the current orientation for any timestep of q_imu

        for i in range(self.data_idx[self.devIds[2]]):
            q_pv_t = self.q_cal[self.devIds[0]][0] * quaternion.as_quat_array(self.qutSer[self.devIds[0]][min(i,self.data_idx[self.devIds[0]]-1)]) * self.q_cal[self.devIds[0]][1]
            q_th_t = (self.q_cal[self.devIds[1]][0] * quaternion.as_quat_array(self.qutSer[self.devIds[1]][min(i,self.data_idx[self.devIds[1]]-1)]) * self.q_cal[self.devIds[1]][1]).conj()
            q_sh_t = (self.q_cal[self.devIds[2]][0] * quaternion.as_quat_array(self.qutSer[self.devIds[2]][min(i,self.data_idx[self.devIds[2]]-1)]) * self.q_cal[self.devIds[2]][1]).conj()
            q_ft_t =  self.q_cal[self.devIds[3]][0] * quaternion.as_quat_array(self.qutSer[self.devIds[3]][min(i,self.data_idx[self.devIds[3]]-1)]) * self.q_cal[self.devIds[3]][1]
            
            q_h=quaternion.as_float_array(q_pv_t.conj()*q_th_t)
            q_k=quaternion.as_float_array(q_sh_t.conj()*q_th_t)
            q_a=quaternion.as_float_array(q_sh_t.conj()*q_ft_t)

            theta_h = 2*np.arccos(q_h[0]/np.sqrt(np.power(q_h[0],2)+np.power(q_h[3],2)))*np.sign(q_h[3])*180/np.pi
            theta_k = 2*np.arccos(q_k[0]/np.sqrt(np.power(q_k[0],2)+np.power(q_k[3],2)))*np.sign(q_k[3])*180/np.pi
            theta_a = 2*np.arccos(q_a[0]/np.sqrt(np.power(q_a[0],2)+np.power(q_a[3],2)))*np.sign(q_a[3])*180/np.pi

            self.joint_ang[0].append(theta_h)
            self.joint_ang[1].append(theta_k)
            self.joint_ang[2].append(theta_a)

    def startCollection(self):
        self.pill = threading.Event()  # create a poison pill that will be used to gracefully end data collection
        self.collection_thread = threading.Thread(target = self.run_collection, args=(self.pill, ), name = 'Collection')  # create Thread object, pass in pill
        self.collection_thread.start()  # start real-time data collection

    def stopCollection(self, trial_stamps):
        self.pill.set()  # trigger poison pill to signal to data collection thread to process any remaining data then exit
        self.collection_thread.join()  # wait for thread to finish processing before continuing
        del self.pill
        del self.collection_thread
        
        fname = "".join(filter(lambda x: str.isalnum(x) or str.isspace(x),input('Enter name for file: ')))
        # save data to file
        fname = 'data/' + fname + '.nc'
        print(f'Saving data to file {fname}')
        data_list = []
        
        for i in range(min(len(self.devIds),8)):
            did = self.devIds[i]
            data = np.hstack([self.accSer[did],self.gyrSer[did],self.qutSer[did],self.zupSer[did][:,None],self.pos[did],self.vel[did],self.calAcc[did],self.eulAng[did]])
            time = self.time[did][:self.data_idx[did]]
            da = xr.DataArray(data[:self.data_idx[did]],coords=[time,['FreeAccX', 'FreeAccY', 'FreeAccZ','AngVelX', 'AngVelY', 'AngVelZ','QuaternionW', 'QuaternionX', 'QuaternionY', 'QuaternionZ','ZeroVelUpdate','PosX', 'PosY', 'PosZ','VelX', 'VelY', 'VelZ','CalAccX', 'CalAccY', 'CalAccZ','Pitch', 'Yaw', 'Roll']],dims=['time','Data'],name=self.imu_locs[i])
            data_list.append(da)
        time = self.time[self.devIds[2]][:self.data_idx[self.devIds[2]]]
        fs = np.zeros(len(time))
        fs[self.fs] = 1
        fo = np.zeros(len(time))
        fo[self.fo] = 1
        data = np.hstack([np.array(self.joint_ang).T,np.array(fs)[:,None],np.array(fo)[:,None]])
        da = xr.DataArray(data,coords=[time,['Hip', 'Knee', 'Ankle','IC','TO']],dims=['time','Data2'],name='CalculatedValues')
        data_list.append(da)
        ds = xr.merge(data_list)
        ds.attrs["cal_time"] = self.cal_time
        ds.attrs["trials"]=[]
        for trial in trial_stamps:
            ds.attrs["trials"].append(trial[0])
            ds.attrs[trial[0]] = [trial[1],trial[2]]
        ds.attrs["LLTE_Vals"] = self.llte_val
        for key,val in self.anthro_data.items():
            ds.attrs[key] = val
        ds.to_netcdf(fname)

    def run_collection(self,pill):
        while not pill.is_set() or self.callback.packetAvailable():
            if self.callback.packetAvailable():
                # Retrieve a packet
                did,data = self.callback.getNextPacket()

                if data[0] not in self.time[did]:
                    self.time[did][self.data_idx[did]] = data[0]
                    self.accSer[did][self.data_idx[did]] = data[1]
                    self.gyrSer[did][self.data_idx[did]] = data[2]
                    self.qutSer[did][self.data_idx[did]] = data[3]
                    self.calAcc[did][self.data_idx[did]] = data[4]
                    self.eulAng[did][self.data_idx[did]] = data[5]

                    a_mag = np.linalg.norm(data[1])
                    w_mag = np.linalg.norm(data[2])
                    if w_mag < self.W_LIM and a_mag < self.A_LIM:
                        zup = 0
                    else:
                        zup = 1
                    self.zupSer[did][self.data_idx[did]] = zup
                    
                    self.data_idx[did] += 1

                    # update zup index for identifying chunks of data between zero-velocity occurances
                    if data[4] == 0:
                        if self.zupInd[did][1] == 0:
                            self.zupInd[did][1] = self.data_idx[did]
                    else:
                        self.zupInd[did][1] = 0

                    # process a chunk of data whenever we have a new zero-velocity update
                    if self.integration_flag[did] == False and self.time[did][self.data_idx[did]-1] >= self.time[self.devIds[2]][self.mst[-1]]:
                        if len(self.mst) == 1:
                            count = np.nonzero(self.time[did][:self.data_idx[did]] < self.time[self.devIds[2]][self.mst[-1]])[0][-1] + 1
                            self.vel[did][:count] = [[0,0,0]] * count
                            samp_theta = quaternion.as_float_array(self.q_cal[self.devIds[2]][0] * quaternion.from_float_array(self.qutSer[self.devIds[2]][self.mst[-1]]) * self.q_cal[self.devIds[2]][1])
                            self.zero_cross.append(self.mst[-1])
                            w,x,y,z = samp_theta
                            
                            shank_ang = np.arctan2(2.0* ( w*z + x*y), 1.0-2.0*(y**2+z**2))
                            est_h = self.leg_length*np.cos(shank_ang)
                            self.pos[did][:count] = [[0,est_h,0]] * count
                            self.calc_idx[did] += count
                            self.integration_flag[did] = True

                        else: # if length is not 1, then we have 2 or more so we can begin processing
                            threading.Thread(target=self.chunkProcess, args=(did,self.time[self.devIds[2]][self.mst[-2]],self.time[self.devIds[2]][self.mst[-1]]),name=f'{did}-{self.time[self.devIds[2]][self.mst[-2]]}').start()
                            self.integration_flag[did] = True


                    if did == self.devIds[2]: # update joint angle every time we get a new shank reading, using the most recent data from each sensor
                        q_pv_t =  self.q_cal[self.devIds[0]][0] * quaternion.as_quat_array(self.qutSer[self.devIds[0]][self.data_idx[self.devIds[0]]-1]) * self.q_cal[self.devIds[0]][1]
                        q_th_t = (self.q_cal[self.devIds[1]][0] * quaternion.as_quat_array(self.qutSer[self.devIds[1]][self.data_idx[self.devIds[1]]-1]) * self.q_cal[self.devIds[1]][1]).conj()
                        q_sh_t = (self.q_cal[self.devIds[2]][0] * quaternion.as_quat_array(self.qutSer[self.devIds[2]][self.data_idx[self.devIds[2]]-1]) * self.q_cal[self.devIds[2]][1]).conj()
                        q_ft_t =  self.q_cal[self.devIds[3]][0] * quaternion.as_quat_array(self.qutSer[self.devIds[3]][self.data_idx[self.devIds[3]]-1]) * self.q_cal[self.devIds[3]][1]
                        
                        q_h=quaternion.as_float_array(q_pv_t.conj()*q_th_t)
                        q_k=quaternion.as_float_array(q_sh_t.conj()*q_th_t)
                        q_a=quaternion.as_float_array(q_sh_t.conj()*q_ft_t)

                        theta_h = 2*np.arccos(q_h[0]/np.sqrt(np.power(q_h[0],2)+np.power(q_h[3],2)))*np.sign(q_h[3])*180/np.pi
                        theta_k = 2*np.arccos(q_k[0]/np.sqrt(np.power(q_k[0],2)+np.power(q_k[3],2)))*np.sign(q_k[3])*180/np.pi
                        theta_a = 2*np.arccos(q_a[0]/np.sqrt(np.power(q_a[0],2)+np.power(q_a[3],2)))*np.sign(q_a[3])*180/np.pi

                        self.joint_ang[0].append(theta_h)
                        self.joint_ang[1].append(theta_k)
                        self.joint_ang[2].append(theta_a)

                        # index for gait event detection lags behind one timestep because we need to compare angular velocity to both previous and next values to find local minima/maxima
                        i = self.data_idx[did]-2
                        # q_in_1 = quaternion.from_vector_part(self.gyrSer[did][-1])
                        # q_f = self.q_cal[did][0]
                        # q_imu = quaternion.as_quat_array(self.qutSer[did][-1])
                        # q_in = q_f * q_imu * q_in_1 * q_imu.conj() * q_f.conj()

                        # w_ft = quaternion.as_vector_part(q_in)[2]
                        # if self.cur_search == 1:
                        #     if w_ft > 1.0:
                        #         if w_ft > self.w_temp[0]:
                        #             self.w_temp = [w_ft, i]
                        #     elif self.w_temp[1] > 0:
                        #         self.fs.append(self.w_temp[1])
                        #         self.fs_flag = True
                        #         self.w_temp = [0,0]
                        #         self.cur_search = 2

                        # elif self.cur_search == 2:
                        #     if self.zupInd[did][-1] == 0:
                        #         self.mst.append(i)
                        #         self.mst_flag = True
                        #         self.cur_search = 3

                        # elif self.cur_search == 3:
                        #     if w_ft > 2:
                        #         if w_ft > self.w_temp[0]:
                        #             self.w_temp = [w_ft,i]
                        #     elif self.w_temp[1] > 0:
                        #         self.fo.append(self.w_temp[1])
                        #         self.w_temp = [0,0]
                        #         self.cur_search = 4

                        # elif self.cur_search == 4:
                        #     if w_ft < -0.5:
                        #         if w_ft < self.w_temp[0]:
                        #             self.w_temp = [w_ft, i]
                        #     elif self.w_temp[1] > 0:
                        #         self.msw.append(self.w_temp[1])
                        #         self.w_temp = [0,0]
                        #         self.cur_search = 1
                                
                        recent_gyro = self.gyrSer[did][i-2:self.data_idx[did]]
                        recent_gyro = quaternion.rotate_vectors(self.q_cal[did][0],quaternion.rotate_vectors(quaternion.as_quat_array(self.qutSer[did][self.data_idx[did]-1]),recent_gyro))
                        if self.cur_search == 1:  # mid-swing search
                            if self.flag:  # wait for shank angular velocity falling edge zero crossing
                                if recent_gyro[1][2] < recent_gyro[0][2] and recent_gyro[1][2] <= recent_gyro[2][2] and recent_gyro[1][2] < -2:  # look for local minima with vel < -2 rad/s
                                    self.msw.append(i-1)
                                    self.cur_search = 2  # move on to FS search
                                    self.flag = False  # reset for next search segment
                            elif recent_gyro[0][2] >= 0 and recent_gyro[1][2] < 0:  # identify falling edge zero crossing
                                self.flag = True
                        elif self.cur_search == 2:  # FS search
                            if self.flag:  # wait for shank ang vel rising edge zero crossing
                                if recent_gyro[1][2] > recent_gyro[0][2] and recent_gyro[1][2] > recent_gyro[2][2]:  # look for local maxima
                                    self.fs.append(i-1)
                                    self.fs_flag = True
                                    wait_time=self.time[did][i-1]
                                    self.cur_search = 3  # move on the FS search
                                    self.flag = False  # reset flag for next search segment
                            elif recent_gyro[1][2] > 0:  # identify rising edge zero crossing; we already know from midswing segment that angular velocity is less than zero, so we only need to identify a positive value
                                self.flag = True
                        elif self.cur_search == 3:  # FO search
                            if self.flag:  # wait at least 200 ms
                                # look for local maxima
                                if recent_gyro[1][2] > recent_gyro[0][2] and recent_gyro[1][2] >= recent_gyro[2][2]: 
                                    self.temp = i-1
                                # identify falling edge and set fo to be the maxima id'd above    
                                if recent_gyro[0][2] >= 0 and recent_gyro[1][2] < 0:  
                                    if self.temp > 0:
                                        self.fo.append(self.temp)  # FO occurs at last local maxima
                                        self.mst.append(int((self.fs[-1]+self.temp)/2))
                                        for d in self.devIds:
                                            self.integration_flag[d] = False
                                        self.mst_flag = True
                                    self.temp = -1  # reset temp for next gait cycle
                                    self.cur_search = 1  # move on to MSW search
                                    #  DO NOT reset flag since we have already identified a falling edge zero crossing
                            elif self.time[did][i-1] - wait_time >= .2:
                                self.flag = True
                        else:
                            raise RuntimeError(f'unexpected value of cur_search: {self.cur_search}')
                else:
                    warn(f'Device {did} received packet with duplicate timestamp {data[0]}')
        # after ending data collection and processing all remaining packets, do one more chunk process per device to ensure pos and vel are the same length as other data lists
        for did in self.devIds:
            if len(self.mst) > 1:
                self.chunkProcess(did,self.time[self.devIds[2]][self.mst[-1]])
            else:
                self.chunkProcess(did,self.time[did][self.calc_idx[did]])
        
    def chunkProcess(self,devId,startTime, stopTime=None):
        start = max(np.nonzero(self.time[devId][:self.data_idx[devId]] >= startTime)[0][0]-1,0)
        if stopTime is not None:
            stop = np.nonzero(self.time[devId][:self.data_idx[devId]] < stopTime)[0][-1]+1
        else:
            stop = self.data_idx[devId]
        time_samp=self.time[devId][start:stop]
        fa_samp=self.accSer[devId][start:stop]
        fa_samp = quaternion.as_vector_part(self.q_cal[devId][0] * quaternion.from_vector_part(fa_samp) * self.q_cal[devId][0].conj())  # rotate acc to align with initial anatomical frame
        int_time = time_samp[-1] - time_samp[0]

        samp_vel = integrate.cumulative_trapezoid(fa_samp,time_samp,initial=0,axis=0)

        v_slope = (samp_vel[-1]-samp_vel[0])/int_time
        v_drift = np.multiply(v_slope,(time_samp-time_samp[0])[:,np.newaxis])
        samp_vel=samp_vel-v_drift

        if any(abs(samp_vel[0]) >= 1e-5) or any(abs(samp_vel[-1]) >= 1e-5):
            raise RuntimeError(f'Issue with drift correction. First and last values should be zero, but are {samp_vel[0]} and {samp_vel[-1]}')

        samp_pos = integrate.cumulative_trapezoid(samp_vel,time_samp,initial=0,axis=0)
        
        threads = [thread for thread in threading.enumerate() if devId in thread.name]
        threads.sort(reverse=True, key=lambda x: x.name)

        for thread in threads:
            _,threadTime = thread.name.split('-')
            if float(threadTime)<startTime:
                thread.join()
                break
        
        try:
            samp_pos += self.pos[devId][self.calc_idx[devId]-1]
        
            if devId == self.devIds[2] and len(self.mst) > 0:
                samp_theta = quaternion.as_float_array(self.q_cal[self.devIds[2]][0] * quaternion.from_float_array(self.qutSer[self.devIds[2]][self.mst[-1]]) * self.q_cal[self.devIds[2]][1])
                self.zero_cross.append(self.mst[-1])
                w,x,y,z = samp_theta
                
                shank_ang = np.arctan2(2.0* ( w*z + x*y), 1.0-2.0*(y**2+z**2))
                est_h = self.leg_length*np.cos(shank_ang) 
                
                int_time = time_samp[-1] - time_samp[0]
                x2 = samp_pos[-1][1]
                pos_slope = (est_h - x2)/int_time
                pos_drift = np.multiply(pos_slope,(np.array(time_samp)-time_samp[0]))
                samp_pos [:,1] += pos_drift

            self.vel[devId][self.calc_idx[devId]:self.calc_idx[devId]+len(samp_vel)-1] = samp_vel[1:,:]
            self.pos[devId][self.calc_idx[devId]:self.calc_idx[devId]+len(samp_pos)-1] = samp_pos[1:,:]
            self.calc_idx[devId] += len(samp_pos)-1
            if devId == self.devIds[2] and len(self.fo) >= 3 and self.calc_idx[devId] >= self.fo[-2] and self.fo[-2] > self.fs[-2]:
                self.real_time_llte(self.fs[-2], self.fo[-2])
                
        except IndexError:
            self.vel[devId][self.calc_idx[devId]:self.calc_idx[devId]+len(samp_vel)] = samp_vel[1:,:]
            self.pos[devId][self.calc_idx[devId]:self.calc_idx[devId]+len(samp_pos)] = samp_pos[1:,:]
            self.calc_idx += len(samp_pos)-1
    
    # Function definition for LLTE calculation
    def LLTE_calc(self, LLTE_tar, input_data):
        # Extracting specific columns from LLTE_tar
        x_tar = LLTE_tar[0, 8:-8]
        y_tar = LLTE_tar[1, 8:-8]
        theta_tar = LLTE_tar[2, 8:-8]

        # Extracting specific columns from input_data
        y = input_data[1, 8:-8]
        x = input_data[0, 8:-8]
        theta = input_data[2, 8:-8]
        
        # LLTE calculation using a formula
        LLTE_val = np.sqrt(np.mean(np.square((y / self.leg_length) - y_tar) +
                                    np.square((x / self.leg_length) - x_tar) +
                                    np.square((theta / (np.arctan(self.leg_length / self.foot_length))) - theta_tar) ))
        # LLTE_val = np.sqrt(np.mean( np.square((theta / (np.arctan(self.leg_length / self.foot_length))) - theta_tar) ))
        # self.llte_parts[0] +=list(np.square((x / self.leg_length) - x_tar))
        # self.llte_parts[1] += list(np.square((y / self.leg_length) - y_tar))
        # self.llte_parts[2] += list(np.square((theta / (np.arctan(self.leg_length / self.foot_length))) - theta_tar))
        self.llte_parts[0].append(np.sqrt(np.mean(np.square((x / self.leg_length) - x_tar))))
        self.llte_parts[1].append(np.sqrt(np.mean(np.square((y / self.leg_length) - y_tar))))
        self.llte_parts[2].append(np.sqrt(np.mean(np.square((theta / (np.arctan(self.leg_length / self.foot_length))) - theta_tar))))
        
        # Return the resulting LLTE values
        return LLTE_val
    
    def real_time_llte(self, fs, fo):
        # leg position
        pos = np.array(self.pos[self.devIds[2]][fs:fo])
        leg_position_ap = []
        leg_theta_anatomical = []
        # Normalize to matlab
        # print(fo - fs)
        samp_theta = quaternion.as_float_array(self.q_cal[self.devIds[2]][0] * quaternion.from_float_array(self.qutSer[self.devIds[2]][fs:fo]) * self.q_cal[self.devIds[2]][1])
        for quat in samp_theta:
            w,x,y,z = quat
            shank_ang = np.arctan2(2.0* ( w*z + x*y), 1.0-2.0*(y**2+z**2))
            leg_theta_anatomical.append(shank_ang)
        leg_position_ap = -pos[:,0]
        leg_position_ap += self.leg_length*np.sin(leg_theta_anatomical[0]) - leg_position_ap[0]
        leg_position_ap *= self.leg_length/(self.leg_length-self.imu2knee)
        leg_position_vert = (pos[:,1])
    
        # AP knee, vertical Knee, theta
        LLTE_vars = [leg_position_ap, leg_position_vert, leg_theta_anatomical]
        # Time normalize Stance phase
        t=self.time[self.devIds[2]][fs:fo]
        # t=(t-t[0])
        # t = t * 100/t[-1]
        llte_time_norm = np.vstack([my_resample(101,LLTE_vars[i],t) for i in range(3)])  

        current_llte_val = self.LLTE_calc(self.llte_tar, llte_time_norm)
        self.llte_val.append(current_llte_val)
        if len(self.llte_val) >= 3:
            self.llte_mean.append(-np.mean(self.llte_val[-3:]))
        # print(f'{fo - fs}', current_llte_val, LLTE_vars[0],LLTE_vars[-1])
