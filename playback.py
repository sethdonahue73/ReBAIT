import numpy as np

class playback:
    def __init__(self,data_set,devIds,cal_time):
        self.ds = data_set
        self.devIds = devIds
        self.cal_time = cal_time
        self.mode = 0 # default to calibration mode
        self.packets = []

        time = self.ds['time'].to_numpy()
        for did in self.devIds:
            data = self.ds[did].loc[:,['FreeAccX','FreeAccY','FreeAccZ','AngVelX','AngVelY','AngVelZ','QuaternionW','QuaternionX','QuaternionY','QuaternionZ','CalAccX','CalAccY','CalAccZ','Pitch','Yaw','Roll']].to_numpy()
            for i in range(len(time)):
                if np.isnan(data[i,:]).any():
                    continue
                t = time[i]
                acc = data[i,0:3]
                gyr = data[i,3:6]
                qut = data[i,6:10]
                cal_acc = data[i,10:13]
                eul = data[i,13:16]
                p = Packet(did,t,acc,gyr,qut,cal_acc,eul)
                self.packets.append(p)
        self.packets.sort(key=lambda x: x.estimatedTimeOfSampling().secTime())
        print(f'Total packets: {len(self.packets)}, total file time: {time[-1]-time[0]}')

    
    def packetAvailable(self):
        if self.mode == 0:
            if self.packets[0].estimatedTimeOfSampling().secTime() < self.cal_time:
                return True
            else:
                self.mode = 1
                return False
        return len(self.packets) > 0


    def getNextPacket(self):
        packet = self.packets.pop(0)
        return self.packetExtract(packet)
    
    def packetExtract(self,packet):
        did = packet.deviceId().toXsString()
        data = []

        if packet.containsFreeAcceleration and packet.containsCalibratedGyroscopeData and packet.containsOrientation(): #TODO: check if these are functions or attributes
            tos = packet.estimatedTimeOfSampling().secTime()
            data.append(tos)

            acc = packet.freeAcceleration()
            data.append(acc)

            gyr = packet.calibratedGyroscopeData()
            data.append(gyr)

            qut = packet.orientationQuaternion()
            data.append(qut)

            acc = packet.calibratedAcceleration()
            data.append(acc)

            eul = packet.orientationEuler()
            eul = [eul.pitch(), eul.yaw(), eul.roll()]
            data.append(eul)
        return did, data

class Packet:
    def __init__(self,id,time,acc,gyr,qut,cal_acc,eul):
        self.id = DeviceId(id)
        self.time = Time(time)
        self.acc = acc
        self.gyr = gyr
        self.qut = qut
        self.cal_acc = cal_acc
        self.eul = EulerAng(eul)
        self.containsFreeAcceleration = True
        self.containsCalibratedGyroscopeData = True
        # self.containsOrientation = True

    def __str__(self):
        return f'{self.id: >6}, {self.time:12.3f}'
    
    def containsOrientation(self):
        return True

    def deviceId(self):
        return self.id

    def estimatedTimeOfSampling(self):
        return self.time

    def freeAcceleration(self):
        return self.acc

    def calibratedGyroscopeData(self):
        return self.gyr

    def orientationQuaternion(self):
        return self.qut

    def calibratedAcceleration(self):
        return self.cal_acc

    def orientationEuler(self):
        return self.eul

class DeviceId:
    def __init__(self,id):
        self.id = id
    def toXsString(self):
        return self.id

class Time:
    def __init__(self,time):
        self.time = time
    def secTime(self):
        return self.time

class EulerAng:
    def __init__(self,eul):
        self.eul = eul
    def pitch(self):
        return self.eul[0]
    def yaw(self):
        return self.eul[1]
    def roll(self):
        return self.eul[2]