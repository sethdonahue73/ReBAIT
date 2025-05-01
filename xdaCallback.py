from warnings import warn
import xsensdeviceapi as xda
from threading import Lock

## Class for passing data to DataCollector
class XdaCallback(xda.XsCallback):
    def __init__(self, max_buffer_size = 5, samp_freq = 100):
        self.control = xda.XsControl_construct()

        xdaVersion = xda.XsVersion()
        xda.xdaVersion(xdaVersion)

        p_br =  xda.XsScanner_scanPorts()
        if p_br.size() == 0:
            raise RuntimeError('No base station found. Please connect base station.')
        elif p_br.size() > 1:
            raise RuntimeError('More than one device found. Please disconnect all but one base station.')
        elif not p_br[0].deviceId().isAwindaXStation():
            raise RuntimeError('Device connected is not base station. Please connect base station.')
        
        deviceId = p_br[0].deviceId()
        self.portS = p_br[0].portName()
        baudRate = p_br[0].baudrate()

        if not self.control.openPort(self.portS, baudRate):
            raise RuntimeError("Could not open port. Aborting.")

        # Get the station object
        self.station = self.control.device(deviceId)

        children = self.station.children()
        self.devIds = [child.deviceId().toXsString() for child in children if child.connectivityState() == xda.XCS_Wireless]

        # Put the station into configuration mode before configuring the station
        if not self.station.gotoConfig():
            raise RuntimeError("Could not put station into configuration mode. Aborting.")
        self.station.setOptions(xda.XSO_Orientation + xda.XSO_Calibrate,0)
        self.station.setUpdateRate(samp_freq)

        xda.XsCallback.__init__(self)
        self.m_maxNumberOfPacketsInBuffer = max_buffer_size
        self.m_packetBuffer = list()
        self.m_lock = Lock()
        self.samp_freq = self.station.getUpdateRate()  # TODO: verify if this is the correct function

    def packetAvailable(self):
        xda.XsTimeStamp_nowMs()
        self.m_lock.acquire()
        res = len(self.m_packetBuffer) > 0
        self.m_lock.release()
        return res

    def getNextPacket(self):
        self.m_lock.acquire()
        assert(len(self.m_packetBuffer) > 0)
        oldest_packet = xda.XsDataPacket(self.m_packetBuffer.pop(0))
        self.m_lock.release()
        return self.packetExtract(oldest_packet)
    
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

    def onLiveDataAvailable(self, dev, packet):
        self.m_lock.acquire()
        assert(packet != 0)
        while len(self.m_packetBuffer) >= self.m_maxNumberOfPacketsInBuffer:
            self.m_packetBuffer.pop()
            warn('Cache size exceeded. Dropping packets!')
        self.m_packetBuffer.append(xda.XsDataPacket(packet))
        self.m_lock.release()

    def enable(self):
        if not self.station.isRadioEnabled:
            output = self.station.enableRadio(11)
            if not output:
                raise RuntimeError('Failed to enable radio')

        if not self.station.gotoMeasurement():
            raise RuntimeError("Could not put station into measurement mode. Aborting.")
        
    def attach(self):
        self.station.addCallbackHandler(self)
    def detach(self):
        self.station.removeCallbackHandler(self)

    def close(self):
        self.control.closePort(self.portS)
        self.control.close()
