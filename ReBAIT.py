#  author: Zach Hoegberg
#  

from os import listdir, remove
import os
from os.path import isfile, join
import pickle
from time import sleep
import tkinter as tk
import numpy as np
from LltePlotFunc import MainWindow
from DataCollector import DataCollector
from xdaCallback import XdaCallback

def ReBAIT(dc, callback, imu_locs, plotter):
    # create callback function and pass it to data collector object
    devIds = callback.devIds

    print(f'Found {len(devIds)} devices. Please attach to the following segments:')
    for i in range(min(len(devIds),len(imu_locs))):
        print(f' {imu_locs[i]} : {devIds[i]}')
    input("Once IMUs are in place, press enter.")

    os.makedirs('data',exist_ok=True)
    os.makedirs('temp',exist_ok=True)

    while True:
        plotter.reset_plot()
        callback.enable()
        
        input('Press enter to begin 5 second static calibration period.')

        print("Starting calibration period.")
        callback.attach()  # attach callback handler to station to start collecting data packets
        dc.staticCalibrate()  # runs calibration function. Collects 5 seconds of stationary data to determine IMU orientation.
        callback.detach()  # remove callback to stop collecting data

        input('Stationary calibration completed. Press enter to begin dynamic calibration.')

        callback.attach()  # attach callback handler to station to start collecting data packets
        dc.startCalibration()  # starts a background thread for data collection and processing
        input('Press enter after toe touches and several steps to end dynamic calibration.')
        callback.detach()  # remove callback to stop collecting data
        dc.stopCalibration()

        input('Calibration complete. Press enter to start data collection.')

        dc.startCollection()  # starts a background thread for data collection and processing

        flag = True
        trial_stamps = []

        while flag:
            start_index = dc.data_idx[devIds[2]] #len(dc.time[devIds[2]])
            callback.attach()  # attach callback handler to station to start collecting data packets

            print('Data collection starting. Close plot to end data collection.')

            # w.start_plot() will pause execution in this thread until the plot is closed. Data collection and plot animation will continue running in the background
            plotter.start_plot()

            callback.detach()  # remove callback to stop collecting data
            while callback.packetAvailable():
                sleep(0.1)
            stop_time = dc.time[devIds[2]][dc.data_idx[devIds[2]]-1]

            user_input = ''
            while user_input is None or len(user_input) < 1:
                user_input = "".join(filter(lambda x: str.isalnum(x) or str.isspace(x),input("Enter string to label trial data. ")))
            trial_stamps.append([user_input, dc.time[devIds[2]][start_index], stop_time])
            with open(f'temp/{user_input}.pkl','wb') as f:
                temp_data = [dc.time, dc.accSer, dc.gyrSer, dc.qutSer, dc.zupSer, dc.calAcc, dc.eulAng]
                pickle.dump(temp_data,f,-1)
            user_input = input("Press Enter to collect another trial with same target. Enter \'t\' to select new target. Enter \'q\' to quit data collection and save file. ")
            while True:
                if user_input:
                    if user_input == 't':
                        dc.llte_tar = UI_get_pkl()
                        break
                    elif user_input == 'q':
                        flag = False
                        break
                    else:
                        user_input = input("Input not recognized. Please press Enter to collect another trial with same target. Enter \'t\' to select new target. Enter \'q\' to quit data collection and save file. ")
                else:
                    break

        dc.stopCollection(trial_stamps)  # this function will wait until any remaining packets are processed, then save the file

        for f in [join('temp',f) for f in listdir('temp') if isfile(join('temp', f))]:
            try:
                remove(f)
            except FileNotFoundError:
                print(f'temp file error: {f}')

        if input('Enter y to start calibration and collect another sample. Enter n to exit. ') == 'n':
            break
            
    callback.close()

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
    file_path = tk.filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    
    # Check if the user selected a file
    if file_path:
        # Extract the original file name
        # Open and read the selected pickle file
        with open(file_path, 'rb') as file:
            loaded_data = np.load(file)

        # Now, 'loaded_data' contains the content of the pickle file
    else:
        print("No file selected")
    return loaded_data

if __name__ == '__main__':
    callback = XdaCallback(max_buffer_size=800,samp_freq=100)
    imu_locs = ['Pelvis','RThigh','RShank','RFoot', 'LThigh', 'LShank', 'LFoot', 'Torso']
    anthro_data_input = ['Foot Length', 'Leg Length', 'IMU to KJC']
    anthro_data = {}
    for d in anthro_data_input:
        m = None
        while m is None:
            try:
                m = float(input(f'Enter value for {d}: '))
            except ValueError:
                print('Invalid entry. Please enter a number.')
        anthro_data[d] = m
    
    txt = input("Press enter to use default LLTE target data or enter any character to select LLTE target data.")
    if txt:
        print("Select LLTE Target Data")
        target = UI_get_pkl() # Get pickle file with the target dictionary 
    else:
        with open('targets/target0.pkl', 'rb') as file:
                target = np.load(file)
    
    dc = DataCollector(callback,callback.devIds,imu_locs,anthro_data,target)
    plotter = MainWindow(dc)
    ReBAIT(dc, callback,imu_locs,anthro_data,target,plotter)
