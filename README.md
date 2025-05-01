# ReBAIT
### Real-time Biofeedback and Analysis using IMU Tracking

ReBAIT is an efficient system for collecting IMU data and calculating real-time biomechanically relevant measures for biofeedback or clinical monitoring applications. It is fully extensible and highly adaptable for different sensors, body segments, biomechanical calculations, and feedback.

**Table of contents:**
 - [Callback](#callback)
 - [DataCollector](#dc)
 - [Plotter](#plotter)

 <a id="callback"></a>
 ## Callback
 This module controls the physical devices, including initiating collection and packaging raw data to send to the `DataCollector` module. Alternatively, this class can be used to replay data that was collected for development and post-processing purposes. All below functions and fields must exist in a valid Callback class, even if they simply return immediately. For optimal performance, this class should contain a buffer for incoming packets as no interrupts are implemented and synchronous processing is not guaranteed. For representative Callback functions, see [xdaCallback.py](xdaCallback.py) or [playback.py](playback.py).

 **Fields:**
 - `devIds`: This field contains a list of unique strings to identify each sensor connected to the system.
 - `samp_freq`: This field contains the actual sample frequency of the sensors in Hertz. It does not need to be exact, as timestamps are collected from data packets, but it is used to preallocate arrays for storing data efficiently so it should be close to avoid issues with overrunning the array sizes during data collection.

 **Functions:**
 - `enable()`: This function is called at the beginning of data collection, before calibration procedures begin. Any necessary steps for managing the sensors, such as enabling radio communication or starting measurement, should be placed here.
 - `attach()`: This function allows data packets to start accumulating in the internal buffer.
 - `packetAvailable()`: This function returns `True` while unprocessed data packets exist, and `False` when there are no data packets to process.
 - `getNextPacket(`): This function returns the device ID and data from the next available packet, which must conform to the structure expected by the DataCollector. An example structure is shown below. This function is only called after `packetAvailable()` is called and returned `True`, so behavior is undefined if it is called when a packet does not exist.

 |Value|Type and Shape|
 |---|:---:|
 |Time|1x1 float|
 |Acceleration| 3x1 float|
 |Angular Velocity| 3x1 float|
 |Quaternion| 4x1 float|
 |Calibrated Acceleration| 3x1 float|
 |Euler Angles| 3x1 float|


 - `detach()`: This function stops data packets from continuing to accumulate in the buffer. This is very important, as data collection during trials will not end until the packetAvilable() function returns `False`
 - `close()`: This function is called at the end of data collection and should contain any necessary steps for cleanly exiting the program, such as turning off radios and closing ports.

<a id="dc"></a>
## DataCollector
This module stores incoming data, calculates calibration parameters, and performs any required processing. This module can be customized for use with different body segments or to calculate different values as necessary for different applications. The minimum required fields and functions are listed below.

**Fields:**
- `data_idx`: This field is a library containing a value for each device representing the index at which the next data packet should be stored in the pre-allocated arrays. Device IDs should be used as the keys for this library.
- `time`: this field is a library containing a 1-D list or array for each device representing the timestamp of each received and processed packet. Device IDs should be used as the keys for this library.

**Funtions:**
- `staticCalibrate()`: This function must return on its own. It is designed to collect 5 seconds of data while the subject is stationary to use in calculating baseline positions and orientations.
- `startCalibration()`: This function should create a new thread for collecting data that will run until stopped by the `stopCalibration()` function. It is designed to collect data necessary to align sensor cooridnate systems with anatomical coordinate systems using functional movements.
- `stopCalibration()`: This function stops collection in the calibration thread. Calibration parameters can then be calculated as necessary before proceding with data collection trials.
- `startCollection()`: This function starts a new thread for data collection and processing. It should run until stopped by the `stopCollection()` function.
- `stopCollection()`: This function allows any remaining data packets to be processed, then closes the collection thread and saves the file.

<a id="plotter"></a>
## Plotter
This module provides real-time plotting capabilities, either for providing biofeedback to users in learning and adaptation studies, or for researchers to monitor ongoing data collection. It can easily be customized to show different values or different plot types. A Plotter class requires the functions listed below.

**Functions:**
- `reset_plot()`: This function is called just before data collection and calibration starts. It is designed to initialize the plot canvas and artists for drawing the desired plots.
- `start_plot()`: This function is called after starting data collection for each trial. It must pause execution in the main thread when called, and return to the main thread when the trial is over.
