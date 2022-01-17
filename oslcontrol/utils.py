#!/usr/bin/python3
import multiprocessing
import time

import numpy as np
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
from flexsea import fxUtils as fxu


class opensourceleg:
    """
    The OSL class
    """

    def __init__(self, port, baud_rate, debug_level=0) -> None:
        self.port = port
        self.baud_rate = baud_rate

        # Actpack parameters
        self.fxs = flex.FlexSEA()
        self.dev_id = self.fxs.open(self.port, self.baud_rate, debug_level)
        self.app_type = self.fxs.get_app_type(self.dev_id)

        # Loadcell paramters
        self.loadcell_matrix = np.array([(-38.72600, -1817.74700, 9.84900, 43.37400, -44.54000, 1824.67000),
                                        (-8.61600, 1041.14900, 18.86100, -2098.82200, 31.79400, 1058.6230),
                                        (-1047.16800, 8.63900, -1047.28200, -20.70000, -1073.08800, -8.92300),
                                        (20.57600, -0.04000, -0.24600, 0.55400, -21.40800, -0.47600),
                                        (-12.13400, -1.10800, 24.36100, 0.02300, -12.14100, 0.79200),
                                        (-0.65100, -28.28700, 0.02200, -25.23000, 0.47300, -27.3070)]) 

        self.amp_gain = 125.0
        self.exc = 5.0

    def start(self, frequency=500, log_en=False):
        """
        Starts the actuator

        Parameters:
        -----------
        """
        self.fxs.start_streaming(self.dev_id, freq=frequency, log_en=log_en)

    def stop(self):
        """
        Shutsdown the actuator
        """
        self.fxs.send_motor_command(self.dev_id, fxe.FX_VOLTAGE, 0)
        self.fxs.stop_streaming(self.dev_id)
        self.fxs.close(self.dev_id)

    def initialize_loadcell(self, amp_gain, exc, loadcell_matrix):
        """
        Initializes Loadcell Matrix

        Parameters:
        -----------

        """       
        self.amp_gain = amp_gain
        self.exc = exc
        self.loadcell_matrix = loadcell_matrix

    def get_loadcell_data(self, loadcell_raw, loadcell_zero):
        """
        Computes Loadcell data

        """
        loadcell_signed = (loadcell_raw - 2048) / 4095 * self.exc
        loadcell_coupled = loadcell_signed * 1000 / (self.exc * self.amp_gain)

        return (
            np.transpose(self.loadcell_matrix.dot(np.transpose(loadcell_coupled)))
            - loadcell_zero
        )

    def get_loadcell_zero(self, loadcell_raw, number_of_iterations=2000):
        """
        Obtains the initial loadcell reading (aka) loadcell_zero
        """
        initial_loadcell_zero = np.zeros((6, 1), dtype=np.double)
        loadcell_zero = loadcell_offset = self.get_loadcell_data(
            loadcell_raw, initial_loadcell_zero
        )

        for i in range(number_of_iterations):
            loadcell_offset = self.get_loadcell_data(loadcell_raw, initial_loadcell_zero)
            loadcell_zero = (loadcell_offset + loadcell_zero) / 2.0

        return loadcell_zero

    def get_actuator_data(self):
        """
        Streams data from the actuator.

        Parameters:
        -----------

        Returns:
        --------
        acc:
        gyro:
        loadcell_raw:

        """
        data = self.fxs.read_device(self.dev_id)

        motor = np.array([data.mot_ang, data.mot_vel, data.mot_acc])
        joint = np.array([data.ank_ang, data.ank_vel])
        acc = np.array([data.accelx, data.accely, data.accelz])
        gyro = np.array([data.gyrox, data.gyroy, data.gyroz])
        loadcell = np.array(
            [
                data.genvar_0,
                data.genvar_1,
                data.genvar_2,
                data.genvar_3,
                data.genvar_4,
                data.genvar_5,
            ]
        )

        return motor, joint, acc, gyro, loadcell

    def read(self, duration, time_step=0.01):
        number_of_iterations = int(duration / time_step)

        # Start streaming data from the actpack
        self.start()

        for i in range(number_of_iterations):
            time.sleep(time_step)
            fxu.clear_terminal()

            motor, joint, acc, gyro, loadcell_raw = self.get_actuator_data()

            if i == 0:
                print("*** Obtaining initial value of loadcell ***")
                loadcell_zero = self.get_loadcell_zero(loadcell_raw)

            loadcell = self.get_loadcell_data(loadcell_raw, loadcell_zero)
            print(loadcell)

        time.sleep(1)
        self.stop()


if __name__ == "__main__":
    start = time.perf_counter()

    osl = opensourceleg(port="/dev/ttyACM0", baud_rate=230400)
    osl.read(15)

    # fxs = flex.FlexSEA()
    # dev_id = fxs.open('/dev/ttyACM0', 230400, log_level=6)
    # fxs.start_streaming(dev_id, 500, log_en=False)

    # p1 = multiprocessing.Process(target=stream, args=(fxs, dev_id, 15, 0.01, motor, joint, accelerometer, gyroscope, loadcell))
    # p1.start()

    # print(np.array(accelerometer), np.array(gyroscope))

    finish = time.perf_counter()
    print(f"Script ended at {finish-start:0.4f}")
