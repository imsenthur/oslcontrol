#!/usr/bin/python3
import multiprocessing
import time
from ctypes import util
from fileinput import filename
from unicodedata import name

import numpy as np
import scipy.signal
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
from flexsea import fxUtils as fxu
from scipy import interpolate


class joint:
    def __init__(
        self, name, fxs, dev_id, homing_voltage=2500, homing_rate=0.005
    ) -> None:
        self.name = name
        self.filename = "./encoder_map_" + self.name + ".txt"

        self.fxs = fxs
        self.dev_id = dev_id

        self.homing_voltage = homing_voltage
        self.homing_rate = homing_rate

        self.count2deg = 360 / 2 ** 14
        self.deg2count = 2 ** 14 / 360

    def home(self, save=True):
        print("*** Initiating Homing Routine ***")
        filename = "./encoder_map_" + self.name + ".txt"

        minpos_motor, minpos_joint, min_output = self._homing_routine(direction=1.0)
        print(
            f"\nMinimum Motor angle: {minpos_motor}, Minimum Joint angle: {minpos_joint}"
        )
        time.sleep(0.5)
        maxpos_motor, maxpos_joint, max_output = self._homing_routine(direction=-1.0)
        print(
            f"\nMaximum Motor angle: {maxpos_motor}, Maximum Joint angle: {maxpos_joint}"
        )

        max_output = np.array(max_output).reshape((len(max_output), 2))
        output_motor_count = max_output[:, 1]

        _, ids = np.unique(output_motor_count, return_index=True)

        if save:
            self.save_encoder_map(data=max_output[ids])

        print("*** Homing Successfull ***")

    def _homing_routine(self, direction):
        """
        Private function to aid homing process
        """
        output = []
        velocity_threshold = 0
        go_on = True

        data = self.fxs.read_device(self.dev_id)
        current_motor_position = data.mot_ang
        current_joint_position = data.ank_ang

        try:
            self.fxs.send_motor_command(
                self.dev_id, fxe.FX_VOLTAGE, direction * self.homing_voltage
            )
            time.sleep(0.05)

            data = self.fxs.read_device(self.dev_id)
            cpos_motor = data.mot_ang
            initial_velocity = data.ank_vel
            output.append([data.ank_ang * self.count2deg] + [cpos_motor])

            velocity_threshold = abs(initial_velocity / 2.0)

            while go_on:
                time.sleep(self.homing_rate)
                data = self.fxs.read_device(self.dev_id)
                cpos_motor = data.mot_ang
                cvel_joint = data.ank_vel

                output.append([data.ank_ang * self.count2deg] + [cpos_motor])

                if abs(cvel_joint) <= velocity_threshold:
                    self.fxs.send_motor_command(self.dev_id, fxe.FX_VOLTAGE, 0)
                    current_motor_position = data.mot_ang
                    current_joint_position = data.ank_ang

                    go_on = False

        except KeyboardInterrupt:
            print("Stopping homing routine!")
            self.fxs.send_motor_command(self.dev_id, fxe.FX_NONE, 0)

        return current_motor_position, current_joint_position, output

    def save_encoder_map(self, data):
        """
        Saves encoder_map: [Joint angle, Motor count] to a text file
        """
        # Saving reversed array because of our joint's zero position
        np.savetxt(self.filename, data, fmt="%.5f")

    def load_encoder_map(self):
        """
        Loads Joint angle array, Motor count array, Min Joint angle, and Max Joint angle
        """
        data = np.loadtxt(self.filename, dtype=np.float64)
        self.joint_angle_array = data[:, 0]
        self.motor_count_array = np.array(data[:, 1], dtype=np.int32)

        self.min_joint_angle = np.min(self.joint_angle_array)
        self.max_joint_angle = np.max(self.joint_angle_array)

        self.joint_angle_array = self.max_joint_angle - self.joint_angle_array

        # Applying a median filter with a kernel size of 3
        self.joint_angle_array = scipy.signal.medfilt(
            self.joint_angle_array, kernel_size=3
        )
        self.motor_count_array = scipy.signal.medfilt(
            self.motor_count_array, kernel_size=3
        )

    def joint_angle_2_motor_count(self, desired_joint_angle):
        desired_joint_angle_array = np.array(desired_joint_angle)
        np.set_printoptions(suppress=True)

        func = interpolate.interp1d(self.joint_angle_array, self.motor_count_array)
        desired_motor_count = np.interp(
            desired_joint_angle, self.joint_angle_array, self.motor_count_array
        )
        return desired_motor_count

    def motor_count_2_joint_angle(self):
        pass


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

        # Joint parameters
        self.knee = joint(name="knee", fxs=self.fxs, dev_id=self.dev_id)

        # Loadcell paramters
        self.loadcell_matrix = np.array(
            [
                (-38.72600, -1817.74700, 9.84900, 43.37400, -44.54000, 1824.67000),
                (-8.61600, 1041.14900, 18.86100, -2098.82200, 31.79400, 1058.6230),
                (-1047.16800, 8.63900, -1047.28200, -20.70000, -1073.08800, -8.92300),
                (20.57600, -0.04000, -0.24600, 0.55400, -21.40800, -0.47600),
                (-12.13400, -1.10800, 24.36100, 0.02300, -12.14100, 0.79200),
                (-0.65100, -28.28700, 0.02200, -25.23000, 0.47300, -27.3070),
            ]
        )

        self.amp_gain = 125.0
        self.exc = 5.0

    def start(self, frequency=500, log_en=False):
        """
        Starts the actuator

        Parameters:
        -----------
        """
        self.fxs.start_streaming(self.dev_id, freq=frequency, log_en=log_en)
        time.sleep(2)

        if input("Do you want to initiate homing process? (y/n) ") == "y":
            self.knee.home()

        self.knee.load_encoder_map()

        # TEST
        print(self.knee.joint_angle_2_motor_count(0))

    def stop(self):
        """
        Shutsdown the actuator
        """
        self.fxs.send_motor_command(self.dev_id, fxe.FX_NONE, 0)
        time.sleep(1)
        self.fxs.stop_streaming(self.dev_id)
        time.sleep(1)
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
            loadcell_offset = self.get_loadcell_data(
                loadcell_raw, initial_loadcell_zero
            )
            loadcell_zero = (loadcell_offset + loadcell_zero) / 2.0

        return loadcell_zero

    def get_battery_voltage(self):
        """
        Returns battery voltage
        """
        data = self.fxs.read_device(self.dev_id)

        return data.batt_volt

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

    def read(self, duration=15, time_step=0.01):
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
    osl.start()
    osl.stop()

    finish = time.perf_counter()
    print(f"Script ended at {finish-start:0.4f}")
