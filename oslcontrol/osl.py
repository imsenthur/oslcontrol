#!/usr/bin/python3
from typing import Any, Callable, List, Optional

import time

import numpy as np
import scipy.signal
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
from flexsea import fxUtils as fxu


class State:
    """
    State class
    """

    def __init__(self, name, theta=0.0, k=0.0, b=0.0) -> None:
        self._name = name

        # Impedance parameters
        self._theta = theta
        self._k = k
        self._b = b

        # Callbacks
        self._entry_callbacks: list[Callable[[Any], None]] = []
        self._exit_callbacks: list[Callable[[Any], None]] = []

    def __eq__(self, __o: object) -> bool:
        if __o.name == self._name:
            return True
        else:
            return False

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)

    def __call__(self, data: Any) -> Any:
        pass

    def set_impedance_paramters(self, theta, k, b) -> None:
        self._theta = theta
        self._k = k
        self._b = b

    def get_impedance_paramters(self):
        return self._theta, self._k, self._b

    def on_entry(self, callback: Callable[[Any], None]):
        self._entry_callbacks.append(callback)

    def on_exit(self, callback: Callable[[Any], None]):
        self._exit_callbacks.append(callback)

    def start(self, data: Any):
        print("Entering: ", self._name)
        for c in self._entry_callbacks:
            c(data)

    def stop(self, data: Any):
        print("Exiting: ", self._name)
        for c in self._exit_callbacks:
            c(data)

    @property
    def name(self):
        return self._name


class Idle(State):
    def __init__(self, status="Idle") -> None:
        self._name = status
        super().__init__(self._name)

    @property
    def status(self):
        return self._name


class Event:
    """
    Event class
    """

    def __init__(self, name) -> None:
        self._name = name

    def __eq__(self, __o: object) -> bool:
        if __o.name == self._name:
            return True
        else:
            return False

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__

    @property
    def name(self):
        return self._name


class Transition:
    """
    Transition class
    """

    def __init__(
        self,
        event: Event,
        source: State,
        destination: State,
        callback: Callable[[Any], bool] = None,
    ) -> None:
        self._event = event
        self._source_state = source
        self._destination_state = destination

        self._criteria: Optional[Callable[[Any], bool]] = callback
        self._action: Optional[Callable[[Any], None]] = None

    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

    def add_criteria(self, callback: Callable[[Any], bool]):
        self._criteria = callback

    def add_action(self, callback: Callable[[Any], Any]):
        self._action = callback

    @property
    def event(self):
        return self._event

    @property
    def source_state(self):
        return self._source_state

    @property
    def destination_state(self):
        return self._destination_state


class FromToTransition(Transition):
    def __init__(
        self,
        event: Event,
        source: State,
        destination: State,
        callback: Callable[[Any], bool] = None,
    ) -> None:
        super().__init__(event, source, destination, callback)

        self._from = source
        self._to = destination

    def __call__(self, data: Any):
        if not self._criteria or self._criteria(data):
            if self._action:
                self._action(data)

            self._from.stop(data)
            self._to.start(data)

            return self._to
        else:
            return self._from


class Joint:
    def __init__(
        self, name, fxs, dev_id, homing_voltage=2500, homing_rate=0.005
    ) -> None:
        self._name = name
        self._filename = "./encoder_map_" + self._name + ".txt"

        self._fxs = fxs
        self._dev_id = dev_id

        self._homing_voltage = homing_voltage
        self._homing_rate = homing_rate

        self._count2deg = 360 / 2 ** 14
        self._deg2count = 2 ** 14 / 360

        self._bit_2_degree_per_sec = 1 / 32.8
        self._bit_2_g = 1 / 8192

    def home(self, save=True):
        """
        Homing function
        """
        print("*** Initiating Homing Routine ***")

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

        data = self._fxs.read_device(self._dev_id)
        current_motor_position = data.mot_ang
        current_joint_position = data.ank_ang

        try:
            self._fxs.send_motor_command(
                self._dev_id, fxe.FX_VOLTAGE, direction * self._homing_voltage
            )
            time.sleep(0.05)

            data = self._fxs.read_device(self._dev_id)
            cpos_motor = data.mot_ang
            initial_velocity = data.ank_vel
            output.append([data.ank_ang * self._count2deg] + [cpos_motor])

            velocity_threshold = abs(initial_velocity / 2.0)

            while go_on:
                time.sleep(self._homing_rate)
                data = self._fxs.read_device(self._dev_id)
                cpos_motor = data.mot_ang
                cvel_joint = data.ank_vel

                output.append([data.ank_ang * self._count2deg] + [cpos_motor])

                if abs(cvel_joint) <= velocity_threshold:
                    self._fxs.send_motor_command(self._dev_id, fxe.FX_VOLTAGE, 0)
                    current_motor_position = data.mot_ang
                    current_joint_position = data.ank_ang

                    go_on = False

        except KeyboardInterrupt:
            print("Stopping homing routine!")
            self._fxs.send_motor_command(self._dev_id, fxe.FX_NONE, 0)

        return current_motor_position, current_joint_position, output

    def save_encoder_map(self, data):
        """
        Saves encoder_map: [Joint angle, Motor count] to a text file
        """
        # Saving reversed array because of our joint's zero position
        np.savetxt(self._filename, data, fmt="%.5f")

    def load_encoder_map(self):
        """
        Loads Joint angle array, Motor count array, Min Joint angle, and Max Joint angle
        """
        data = np.loadtxt(self._filename, dtype=np.float64)
        self._joint_angle_array = data[:, 0]
        self._motor_count_array = np.array(data[:, 1], dtype=np.int32)

        self._min_joint_angle = np.min(self._joint_angle_array)
        self._max_joint_angle = np.max(self._joint_angle_array)

        self._joint_angle_array = self._max_joint_angle - self._joint_angle_array

        # Applying a median filter with a kernel size of 3
        self._joint_angle_array = scipy.signal.medfilt(
            self._joint_angle_array, kernel_size=3
        )
        self._motor_count_array = scipy.signal.medfilt(
            self._motor_count_array, kernel_size=3
        )

    def joint_angle_2_motor_count(self, desired_joint_angle):
        desired_joint_angle_array = np.array(desired_joint_angle)
        desired_motor_count = np.interp(
            desired_joint_angle_array, self._joint_angle_array, self._motor_count_array
        )
        return desired_motor_count

    def get_orientation(self, raw_acc, raw_g):
        acc = raw_acc * self._bit_2_g
        gyro = raw_g * self._bit_2_degree_per_sec
        return acc, gyro

    @property
    def name(self):
        return self._name


class OSL:
    """
    The OSL class
    """

    def __init__(self, port, baud_rate, debug_level=0) -> None:
        self.port = port
        self.baud_rate = baud_rate

        # Actuator Variables
        self.fxs = flex.FlexSEA()
        self.dev_id = self.fxs.open(self.port, self.baud_rate, debug_level)
        self.app_type = self.fxs.get_app_type(self.dev_id)

        # Joint Variables
        self.knee = Joint(name="knee", fxs=self.fxs, dev_id=self.dev_id)

        # Loadcell Variables
        self.loadcell_matrix = None

        # State Machine Variables
        self._states: list[State] = []
        self._events: list[Event] = []
        self._transitions: list[Transition] = []
        self._initial_state: Optional[State] = None
        self._current_state: Optional[State] = None
        self._exit_callback: Optional[Callable[[Idle, Any], None]] = None
        self._exit_state = Idle()
        self.add_state(self._exit_state)
        self._exited = True

    def start(self, data: Any = None):
        if not self._initial_state:
            raise ValueError("Initial state hasn't been set.")

        self._current_state = self._initial_state
        self._exited = False
        self._current_state.start(data)

    def stop(self, data: Any):
        if not (self._initial_state or self._current_state):
            raise ValueError("OSL isn't active.")

        self._current_state.stop(data)
        self._current_state = self._exit_state
        self._exited = True

    def on_event(self, data: Any = None):
        validity = False

        if not (self._initial_state or self._current_state):
            raise ValueError("OSL isn't active.")

        for transition in self._transitions:
            if transition.source_state == self._current_state:
                self._current_state = transition(data)

                if isinstance(self._current_state, Idle) and not self._exited:
                    self._exited = True

                    if self._exit_callback:
                        self._exit_callback(self._current_state, data)

                validity = True
                break

        if not validity:
            print("Event isn't valid at ", self._current_state)

    def is_running(self) -> bool:
        if self._current_state and self._current_state != self._exit_state:
            return True
        else:
            return False

    def on_exit(self, callback):
        self._exit_callback = callback

    def add_state(self, state: State, initial_state: bool = False):
        if state in self._states:
            raise ValueError("State already exists.")

        self._states.append(state)

        if not self._initial_state and initial_state:
            self._initial_state = state

    def add_event(self, event: Event):
        self._events.append(event)

    def add_transition(
        self,
        source: State,
        destination: State,
        event: Event,
        callback: Callable[[Any], bool] = None,
    ) -> Optional[Transition]:
        transition = None

        if (
            source in self._states
            and destination in self._states
            and event in self._events
        ):
            transition = FromToTransition(event, source, destination, callback)
            self._transitions.append(transition)

        return transition

    def _start_streaming_data(self, frequency=500, log_en=False):
        """
        Starts the actuator

        Parameters:
        -----------
        """

        # If loadcell hasn't been initialized, assign default values.
        if not self.loadcell_matrix:
            self.initialize_loadcell()

        self.fxs.start_streaming(self.dev_id, freq=frequency, log_en=log_en)
        time.sleep(2)

        # if input("Do you want to initiate homing process? (y/n) ") == "y":
        #     self.knee.home()

        self.knee.load_encoder_map()

    def _stop_streaming_data(self):
        """
        Shuts down the actuator
        """
        self.fxs.send_motor_command(self.dev_id, fxe.FX_NONE, 0)
        time.sleep(1)
        self.fxs.stop_streaming(self.dev_id)
        time.sleep(1)
        self.fxs.close(self.dev_id)

    def initialize_loadcell(self, amp_gain=125.0, exc=5.0, loadcell_matrix=None):
        """
        Initializes Loadcell Matrix

        Parameters:
        -----------

        """
        self.amp_gain = amp_gain
        self.exc = exc

        if not loadcell_matrix:
            self.loadcell_matrix = np.array(
                [
                    (-38.72600, -1817.74700, 9.84900, 43.37400, -44.54000, 1824.67000),
                    (-8.61600, 1041.14900, 18.86100, -2098.82200, 31.79400, 1058.6230),
                    (
                        -1047.16800,
                        8.63900,
                        -1047.28200,
                        -20.70000,
                        -1073.08800,
                        -8.92300,
                    ),
                    (20.57600, -0.04000, -0.24600, 0.55400, -21.40800, -0.47600),
                    (-12.13400, -1.10800, 24.36100, 0.02300, -12.14100, 0.79200),
                    (-0.65100, -28.28700, 0.02200, -25.23000, 0.47300, -27.3070),
                ]
            )
        else:
            self.loadcell_matrix = loadcell_matrix

    def _get_loadcell_data(self, loadcell_raw, loadcell_zero):
        """
        Computes Loadcell data

        """
        loadcell_signed = (loadcell_raw - 2048) / 4095 * self.exc
        loadcell_coupled = loadcell_signed * 1000 / (self.exc * self.amp_gain)

        return (
            np.transpose(self.loadcell_matrix.dot(np.transpose(loadcell_coupled)))
            - loadcell_zero
        )

    def _get_loadcell_zero(self, loadcell_raw, number_of_iterations=2000):
        """
        Obtains the initial loadcell reading (aka) loadcell_zero
        """
        initial_loadcell_zero = np.zeros((1, 6), dtype=np.double)
        loadcell_zero = loadcell_offset = self._get_loadcell_data(
            loadcell_raw, initial_loadcell_zero
        )

        for i in range(number_of_iterations):
            loadcell_offset = self._get_loadcell_data(
                loadcell_raw, initial_loadcell_zero
            )
            loadcell_zero = (loadcell_offset + loadcell_zero) / 2.0

        return loadcell_zero

    def _get_sensor_data(self):
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
        """
        Reads data from the actuator for 'n' number of seconds.
        """
        number_of_iterations = int(duration / time_step)

        self._start_streaming_data()

        now = then = time.time()

        for i in range(number_of_iterations):
            now = time.time()
            dt = now - then

            time.sleep(time_step)
            fxu.clear_terminal()

            motor, joint, acc, gyro, loadcell_raw = self._get_sensor_data()

            if i == 0:
                print("*** Obtaining initial value of loadcell ***")
                loadcell_zero = self._get_loadcell_zero(loadcell_raw)

            loadcell = self._get_loadcell_data(loadcell_raw, loadcell_zero)
            np.set_printoptions(suppress=True)
            print(self.knee.get_orientation(acc, gyro))

            then = now

        time.sleep(1)
        self._stop_streaming_data()

    def estance2lstance(self, data):
        if data[0] > self.knee.joint_angle_2_motor_count(45):
            return True
        else:
            return False

    def lstance2estance(self, data):
        if data[0] < self.knee.joint_angle_2_motor_count(45):
            return True
        else:
            return False

    def walk(self, duration=15, time_step=0.01):
        """
        Walks for 'n' number of seconds.
        """
        number_of_iterations = int(duration / time_step)

        self._start_streaming_data()
        self.fxs.set_gains(self.dev_id, 40, 400, 0, 0, 0, 128)
        time.sleep(0.5)

        if input("Start walking? (y/n) ") == "y":
            try:
                now = then = time.time()

                # Create states
                early_stance = State("EStance")
                late_stance = State("LStance")
                early_swing = State("ESwing")
                late_swing = State("LSwing")

                # Create events
                initial_flexion = Event("ini_flex")
                late_flexion = Event("late_flex")

                self.add_state(early_stance, initial_state=True)
                self.add_state(late_stance)
                self.add_state(early_swing)
                self.add_state(late_swing)

                self.add_event(initial_flexion)
                self.add_event(late_flexion)

                self.add_transition(
                    early_stance, late_stance, initial_flexion, self.estance2lstance
                )
                self.add_transition(
                    late_stance, early_stance, late_flexion, self.lstance2estance
                )

                self.start()

                print("Starting OSL with state: ", self.current_state.name)

                for i in range(number_of_iterations):
                    now = time.time()
                    dt = now - then

                    self.fxs.send_motor_command(self.dev_id, fxe.FX_CURRENT, 0.0)
                    time.sleep(time_step)

                    # fxu.clear_terminal()

                    motor, joint, acc, gyro, loadcell_raw = self._get_sensor_data()

                    if i == 0:
                        loadcell_zero = self._get_loadcell_zero(loadcell_raw)

                    loadcell = self._get_loadcell_data(loadcell_raw, loadcell_zero)
                    np.set_printoptions(suppress=True)

                    self.on_event(motor)
                    print(self.current_state.name)

                    then = now

                time.sleep(1)
                self._stop_streaming_data()

            except KeyboardInterrupt:
                print("Keyboard Interrupt detected, exiting the program.")
                time.sleep(1)
                self._stop_streaming_data()

    @property
    def exit_state(self):
        return self._exit_state

    @property
    def current_state(self):
        return self._current_state


if __name__ == "__main__":
    start = time.perf_counter()

    osl = OSL(port="/dev/ttyACM0", baud_rate=230400)
    osl.walk(10)

    finish = time.perf_counter()
    print(f"Script ended at {finish-start:0.4f}")
