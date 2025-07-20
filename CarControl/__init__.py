import mmap
import ctypes
import time
import math
from .data_types import Float3, CaiWheelData, CaiCarData, CaiCarControls, CaiSimControl


class CarController:
    def __init__(self, car_id=0):
        self.car_id = car_id
        self.car_state_file = f"AcTools.CSP.NewBehaviour.CustomAI.Car{car_id}.v0"
        self.car_control_file = f"AcTools.CSP.NewBehaviour.CustomAI.CarControls{car_id}.v0"
        self.sim_state_file = "AcTools.CSP.NewBehaviour.CustomAI.SimState.v0"
        self.control_memory = mmap.mmap(-1, ctypes.sizeof(CaiCarControls), tagname=self.car_control_file, access=mmap.ACCESS_WRITE)
        self.sim_control_memory = mmap.mmap(0, ctypes.sizeof(CaiSimControl), tagname=self.sim_state_file, access=mmap.ACCESS_WRITE)

    def write_car_controls(self, gas=0.0, brake=0.0, steer=0.0):
        self.control_memory.seek(0)
        car_controls = CaiCarControls(gas=gas, brake=brake, steer=steer, gear_up=False, gear_dn=False, autoclutch_on_start=True, autoclutch_on_change=True, autoblip_active=True, autoshift_active=True)
        self.control_memory.write(bytearray(car_controls))
        self.control_memory.flush()

    def write_car_state(self):
        car_state_memory = mmap.mmap(-1, ctypes.sizeof(CaiCarData), tagname=self.car_state_file, access=mmap.ACCESS_WRITE)
        car_state_memory.seek(0)
        car_state = self.read_car_state()
        car_state.speed_kmh = 100
        car_state_memory.write(bytearray(car_state))
        car_state_memory.flush()
        car_state_memory.close()

    def teleport(self, x, y, z, lookx, looky, lookz, teleport_to=0):
        pos = Float3(x=x, y=y, z=z)
        _dir = Float3(x=lookx, y=looky, z=lookz)
        self.control_memory.seek(0)
        car_controls = CaiCarControls(
            teleport_pos=Float3(x, y, z), teleport_dir=_dir, teleport_to=ctypes.c_ubyte(teleport_to)
        )
        self.control_memory.write(bytearray(car_controls))
        self.control_memory.flush()
        self.write_car_state()
        time.sleep(1)
        print("speedddd: ",self.read_car_state().speed_kmh)

    def read_car_state(self):
        try:
            car_state_memory = mmap.mmap(-1, ctypes.sizeof(CaiCarData), tagname=self.car_state_file, access=mmap.ACCESS_READ)
            car_state_memory.seek(0)
            car_state = CaiCarData()
            ctypes.memmove(ctypes.addressof(car_state), car_state_memory.read(ctypes.sizeof(CaiCarData)), ctypes.sizeof(CaiCarData))
            car_state_memory.close()
        except Exception as e:
            print(f"Error reading car state: {e}")
            car_state = None
        return car_state

    def read_car_state2(self):
        physics_memory = mmap.mmap(-1, ctypes.sizeof(SPageFilePhysics), "Local\\acpmf_physics", access=mmap.ACCESS_READ)
        physics = SPageFilePhysics()
        ctypes.memmove(ctypes.addressof(physics), physics_memory.read(ctypes.sizeof(SPageFilePhysics)), ctypes.sizeof(SPageFilePhysics))
        physics_memory.close()
        return physics

    def write_session_controls(self, pause=False, restart_session=False, disable_collisions=False, extra_sleep_ms=0):
        try:
            self.sim_control_memory.seek(0)
            # Convert extra_sleep_ms to a byte (0-255)
            extra_sleep_ms_byte = min(max(0, extra_sleep_ms), 255)
            sim_control = CaiSimControl(
                pause=pause,
                restart_session=restart_session,
                disable_collisions=disable_collisions,
                extra_sleep_ms=extra_sleep_ms_byte
            )
            self.sim_control_memory.write(bytearray(sim_control))
            self.sim_control_memory.flush()
            print("Session controls written successfully.")
        except Exception as e:
            print(f"Error writing session controls: {e}")

    def read_session_controls(self):
        try:
            self.sim_control_memory.seek(0)
            sim_control = CaiSimControl()
            ctypes.memmove(ctypes.addressof(sim_control), self.sim_control_memory.read(ctypes.sizeof(CaiSimControl)), ctypes.sizeof(CaiSimControl))
            self.sim_control_memory.close()
        except Exception as e:
            print(f"Error reading session controls: {e}")
            sim_control = None
        return sim_control
