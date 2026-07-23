import ctypes

# Define Basic Float3 Structure
class Float3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]

# Define Wheel Data Structure
class CaiWheelData(ctypes.Structure):
    _fields_ = [
        ("position", Float3),
        ("contact_point", Float3),
        ("contact_normal", Float3),
        ("look", Float3),
        ("side", Float3),
        ("velocity", Float3),
        ("slip_ratio", ctypes.c_float),
        ("load", ctypes.c_float),
        ("pressure", ctypes.c_float),
        ("angular_velocity", ctypes.c_float),
        ("wear", ctypes.c_float),
        ("dirty_level", ctypes.c_float),
        ("core_temperature", ctypes.c_float),
        ("camber_rad", ctypes.c_float),
        ("disc_temperature", ctypes.c_float),
        ("slip", ctypes.c_float),
        ("slip_angle_deg", ctypes.c_float),
        ("nd_slip", ctypes.c_float),
    ]

# Define Car State Structure
class CaiCarData(ctypes.Structure):
    _fields_ = [
        ("packet_id", ctypes.c_int),
        ("gas", ctypes.c_float),
        ("brake", ctypes.c_float),
        ("clutch", ctypes.c_float),
        ("steer", ctypes.c_float),
        ("handbrake", ctypes.c_float),
        ("fuel", ctypes.c_float),
        ("gear", ctypes.c_int),
        ("rpm", ctypes.c_float),
        ("speed_kmh", ctypes.c_float),
        ("velocity", Float3),
        ("acc_g", Float3),
        ("look", Float3),
        ("up", Float3),
        ("position", Float3),
        ("local_velocity", Float3),
        ("local_angular_velocity", Float3),
        ("cg_height", ctypes.c_float),
        ("car_damage", ctypes.c_float * 5),
        ("wheels", CaiWheelData * 4),
        ("turbo_boost", ctypes.c_float),
        ("final_ff", ctypes.c_float),
        ("final_pure_ff", ctypes.c_float),
        ("pit_limiter", ctypes.c_bool),
        ("abs_in_action", ctypes.c_bool),
        ("traction_control_in_action", ctypes.c_bool),
        ("lap_time_ms", ctypes.c_uint),
        ("best_lap_time_ms", ctypes.c_uint),
        ("drivetrain_torque", ctypes.c_float),
        ("spline_position", ctypes.c_float),
        ("collision_depth", ctypes.c_float),
        ("collision_counter", ctypes.c_uint),
        ("wheels_valid_surface", ctypes.c_uint),
    ]

# Define Car Control Structure
class CaiCarControls(ctypes.Structure):
    _fields_ = [
        ("gas", ctypes.c_float),
        ("brake", ctypes.c_float),
        ("clutch", ctypes.c_float),
        ("steer", ctypes.c_float),
        ("handbrake", ctypes.c_float),
        ("gear_up", ctypes.c_bool),
        ("gear_dn", ctypes.c_bool),
        ("drs", ctypes.c_bool),
        ("kers", ctypes.c_bool),
        ("brake_balance_up", ctypes.c_bool),
        ("brake_balance_dn", ctypes.c_bool),
        ("abs_up", ctypes.c_bool),
        ("abs_dn", ctypes.c_bool),
        ("tc_up", ctypes.c_bool),
        ("tc_dn", ctypes.c_bool),
        ("turbo_up", ctypes.c_bool),
        ("turbo_dn", ctypes.c_bool),
        ("engine_brake_up", ctypes.c_bool),
        ("engine_brake_dn", ctypes.c_bool),
        ("mguk_delivery_up", ctypes.c_bool),
        ("mguk_delivery_dn", ctypes.c_bool),
        ("mguk_recovery_up", ctypes.c_bool),
        ("mguk_recovery_dn", ctypes.c_bool),
        ("mguh_mode", ctypes.c_ubyte),
        ("headlights", ctypes.c_bool),
        ("teleport_to", ctypes.c_ubyte),
        ("autoclutch_on_start", ctypes.c_bool),
        ("autoclutch_on_change", ctypes.c_bool),
        ("autoblip_active", ctypes.c_bool),
        ("teleport_pos", Float3),
        ("teleport_dir", Float3),
        ("autoshift_active", ctypes.c_bool),
    ]

class CaiSimControl(ctypes.Structure):
    _fields_ = [
        ("pause", ctypes.c_bool),
        ("restart_session", ctypes.c_bool),
        ("disable_collisions", ctypes.c_bool),
        ("extra_sleep_ms", ctypes.c_ubyte),
    ]