#Create Vehicle class 
import numpy as np
import carla
import json 

class VelocityTuple:
    def __init__(self, velocity_tuple):
        self._velocity_tuple = velocity_tuple
        self._type = type(self._velocity_tuple)

    def convert_to(self):
        if self._type == str:
            return self.to_tuple()
        else:
            return self.to_string()
    
    def to_string(self): 
        vt = self._velocity_tuple
        string = str(vt[0]) + "," + str(vt[1])
        return string

    def to_tuple(self):
        string = self._velocity_tuple
        res = tuple(map(float, string.split(',')))
        return res


class VehicleDesign:
    _json = False 
    def __init__(self, *,
                 model=None, company=None, physical_controls=None, 
                 vehicle_coefficients=None, vehicle_parts=None,
                 acc_time=None): 
        self._model = model 
        self._company = company 
        if physical_controls and not VehicleDesign._json:
            self._phy_controls = physical_controls.get_map()
        else:
            self._phy_controls = physical_controls 
        if vehicle_coefficients and not VehicleDesign._json:
            self._vehicle_coeffs = vehicle_coefficients.get_map()
            self._acc_time = self._vehicle_coeffs["acc_time"]
            del self._vehicle_coeffs["acc_time"]
        else: 
            self._vehicle_coeffs = vehicle_coefficients
            self._acc_time = acc_time
        if vehicle_parts and not VehicleDesign._json:
            self._vehicle_parts = vehicle_parts.get_map()
        else:
            self._vehicle_parts = vehicle_parts

    def to_dict(self): 
        return {
            "model": self._model, 
            "company": self._company,
            "physical_controls": self._phy_controls,
            "vehicle_coefficients": self._vehicle_coeffs,
            "vehicle_parts": self._vehicle_parts,
            "acc_time": self._acc_time
        }
    
    def to_json(self):
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_string):
        VehicleDesign._json = True
        data = json.loads(json_string)
        return cls(**data) 

    @classmethod
    def from_json_file(cls, file_path):
        VehicleDesign._json = True
        with open(file_path, 'r') as file: 
            data = json.load(file)
        return cls(**data) 
    
    def save_to_json_file(self,file_path): 
        with open(file_path, 'w') as file: 
            json.dump(self.to_dict(), file, indent=4)

    def to_serializable(vector):
        return {"x":vector.x, "y":vector.y}
    
    @classmethod
    def from_serializable(cls,data):
        x_values = [point["x"] for point in data]
        y_values = [point["y"] for point in data]
        data = [carla.Vector2D(x,y) for x,y in zip(x_values,y_values)]
        return data

class PhysicalControls(VehicleDesign):
    def __init__(self, physics_control):
        self._physical_control = physics_control
        self._torque_curve = [VehicleDesign.to_serializable(vector) for vector in self._physical_control.torque_curve]
        self._max_rpm = self._physical_control.max_rpm
        self._moi = self._physical_control.moi
        self._damping_rate_full_throttle = self._physical_control.damping_rate_full_throttle
        self._use_gear_autobox = self._physical_control.use_gear_autobox
        self._gear_swith_time = self._physical_control.gear_switch_time
        self._clutch_strength = self._physical_control.clutch_strength
        self._mass = self._physical_control.mass    
        self._drag_coefficient = self._physical_control.drag_coefficient    
        self._steering_curve = [VehicleDesign.to_serializable(vector) for vector in self._physical_control.steering_curve]
        self._use_sweep_wheel_collision = self._physical_control.use_sweep_wheel_collision
        self._wheels_vector = self._physical_control.wheels
        self._wheels = []
        wheel_counter = 0

        for wheel in self._wheels_vector:
            tmp = {"tire_friction": wheel.tire_friction,
                   "damping_rate":wheel.damping_rate, 
                   "max_steer_angle":wheel.max_steer_angle,
                   "radius": wheel.radius,
                   "max_brake_torque": wheel.max_brake_torque,
                   "long_stiff_value": wheel.long_stiff_value,
                   "lat_stiff_max_load": wheel.lat_stiff_max_load,
                   "lat_stiff_value": wheel.lat_stiff_value}
            self._wheels.append(tmp)

    def get_map(self):    
        return  { "torque_curve" : self._torque_curve,
                  "max_rpm" : self._max_rpm,
                  "moi" : self._moi,
                  "damping_rate_full_throttle" : self._damping_rate_full_throttle,
                  "use_gear_autobox" : self._use_gear_autobox,
                  "gear_switch_time" : self._gear_swith_time,
                  "clutch_strength" : self._clutch_strength,
                  "mass" : self._mass,
                  "drag_coefficient" : self._drag_coefficient,
                  "steering_curve" : self._steering_curve,
                  "use_sweep_wheel_collision" : self._use_sweep_wheel_collision,
                  "wheels" : self._wheels}

class VehicleCoefficients(VehicleDesign): 
    def __init__(self, *, 
                 A_coef=None, B_coef=None, C_coef=None, 
                 d_coef=1.1,g_coef=9.81, 
                 ndis=0.92, nchg=0.63,
                 naux_move=0.9, naux_idle=0.9, 
                 P_aux=None, nd=0, n_aux=0 ,
                 velocities=None, time_travel=None):
        self._A_coef = A_coef
        self._B_coef = B_coef
        self._C_coef = C_coef 
        self._d_coef = d_coef 
        self._g_coef = g_coef 
        self._ndis = ndis
        self._nchg = nchg
        self._naux_move = naux_move
        self._naux_idle = naux_idle
        self._Paux = P_aux
        self._nd = nd 
        self._n_aux = n_aux 
        self._velocities = velocities
        self.time_travel = time_travel
        self._acc_time = {}
        if time_travel:
            for time in self.time_travel:
                var = VelocityTuple(time)
                result = var.convert_to()
                self._acc_time[result] = self.time_travel[time]
            
    def get_map(self):    
        return {"A_coef":self._A_coef, 
                "B_coef":self._B_coef,
                "C_coef":self._C_coef,
                "d_coef":self._d_coef,
                "g_coef":self._g_coef,
                "ndis":self._ndis,
                "nchg":self._nchg,
                "naux_move":self._naux_move,
                "naux_idle":self._naux_idle,
                "Paux":self._Paux,
                "nd":self._nd,
                "n_aux":self._n_aux,
                "acc_time":self._acc_time}
        

class VehicleParts(VehicleDesign):
    def __init__(self,vehicle):
        self._vehicle = vehicle
        self._ackerman_settings = self._vehicle.get_ackermann_controller_settings()
        self._vehicle_control = self._vehicle.get_control()
        self._throttle = self._vehicle_control.throttle 
        self._steer = self._vehicle_control.steer 
        self._brake = self._vehicle_control.brake
        self._hand_break = self._vehicle_control.hand_brake
        self._reverse = self._vehicle_control.reverse 
        self._manual_gear_shift = self._vehicle_control.manual_gear_shift
        
        # print("VehicleParts")
        # print(type(self._steer))
        # print(type(self._brake))
        # print(type(self._hand_break))
        # print(type(self._throttle))
        # print(type(self._reverse))
        # print(type(self._manual_gear_shift))
        
    def get_map(self):
        return {"steer": self._steer,
                "hand_brake":self._hand_break,
                "brake":self._brake, 
                "throttle":self._throttle,
                "reverse":self._reverse,
                "manual_gear_shift":self._manual_gear_shift}
        

    


