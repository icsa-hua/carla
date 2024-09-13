from agents.navigation.dummy_application.dummy_vehicle.interface.Design import Design, VectorLike
import carla 
import pdb

class PCI(Design): 

    def __init__(self, pci) -> None:
        super().__init__(pci)
        

    def initialize_performance_parameters(self)->None:
        self.torque_curve = [PCI.to_serialize(vector) for vector in self.pci.torque_curve]
        self.steering_curve = [PCI.to_serialize(vector) for vector in self.pci.steering_curve]


    def objectify_performance_parameters(self) -> None:
        self.torque_curve = PCI.from_serialize(self.pci["torque_curve"])
        self.torque_curve = [torque for torque in self.torque_curve]
        self.steering_curve = PCI.from_serialize(self.pci["steering_curve"])


    def initialize_coefficients(self) -> None:
        self.drag_coeff = self.pci.drag_coefficient
        self.mass = self.pci.mass
        self.clutch = self.pci.clutch_strength
        self.damp_rate_full_throttle = self.pci.damping_rate_full_throttle
        self.gear_switch_time = self.pci.gear_switch_time
        self.moi = self.pci.moi
        self.max_rpm = self.pci.max_rpm


    def objectify_coefficients(self) -> None:
        self.drag_coeff = self.pci["drag_coefficient"]
        self.mass = self.pci["mass"]
        self.clutch = self.pci["clutch_strength"]
        self.damp_rate_full_throttle = self.pci["damping_rate_full_throttle"]
        self.gear_switch_time = self.pci["gear_switch_time"]
        self.moi = self.pci["moi"]
        self.max_rpm = self.pci["max_rpm"]


    def initialize_components(self) -> None:
        self.gear_autobox = self.pci.use_gear_autobox 
        self.sweep_wheel_col = self.pci.use_sweep_wheel_collision
        self.wheels = self.get_wheels(self.pci.wheels)


    def objectify_components(self) -> None:
        self.gear_autobox = self.pci["use_gear_autobox"]
        self.sweep_wheel_col = self.pci["use_sweep_wheel_collision"]
        self.wheels = self.deserialize_wheel()

    @classmethod
    def to_serialize(self, vector: VectorLike) -> dict:
        return {"x":vector.x, "y":vector.y}
    

    @classmethod
    def from_serialize(cls,data): 
        x_values = [point["x"] for point in data]
        y_values = [point["y"] for point in data]
        data = [carla.Vector2D(x,y) for x,y in zip(x_values,y_values)]
        return data
    
    
    def convert_to_map(self) -> dict:
        return { "torque_curve" : self.torque_curve,
                  "max_rpm" : self.max_rpm,
                  "moi" : self.moi,
                  "damping_rate_full_throttle" : self.damp_rate_full_throttle,
                  "use_gear_autobox" : self.gear_autobox,
                  "gear_switch_time" : self.gear_switch_time,
                  "clutch_strength" : self.clutch,
                  "mass" : self.mass,
                  "drag_coefficient" : self.drag_coeff,
                  "steering_curve" : self.steering_curve,
                  "use_sweep_wheel_collision" : self.sweep_wheel_col,
                  "wheels" : self.wheels}
    

    def convert_to_obj(self, vehicle):
        pci = vehicle.get_physics_control()
        pci.torque_curve = self.torque_curve
        pci.steering_curve = self.steering_curve
        pci.max_rpm = self.max_rpm  
        pci.moi = self.moi
        pci.damping_rate_full_throttle = self.damp_rate_full_throttle
        pci.use_gear_autobox = self.gear_autobox
        pci.gear_switch_time = self.gear_switch_time
        pci.clutch_strength = self.clutch
        pci.mass = self.mass
        pci.drag_coefficient = self.drag_coeff
        pci.use_sweep_wheel_collision = self.sweep_wheel_col
        pci.wheels = self.wheels
        return pci 
        
        
    def get_wheels(self, wheels)->list: 
        wheels_list = []
        for wheel in wheels: 
            wheels_list.append(self.serialize_wheel(wheel))
        return wheels_list
    

    def serialize_wheel(self, wheel)->dict: 
        return { "tire_friction": wheel.tire_friction,
            "damping_rate": wheel.damping_rate, 
            "max_steer_angle": wheel.max_steer_angle,
            "radius": wheel.radius,
            "max_brake_torque": wheel.max_brake_torque,
            "long_stiff_value": wheel.long_stiff_value,
            "lat_stiff_max_load": wheel.lat_stiff_max_load,
            "lat_stiff_value": wheel.lat_stiff_value
        }
    
    
    def deserialize_wheel(self):
        if self.pci['wheels'] is None: return []
        
        wheels = [ carla.WheelPhysicsControl(tire_friction=wheel["tire_friction"],damping_rate=wheel["damping_rate"],
                                             max_steer_angle=wheel["max_steer_angle"],long_stiff_value=wheel["long_stiff_value"],
                                             radius=wheel["radius"], max_brake_torque=wheel["max_brake_torque"],
                                             lat_stiff_max_load=wheel["lat_stiff_max_load"],lat_stiff_value=wheel["lat_stiff_value"])
                                             for wheel in self.pci['wheels']]
        return wheels
        

    def run(self) -> dict:
        self.initialize_coefficients()
        self.initialize_components()
        self.initialize_performance_parameters()
        return self.convert_to_map()
    
    
    def run_from_json(self, vehicle):
        if vehicle is None: return None
        self.objectify_coefficients()
        self.objectify_components()
        self.objectify_performance_parameters()
        return self.convert_to_obj(vehicle)
    

class Parts(Design): 

    def __init__(self, pci, vehicle) -> None:
        super().__init__(pci)
        self.vehicle = vehicle


    def initialize_performance_parameters(self) -> None:
        # self.ackerman_settings = self.vehicle.get_ackerman_controller_settings()
        pass


    def objectify_performance_parameters(self) -> None:
        # self.ackerman_settings = self.pci["ackerman_settings"]
        pass


    def initialize_coefficients(self) -> None:
        pass


    def objectify_coefficients(self) -> None:
        pass


    def initialize_components(self) -> None:
        self.control = self.vehicle.get_control() 
        self.throttle = self.control.throttle 
        self.brake = self.control.brake 
        self.steer = self.control.steer 
        self.hand_brake = self.control.hand_brake 
        self.reverse = self.control.reverse
        self.manual_gear = self.control.manual_gear_shift


    def objectify_components(self) -> None:
        self.control = self.vehicle.get_control() 
        self.control.throttle = self.pci['throttle']
        self.control.brake = self.pci['brake']
        self.control.steer = self.pci['steer']
        self.control.hand_brake = self.pci['hand_brake']
        self.control.reverse = self.pci['reverse']
        self.control.manual_gear_shift = self.pci['manual_gear_shift']


    def convert_to_map(self) -> dict:
        return {"steer": self.steer,
                "hand_brake":self.hand_brake,
                "brake":self.brake, 
                "throttle":self.throttle,
                "reverse":self.reverse,
                "manual_gear_shift":self.manual_gear}
    

    def to_serialize(self, vector: VectorLike) -> dict:
        return super().to_serialize(vector)
    

    def from_serialize(self, data) -> list:
        return super().from_serialize(data)


    def convert_to_obj(self):
        return self.control 
    

    def run(self) -> dict:
        if self.vehicle is None: return None
        self.initialize_coefficients()
        self.initialize_components()
        self.initialize_performance_parameters()
        return self.convert_to_map()
    
    
    def run_from_json(self):
        if self.vehicle is None: return None
        if "ackerman_settings" in self.pci.keys(): 
            self.objectify_performance_parameters()
        self.objectify_components()
        self.objectify_coefficients()
        return self.convert_to_obj()


class Mechanical(Design): 

    def __init__(self, pci) -> None:
        super().__init__(pci)
        self.details = {
            "A_coef": 0.0, 
            "B_coef": 0.0, 
            "C_coef": 0.0, 
            "d_coef": 1.1, 
            "g_coef": 9.81, 
            "ndis": 0.92, 
            "nchg": 0.63, 
            "naux_move": 0.9, 
            "naux_idle": 0.9, 
            "Paux": 0.0, 
            "nd": 0.0, 
            "n_aux": 0.0
        }

        self.time_travel = None
        self.acc_time = {}


    def initialize_coefficients(self) -> None:
        for key, value in self.pci.items(): 
            if key in self.details.keys():
                self.details[key] = value


    def objectify_coefficients(self)->None: 
        for key, value in self.vc.items(): 
            if key in self.details.keys():
                self.details[key] = value


    def initialize_components(self) -> None:                
        if self.time_travel: 
            for time in self.time_travel:
                result = self.convert_to(time)
                self.acc_time[result] = self.time_travel[time] 


    def objectify_components(self) -> None:
       self.initialize_components()

            
    def initialize_performance_parameters(self)->None: 
        pass 


    def objectify_performance_parameters(self) -> None:
        pass


    def convert_to(self, pair): 
        self.type = type(pair)
        if self.type == str: 
            return self.to_tuple(pair)
        return self.to_string(pair)


    def to_tuple(self, pair)->tuple:
        return tuple(map(int, pair.split(',')))
    

    def to_string(self, pair)->str: 
        return str(pair[0]) + "," + str(pair[1])
        

    def convert_to_map(self) -> dict:
        self.details["acc_time"] = self.acc_time
        return self.details
    

    def convert_to_obj(self):

        for key, value in self.details.items(): 
            setattr(self, key, value)
        
        setattr(self, 'acc_time', self.acc_time)
        delattr(self, 'details') 
        delattr(self,'time_travel')

    
    def to_serialize(vector: VectorLike) -> dict:
        return super().to_serialize(vector)
    
    
    def from_serialize(data) -> list:
        return super().from_serialize(data)
    

    def run(self)->dict:
        self.time_travel = self.pci['time_travel']
        self.initialize_components()
        self.initialize_coefficients()
        return self.convert_to_map()


    def run_from_json(self):
        self.vc = self.pci["vc"]
        self.time_travel = self.pci["acc_time"]
        self.objectify_coefficients()
        self.objectify_components()
        self.convert_to_obj()
