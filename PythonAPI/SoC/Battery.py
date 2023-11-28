#This will be the vehicle Class Model + battery

import math as m
from os import R_OK
import numpy as np
from utils import Polynomial
import matplotlib.pyplot as plt

class Vehicle_Audi: 
    def __init__(self, car_model, audi_car,agent):
        self.car_model = car_model 
        self.vehicle = audi_car
        self._agent = agent 
        self.battery_type = "LiIon"
        self.Ncells = 432 
        self.Nmodules = 36 
        self.Ncells_in_module = 12 
        self.cell_Volt = 0.92 
        self.cell_Cap = 240 
        self.bat_Volt = 396 
        self.bat_Cap = 240 
        self.bat_energy = 95 
        self.Nmotors = 2 
        self.motorFR = 33.34 
        self.motorPoles = 4 
        self.rpm = (120*self.motorFR)/self.motorPoles 
        self.crub_weight = 2490 
        self.driver_weight = 75 
        self.total_weight = self.crub_weight + self.driver_weight
        self.max_brake_regen = 250 
        self.drag_coeff = 0.27 
        self.wheelbase = 0.2928
        self.transmission_eff = 0.95 #percentage also n_G 
        self.single_speed_G_ratio = 9/1  
        self.top_speed = 200 
        self.acc_time = 5.7 
        self.NEDC_range = 328 
        self.EPA_range = 328.3 
        self.NEDC_ec = (24.2,100)
        self.EPA_ec = (46,100)
        self.total_power = 300 
        self.total_torque = 664 
        self.phi = 0.8 
        self.gravity_Acc = 9.81 
        self.tire_diameter = 0.4064
        self.tire_radius = self.tire_diameter/2
        self.speed_ratio = 1 
        self._max_brake = 0.5 

    def brake_strategy(self):
        #depending on the EPA UDDS the vehicle mass is mass + 5% of that mass due to inertia. 
        vehicle_mass = self.crub_weight+ (0.05)*self.crub_weight
        XBMAX = self.phi * vehicle_mass * self.gravity_Acc
        control = self._agent.run_step() 
        DBfriction = control.throttle
        DBbrake = control.brake
        XBfriction = DBfriction*XBMAX
        XBbrake = DBbrake*XBMAX
        current_speed = self.vehicle.get_velocity()
        PBRmax = self.transmission_eff * self.max_brake_regen
        PBrDemanded = (0.5)*vehicle_mass*0.001*(current_speed^2)/3600
        TBRlimited = 0
        W_motor = 0
        if PBRmax >= PBrDemanded:
            if W_motor == 0 :
                TBRlimited = 0
            else: 
                TBRlimited = PBRmax/(W_motor*self.Nmotors)
        else: 
            if W_motor ==0: 
                TBRlimited = 0
            else:
                TBRlimited = PBrDemanded/(W_motor*self.Nmotors)

        EM_available = (TBRlimited * self.single_speed_G_ratio * self.transmission_eff)/(self.tire_radius*XBMAX)

        
class battery: 
    def __init__(self, total_capacity, R_0, R_1, C_1, R_2, C_2): 
        self.total_capacity = total_capacity * 3600 #Maybe Ah? 
        self.actual_capacity = self.total_capacity

        #Battery based on Thevenin Model : OCV + R0 + R1 // C1 
        self.R0 = R_0 
        self.R1 = R_1 
        self.C1 = C_1 
         
    
        self._current = 0 
        self._RC_voltage = 0 
        
        self._OCV_model = Polynomial([3.1400, 3.9905, -14.2391, 24.4140, -13.5688, -4.0621, 4.5056])

    def update(self, time_delta):
        self.actual_capacity -= self.current * time_delta #time step 
        exp_coeff = m.exp(-time_delta/(self.R1*self.C1))
        self._RC_voltage *= exp_coeff
        self._RC_voltage += self.R1*(1-exp_coeff)*self.current


    @property
    def current(self):
        return self._current

    @current.setter
    def current(self,current):
        self._current = current 

    @property
    def voltage(self):
        return self.OCV - self.R0 * self.current - self._RC_voltage # non - linear vector function g but needs the second rc voltage for the 

    @property
    def state_of_charge(self):
        return self.actual_capacity/self.total_capacity

    @property 
    def OCV_model(self): 
        return self._OCV_model

    @property 
    def OCV(self):
        return self.OCV_model(self.state_of_charge)

if __name__ == '__main__':
    capacity = 240 #our value
    disharge_rate = 1 
    time_step = 10
    cut_off_voltage = 200 

    current = capacity * disharge_rate
    veh_battery = battery(capacity,0.062,0.01,3000)
    veh_battery.current = current 
    time = [0] 
    SoC = [veh_battery.state_of_charge]
    OCV = [veh_battery.OCV]
    RC_volt = [veh_battery._RC_voltage]
    voltage = [veh_battery.voltage]

    while veh_battery.voltage > cut_off_voltage:
        veh_battery.update(time_step)
        time.append(time[-1]+time_step)
        SoC.append(veh_battery.state_of_charge)
        OCV.append(veh_battery.OCV)
        RC_volt.append(veh_battery._RC_voltage)
        voltage.append(veh_battery.voltage)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # title, labels
    ax1.set_title('')    
    ax1.set_xlabel('SoC')
    ax1.set_ylabel('Voltage')

    ax1.plot(SoC, OCV, label="OCV")
    ax1.plot(SoC, voltage, label="Total voltage")

    plt.show()
