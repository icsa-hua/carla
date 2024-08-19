from abc import ABC, abstractmethod
import numpy as np 
import networkx as nx 


class EnergyEstimator(ABC): 

    @abstractmethod
    def __init__(self, vehicle=None, vehicle_coefficients=None, map=None, origins=None):
        self.vehicle = vehicle 
        self.vehicle_specs =  vehicle_coefficients
        self.map = map
        self.origins = origins

    @classmethod
    def get_average_acceleration(self, initial_velocity=0, final_velocity=27.78, time_of_travel=5.7, distance_of_travel=0): 
        pass 

    @abstractmethod
    def phase_calculation(self, phase_acc:float, phase_speed:float, phase_distance:float, phase_time:float, phase_avg_end_speed:float, avg_slope:float)->float:
        pass

    @abstractmethod
    def link_creation(self) -> list:
        pass

    @abstractmethod
    def fix_loop_route(self)->list:
        pass 

    @abstractmethod
    def calculate_slope(self, source_node:int, target_node:int)->float:
        pass

    @abstractmethod
    def calculate_distance(self, source_node:int, target_node:int)->float:
        pass

    @abstractmethod
    def Douglas_Peucker_algorithm(self, points:list, epsilon:float)->list:
        pass

    @abstractmethod
    def run(self): 
        pass 

