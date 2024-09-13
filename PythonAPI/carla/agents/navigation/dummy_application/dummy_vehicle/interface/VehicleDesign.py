from abc import ABC, abstractmethod
from agents.navigation.dummy_application.dummy_vehicle.interface.Design import Design
import json

# @classmethod Takes the class as the first argument to change the state of the class as cls
# use only @abstractmethod for methods to be used by objects of sub classes.  

class VehicleDesign(ABC) : 

    @abstractmethod
    def __init__(self, model, company) -> None:
        self.model = model
        self.company = company  


    @abstractmethod
    def to_dict(self) -> dict: 
        pass 


    @abstractmethod
    def to_json(self)->json: 
        pass 


    @abstractmethod
    def from_json(self, json_string:str)->dict:
        pass


    @classmethod
    def to_serializable(vector)->dict:
        pass


    @classmethod
    def from_serializable(cls, data)->list: 
        pass


    @abstractmethod
    def get_physical_controls(self,data)->dict:
        pass 


    @abstractmethod
    def get_coefficients(self, design)->dict:
        pass 


    @abstractmethod
    def get_components(self, parts)->dict: 
        pass 

