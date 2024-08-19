import logging
import carla 
import json 
import numpy as np 
from agents.navigation.dummy_application.dummy_vehicle.utils.Components import PCI,Parts,Mechanical 
from agents.navigation.dummy_application.dummy_vehicle.interface.VehicleDesign import VehicleDesign

class Entity(VehicleDesign): 

    _json = None

    def __init__(self, **kwargs ) -> None:
        for key, value in kwargs.items(): 
            setattr(self, key, value)

        self.attr_names = kwargs.keys()

    def to_dict(self) -> dict:
        return {
            "model": self.model, 
            "company": self.company,
            "physical_controls": self.pci, 
            "vehicle_coefficients": self.vc,
            "vehicle_parts": self.vp,
            "acc_time": self.vc.acc_time
        }

    def to_json(self)->json:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_string:str):
        Entity._json = True
        data = json.loads(json_string)
        return cls(**data)
        

    @classmethod
    def from_json_file(cls, file_path): 
        Entity._json = True
        with open(file_path, 'r') as file: 
            data = json.load(file)
        return cls(**data)


    def save_to_json_file(self,file_path)->None:  
        with open(file_path, 'w') as file: 
            json.dump(self.to_dict(), file, indent=4)


    def set_PCI(self, data, vehicle)->None: 
        logging.info("PCI CREATED with type: {}".format(type(data)))
        if isinstance(data, dict) and Entity._json :
            logging.info("JSON DATA")
            pci = PCI(data)
            self.pci = pci.run_from_json(vehicle)   
            return None
        
        pci = PCI(data)
        self.pci = pci.run()  
        

    def set_VP(self, data, vehicle)->None: 

        if isinstance(data,dict) and Entity._json: 
            vp = Parts(data, vehicle)
            self.vp = vp.run_from_json() 
            return None
        
        vp = Parts(data, vehicle)
        self.vp = vp.run()


    def set_VC(self, data)->None: 

        if isinstance(data,dict) and Entity._json: 
            self.vc = Mechanical(data)
            self.vc.run_from_json() 
            return None
        
        vc = Mechanical(data)
        self.vc = vc.run() 


    def get_physical_controls(self): 
        return self.pci 
    
    def get_coefficients(self): 
        return self.vc
    
    def get_components(self): 
        return self.vp
        
