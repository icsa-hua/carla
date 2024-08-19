from abc import ABC, abstractmethod
from typing import Protocol 


class VectorLike(Protocol): 
    x: float
    y: float

class Design(ABC): 

    @abstractmethod
    def __init__(self, pci) -> None:
        self.pci = pci 

    @abstractmethod
    def initialize_performance_parameters(self)->None:
        pass 

    @abstractmethod
    def objectify_performance_parameters(self) -> None:
        pass

    @abstractmethod
    def initialize_components(self)->None: 
        pass

    @abstractmethod
    def objectify_components(self) -> None:
        pass

    @abstractmethod
    def initialize_coefficients(self)->None: 
        pass 

    @abstractmethod
    def objectify_coefficients(self) -> None:
        pass

    @abstractmethod
    def convert_to_map(self, data)->dict:
        pass

    @abstractmethod
    def to_serialize(self, vector:VectorLike)->dict: 
        pass

    @abstractmethod
    def from_serialize(self, data)->list:
        pass


    @abstractmethod
    def run(self)->dict:
        pass

    @abstractmethod
    def run_from_json(self, vehicle):
        pass
 
        


