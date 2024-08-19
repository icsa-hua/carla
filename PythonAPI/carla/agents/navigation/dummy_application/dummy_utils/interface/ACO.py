from abc import ABC, abstractmethod
import networkx as nx 

class AntColonyOptim(ABC): 
    
    @abstractmethod
    def __init__(self, origins, graph:nx.Graph, specs:dict) -> None:
        super().__init__()
        self.origins = origins
        self.graph = graph 
        self.specs = specs 
        self.pheromone_matrix = {}


    @abstractmethod
    def initialize_pheromone(self)->None: 
        """
        Initializes the pheromone matrix for the Ant Colony Optimization (ACO) algorithm.
        It is responsible for setting the initial pheromone values in the pheromone matrix,
        which is used by the ACO algorithm to guide the ants towards optimal solutions.
        """
        pass


    @abstractmethod
    def update_pheromone(self, path:list)->None: 
        """
        Update of the pheromone matrix based on the evaporation rate and the amount of pheromone deposited
        Ensure when running this that the edges exist in the graph that the ACO algorithm is running on
        """
        pass 


    @abstractmethod
    def _select_next_node(self, current_node:int, visited_nodes:list) -> int:
        """
        Selection on the next node is a culmination of the amount of pheromone that exists on an edge, 
        and the visibility, a sum of specs and edge cost attributes. 
        """
        pass


    @abstractmethod
    def fitness_calculation(self, path:list) -> float:
        """
        Calculation of the cost expended for a path based on an aggregetion 
        of metrics, e.g. distance, energy, time etc. 
        """
        pass


    @abstractmethod
    def run(self)->list: 
        """
        Execute the ACO algorithm to find the shortest path between the specified origins.
        """
        pass
