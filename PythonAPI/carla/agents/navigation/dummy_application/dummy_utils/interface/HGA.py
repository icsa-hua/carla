from abc import ABC, abstractmethod
import networkx as nx 


class HGA(ABC):

    @abstractmethod
    def __init__(self, origins, map, specs )->None: 
        self.origins = origins
        self.map = map 
        self.specs = specs 


    @abstractmethod
    def build_topology(self)->None: 
        """
        Function that follows the same concept as in the basic_agent.py 
        """
        pass 
    

    @abstractmethod
    def create_graph(self)->None: 
        """
        Function that follows the same concept as in the basic_agent.py
        """
        pass


    @abstractmethod
    def find_loose_ends(self)->None:
        """
        Function that follows the same concept as in the basic_agent.py
        """
        pass


    @abstractmethod
    def lane_change_link(self)->None:
        """
        Function that follows the same concept as in the basic_agent.py
        """
        pass


    @abstractmethod
    def _localize(self,location) -> tuple: 
        """
        Function that follows the same concept as in the basic_agent.py
        """
        pass


    @abstractmethod
    def find_valid_paths(self, graph:nx.Graph, origins:dict, path:list, max_depth:int) -> list: 
        """
        Process the created path solutions based on the graph and initial waypoints state. 
        """
        pass 


    @abstractmethod
    def fitness_calculation(self,path:list) -> list: 
        """
        Calculate the cost of a possible path solution and store it in a list.
        The cost can be the aggregation of different metrics such as time, distance, etc.
        """
        pass


    @abstractmethod
    def initialize_population(self) ->list: 
        """
        Create the initial list of paths that will be the search space for the genetic algorithm.
        """
        pass 


    @abstractmethod 
    def selection(self, fitness_value:list) -> list: 
        """
        Selects a subset of individuals from the population based on their fitness values.
        A common approach is to use a tournament selection process, where a subset of individuals
        is randomly selected from the population, and the individual with the best (lowest) fitness value
        in the tournament is selected to be part of the next generation.
        """
        pass


    @abstractmethod 
    def crossover(self, ind1, ind2)->list: 
        """
        Performs a crossover operation on two individuals (paths) in the genetic algorithm.
        
        The crossover operation combines the common nodes between the two individuals, while
        ensuring that the resulting child path is valid (i.e., all nodes are connected).
        """
        pass 


    @abstractmethod
    def mutation(self,ind): 
        """
        Performs a mutation operation on the given individual (path) in the genetic algorithm.
        
        The mutation operation swaps two nodes in the path, as long as the resulting path is still valid (i.e., all edges between the swapped nodes exist in the graph).
        
        """
        pass


    @abstractmethod
    def run(self, rate:float)->list: 
        """
        Exectue the HGA algorithm to find the optimal path from the origins to the destination.
        """
        pass