from agents.navigation.dummy_application.dummy_utils.interface.ACO import AntColonyOptim
import networkx as nx
import numpy as np 
import heapq 
import warnings 
import random 
import pdb 


class AntColony(AntColonyOptim): 
    def __init__(self, origins, graph, specs): 
        super().__init__(origins, graph, specs)
        self._source = self.origins['start']
        self._destination = self.origins['end']
        self.max_recursions = self.specs['max_recursions']


    def initialize_pheromone(self):
        for u, v in self.graph.edges():
            self.pheromone_matrix[(u,v)] = self.specs['initial_pheromone']


    def update_pheromone(self, path: list):
        path_cost = self.fitness_calculation(path)
        for i in range(len(path)-1):
            edge = (path[i], path[i+1])
            if edge in self.pheromone_matrix:
                self.pheromone_matrix[edge] *= (1-self.specs['evaporation_rate'])
                self.pheromone_matrix[edge]  += self.specs['pheromone_deposit'] / path_cost
            
    
    def fitness_calculation(self, path: list) -> float:
        path_cost = 0
        penalty = 1000

        for i in range(len(path) - 1):
            source_node = path[i]
            target_node = path[i + 1]

            if self.graph.has_edge(source_node, target_node):
                edge_attributes = self.graph.get_edge_data(source_node, target_node)
                path_cost += edge_attributes['time'] + edge_attributes['ease_of_driving'] + edge_attributes['weight']
            else:
                path_cost += penalty

        return path_cost


    def _select_next_node(self, current_node: int, visited_nodes: list) -> int:
        self._probabilities = []
        edges = list(self.graph.edges(current_node, data=True))

        for edge in edges:
            next_node = edge[1]
            if next_node not in visited_nodes:
                edge_key = (edge[0], next_node) if (edge[0], next_node) in self.pheromone_matrix else (next_node, edge[0])
                pheromone = self.pheromone_matrix[edge_key]
                visibility = (self.specs['alpha'] / edge[2]['weight']) + (self.specs['beta'] / edge[2]['time']) + (self.specs['gamma'] / edge[2]['ease_of_driving'])
                probability = (pheromone ** self.specs['alpha']) * (visibility ** self.specs['beta'])
                self._probabilities.append(probability)

        total = sum(self._probabilities)

        if total > 0:
            self._probabilities = [p / total for p in self._probabilities]
            choice = [edge[1] for edge in edges if edge[1] not in visited_nodes]
            next_node = np.random.choice(choice, p=self._probabilities)
        else:
            next_node = self._source[0]
            visited_nodes.clear()

        return next_node
        
    
    def run(self):
        self.initialize_pheromone()
        population = [] 
        path_set = set()
        bar_length = 50 

        for iteration in range(self.specs['iterations']): 
            percent = iteration / self.specs['iterations'] * 100 
            filled_length = int(percent / 2) 
            bar = '#' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r|{bar}| {percent:.1f}%', end='\r')

            for _ in range(self.specs['number_of_ants']): 
                path = [self._source[0]]
                current_node = self._source[0]
                visited_nodes = [] 

                while current_node!= self._destination[0]:
                    next_node = self._select_next_node(current_node, visited_nodes)
                    visited_nodes.append(next_node)
                    path.append(next_node)
                    current_node = next_node

                self.update_pheromone(path)

                if self.is_valid_path(path): 
                    path_hash = hash(tuple(path))
                    if path_hash not in path_set: 
                        heapq.heappush(population, (self.fitness_calculation(path), path))
                        path_set.add(path_hash)
                
            while len(population) > self.specs['population_limit']: 
                heapq.heappop(population)

        population_paths = [path for cost, path in population]

        if self.max_recursions >= 0:
            self.max_recursions -= 1
            if len(population_paths) <= 5: 
                self.specs['number_of_ants'] = random.randint(10,15) 
                self.specs['iterations'] = self.specs['number_of_ants'] + 5 
                self.specs['evaporation_rate'] = np.round(random.uniform(0.5, 0.8), 2)
                self.specs['pheromone_deposit'] = np.round(random.uniform(200.00, 250.00), 2)
                self.specs['alpha'] = np.round(random.uniform(0.1,0.25), 2)
                self.specs['beta'] = np.round(random.uniform(0.5,0.75), 2)
                self.specs['gamma'] = 1 - (self.specs['alpha'] + self.specs['beta'])
                return self.run()
            
            else: 
                print("\nFinished Initialization of population")
                return population_paths                                                              
        else: 
            warnings.warn("No initial paths found after 5 consecutive ACO executions.\nCheck the Start and Finish point locations.")
            return []

    def is_valid_path(self, path:list)->bool: 
        for i in range(len(path)-1):
            sn = path[i]
            tn = path[i+1]
            if not self.graph.has_edge(sn,tn):
                return False
        return True