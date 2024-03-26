#ACO import for the Ant Colony Optimization Approach to be used for the initial population generation for the GA. 
import numpy as np
import networkx as nx
import time

class ACO():
    def __init__(self,*,
                 graph=None,
                 source=None, target=None,
                 num_ants=10,iterations=10,
                 initial_pheromone=0.1, alpha=1.0,
                 beta=1.0, gamma=1.0, evaporation_rate = 0.5,
                 pheromone_deposit=100.0):
        self._graph = graph
        self._source = source
        self._target = target
        self._num_ants = num_ants
        self._iterations = iterations
        self._initial_pheromone = initial_pheromone
        self._alpha = alpha 
        self._beta = beta
        self._gamma = gamma
        self._evaporation_rate = evaporation_rate
        self._pheromone_deposit = pheromone_deposit
        self._pheromone_matrix = {}
        self._probabilities = []
        
    def calculate_path_cost(self, path):

        path_cost = 0
        for i in range(len(path) - 1):
            source_node = path[i]
            target_node = path[i+1]
            if self._graph.has_edge(source_node,target_node):
                edge_attributes = self._graph.get_edge_data(source_node,target_node)
                path_cost += edge_attributes['time'] + edge_attributes['ease_of_driving'] + edge_attributes['weight']
            else: 
                #Handle the case for where the edge does not exist for example assign a penalty 
                path_cost += 750
        
        return path_cost

    def _initialize_pheromone_matrix(self):
        for u, v in self._graph.edges():
            self._pheromone_matrix[(u,v)] = self._initial_pheromone
    
    def _choose_next_node(self, current_node, visited_nodes):
        
        self._probabilities = []
        edges = list(self._graph.edges(current_node, data=True))

        for edge in edges: 
            if (edge[1]) not in visited_nodes:
                if (edge[0], edge[1]) not in self._pheromone_matrix: 
                    pheromone = self._pheromone_matrix[(edge[1],edge[0])]
                else: 
                    pheromone = self._pheromone_matrix[(edge[0], edge[1])]
                
                visibility = (self._alpha /edge[2]['weight']) + (self._beta / edge[2]['time']) + (self._gamma / edge[2]['ease_of_driving'])
                self._probabilities.append((pheromone ** self._alpha)*(visibility ** self._beta))
        
        total = sum(self._probabilities)
        
        if total > 0 : 
            self._probabilities = [p / total for p in self._probabilities]
            choice = [edge[1] for edge in edges if edge[1] not in visited_nodes ]
            next_node = np.random.choice(choice, p=self._probabilities)

        elif total <= 0:
            next_node = self._source[0] 
            visited_nodes.clear()
            #Restart the path

        return next_node
            
    def _update_pheromone_matrix(self, paths):
        
        for edge in self._pheromone_matrix:
            self._pheromone_matrix[edge] *= (1-self._evaporation_rate)
        
        for path in paths: 
            path_cost = self.calculate_path_cost(path)
            
            for i in range(len(path)-1):
                
                if self._graph.has_edge(path[i], path[i+1]):

                    if (path[i], path[i+1]) not in self._pheromone_matrix:
                        self._pheromone_matrix[(path[i+1], path[i])] += self._pheromone_deposit / path_cost
                    else:
                        self._pheromone_matrix[(path[i], path[i+1])] += self._pheromone_deposit / path_cost
    
    def _ant_colony_optimization(self):
        self._initialize_pheromone_matrix()
        best_paths = []
        for iteration in range(self._iterations):
            percent = iteration / self._iterations * 100
            filled_length = int(percent / 2)
            bar = '=' * filled_length + '-' * (50 - filled_length)
            print(f'\r|{bar}| {percent:.1f}%', end='\r')
            time.sleep(0.1)
            paths = [] 
            for ant in range(self._num_ants):
                path = [self._source[0]]
                current_node = self._source[0] 
                visited_nodes = []
                while current_node != self._target[0]:
                    next_node = self._choose_next_node(current_node=current_node, visited_nodes=visited_nodes)
                    visited_nodes.append(next_node)
                    path.append(next_node)
                    current_node = next_node
                paths.append(path)
            self._update_pheromone_matrix(paths)
            best_paths.extend(paths) 
        
        print("Finished Initialization of population")
        return best_paths
