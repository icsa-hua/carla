###### Hybrid Genetic Algorithm to find the best possible route based on time travel , ease of driving and route distance. #######
import numpy as np 
import networkx as nx
import time
import carla 
from agents.tools.misc import vector
from agents.navigation.local_planner import RoadOption
from agents.navigation.aco import ACO
import pygad


class HybridGA():

    def __init__(self, start, finish,map, graph,  population_size, generations, mutation_rate):
        self._map = map 
        self._start = start 
        self._finish = finish
        self._population_size = population_size
        self._tournament_size = 100
        self._parents_mating = 2
        self._generations = generations
        self._mutation_rate = mutation_rate 
        self._id_map = None
        self._road_id_to_edge = None
        self._sampling_resolution = 2.0
        self._replacement_rate = 10
        self._cost_matrix = {} 
        self._speed_lms = {}
        self._topology = None
        self._hga_graph = None
        self._parent_selection = "sss"
        self._keep_parents = 1 
        self._crossover_type = "single_point"
        self._mutation_type = "random"
        self._mutation_percent_genes = 10
        self._build_topology()
        self._create_graph()
        self._find_loose_ends()
        self._lane_change_link()
        self._hga_graph = nx.DiGraph(self._hga_graph)
        self._ACO = ACO(graph=self._hga_graph, source=self._start, target=self._finish)

        self._paths = self.find_paths(self._hga_graph,self._start[0],self._finish[0])
        self._num_genes = len(self._paths)
        self._routes = self.Pareto_GA(replacement_rate=0.2)

    def _build_topology(self): 
        """
        This function retrieves topology from the server as a list of
        road segments as pairs of waypoint objects, and processes the
        topology into a list of dictionary objects with the following attributes

        - entry (carla.Waypoint): waypoint of entry point of road segment
        - entryxyz (tuple): (x,y,z) of entry point of road segment
        - exit (carla.Waypoint): waypoint of exit point of road segment
        - exitxyz (tuple): (x,y,z) of exit point of road segment
        - path (list of carla.Waypoint):  list of waypoints between entry to exit, separated by the resolution
        """

        self._topology = []
        # Retrieving waypoints to construct a detailed topology
        for segment in self._map.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            # Rounding off to avoid floating point imprecision
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    seg_dict['path'].append(w)
                    next_ws = w.next(self._sampling_resolution)
                    if len(next_ws) == 0:
                        break
                    w = next_ws[0]
            else:
                next_wps = wp1.next(self._sampling_resolution)
                if len(next_wps) == 0:
                    continue
                seg_dict['path'].append(next_wps[0])
            self._topology.append(seg_dict) 
    
    def _create_graph(self): 
        #Νομίζω η καλύτερη λύση είναι να έχω για κόστος την απόσταση των δύο σημείων, 
        #το αντίστοιχο ease of driving, και να βρω τον χρόνο σαν συνάρτηση του χρόνου με τη μέγιστη ταχύτητα του εκάστοτε δρόμου 
        #Οπότε πρέπει: 
        #α) να βρίσκω την απόσταση μεταξύ των κόμβων της κάθε ακμής, 
        #β) να βρίσκω τον δρόμο στον οποίο ανήκουν τα wp. 
        #γ) να αναγνωρίζω και να αλλάζω την ταχύτητα αν έχουν διαφορετικές μέγιστες ταχύτητες. 
        #δ) να υπολογίζω μέσα από την ταχύτητα τον χρόνο για την απόσταση. 
        #ε) ... 
       
        speed_limits = self._map.get_all_landmarks_of_type('274')
        speed_lms = {lm.road_id: lm for lm in speed_limits}
        self._speed_lms = speed_lms
        self._hga_graph = nx.DiGraph()
        self._id_map = dict()  # Map with structure {(x,y,z): id, ... }
        self._road_id_to_edge = dict()  # Map with structure {road_id: {lane_id: edge, ... }, ... }


        for segment in self._topology: 
            enter_pos, exit_pos = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            enter_wp, exit_wp = segment['entry'], segment['exit']
            intersection = enter_wp.is_junction 
            road_id, section_id, lane_id = enter_wp.road_id, enter_wp.section_id, enter_wp.lane_id
            edge_distance = (np.sqrt((exit_pos[1]-enter_pos[1])**2 + (exit_pos[0]-enter_pos[0])**2 + (exit_pos[2]-enter_pos[2])**2))
            if road_id not in speed_lms: 
                speed_lms[road_id] = 30
            time_distance = edge_distance/speed_lms[road_id]
            weight = np.random.randint(1,10)
            self._cost_matrix[(enter_wp,exit_wp)] = [edge_distance, time_distance, weight]
            for vertex in enter_pos, exit_pos:
                # Adding unique nodes and populating id_map
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    self._hga_graph.add_node(new_id, vertex=vertex)
            n1 = self._id_map[enter_pos]
            n2 = self._id_map[exit_pos]
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_carla_vector = enter_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()
            self._hga_graph.add_edge(
                n1,n2,length=len(path)+1, path=path, entry_waypoint=enter_wp,exit_waypoint=exit_wp,
                entry_vector=np.array([entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
                exit_vector=np.array([exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),
                net_vector=vector(enter_wp.transform.location, exit_wp.transform.location),
                intersection=intersection, type=RoadOption.LANEFOLLOW, time=self._cost_matrix[(enter_wp, exit_wp)][1], ease_of_driving=self._cost_matrix[(enter_wp, exit_wp)][0],weight=self._cost_matrix[(enter_wp, exit_wp)][2])
    
    def _find_loose_ends(self):
        """
        This method finds road segments that have an unconnected end, and
        adds them to the internal graph representation
        """
        count_loose_ends = 0
        hop_resolution = self._sampling_resolution
        for segment in self._topology:
            exit_wp = segment['exit']
            exit_pos = segment['exitxyz']
            road_id, section_id, lane_id = exit_wp.road_id, exit_wp.section_id, exit_wp.lane_id
            if road_id in self._road_id_to_edge \
                    and section_id in self._road_id_to_edge[road_id] \
                    and lane_id in self._road_id_to_edge[road_id][section_id]:
                pass
            else:
                count_loose_ends += 1
                if road_id not in self._road_id_to_edge:
                    self._road_id_to_edge[road_id] = dict()
                if section_id not in self._road_id_to_edge[road_id]:
                    self._road_id_to_edge[road_id][section_id] = dict()
                n1 = self._id_map[exit_pos]
                n2 = -1*count_loose_ends
                self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
                next_wp = exit_wp.next(hop_resolution)
                path = []
                while next_wp is not None and next_wp \
                        and next_wp[0].road_id == road_id \
                        and next_wp[0].section_id == section_id \
                        and next_wp[0].lane_id == lane_id:
                    path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)
                if path:
                    n2_xyz = (path[-1].transform.location.x,
                              path[-1].transform.location.y,
                              path[-1].transform.location.z)
                    self._hga_graph.add_node(n2, vertex=n2_xyz)
                    if (exit_wp,path[-1]) not in self._cost_matrix:
                        edge_distance = (np.sqrt((n2_xyz[1]-exit_pos[1])**2 + (n2_xyz[0]-exit_pos[0])**2 + (n2_xyz[2]-exit_pos[2])**2))
                        time_distance = edge_distance/self._speed_lms[exit_wp.road_id]
                        weight = np.random.randint(1,10)
                        self._cost_matrix[(exit_wp,path[-1])] = [edge_distance, time_distance, weight]

                    self._hga_graph.add_edge(
                        n1, n2,
                        length=len(path) + 1, path=path,
                        entry_waypoint=exit_wp, exit_waypoint=path[-1],
                        entry_vector=None, exit_vector=None, net_vector=None,
                        intersection=exit_wp.is_junction, type=RoadOption.LANEFOLLOW,
                        time=self._cost_matrix[(exit_wp,path[-1])][1], ease_of_driving=self._cost_matrix[(exit_wp,path[-1])][0],weight=self._cost_matrix[(exit_wp,path[-1])][2])
    
    def _lane_change_link(self): 
        """
        This method places zero cost links in the topology graph
        representing availability of lane changes.
        """

        for segment in self._topology:
            left_found, right_found = False, False

            for waypoint in segment['path']:
                if not segment['entry'].is_junction:
                    next_waypoint, next_road_option, next_segment = None, None, None

                    if waypoint.right_lane_marking and waypoint.right_lane_marking.lane_change & carla.LaneChange.Right and not right_found:
                        next_waypoint = waypoint.get_right_lane()
                        if next_waypoint is not None \
                                and next_waypoint.lane_type == carla.LaneType.Driving \
                                and waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANERIGHT
                            next_segment = self._localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                if (waypoint, next_waypoint) not in self._cost_matrix:
                                    edge_distance = (np.sqrt((next_waypoint.transform.location.x-waypoint.transform.location.x)**2 + (next_waypoint.transform.location.y-waypoint.transform.location.y)**2 + (next_waypoint.transform.location.z-waypoint.transform.location.z)**2))
                                    time_distance = edge_distance/self._speed_lms[waypoint.road_id]
                                    weight = np.random.randint(1,10)
                                    self._cost_matrix[(waypoint, next_waypoint)] = [edge_distance, time_distance,weight]
                                    # print("Position inside the right junction")

                                self._hga_graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint,
                                    time=self._cost_matrix[(waypoint, next_waypoint)][1], ease_of_driving=self._cost_matrix[(waypoint, next_waypoint)][0],weight=self._cost_matrix[(waypoint, next_waypoint)][2])
                                right_found = True
                    
                    if waypoint.left_lane_marking and waypoint.left_lane_marking.lane_change & carla.LaneChange.Left and not left_found:
                        next_waypoint = waypoint.get_left_lane()
                        if next_waypoint is not None \
                                and next_waypoint.lane_type == carla.LaneType.Driving \
                                and waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANELEFT
                            next_segment = self._localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                if (waypoint, next_waypoint) not in self._cost_matrix:
                                    edge_distance = (np.sqrt((next_waypoint.transform.location.x-waypoint.transform.location.x)**2 + (next_waypoint.transform.location.y-waypoint.transform.location.y)**2 + (next_waypoint.transform.location.z-waypoint.transform.location.z)**2))
                                    time_distance = edge_distance/self._speed_lms[waypoint.road_id]
                                    weight = np.random.randint(1,10)
                                    self._cost_matrix[(waypoint, next_waypoint)] = [edge_distance, time_distance,weight]
                                    # print("Position inside the left junction")
                                self._hga_graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint,
                                    time=self._cost_matrix[(waypoint, next_waypoint)][1], ease_of_driving=self._cost_matrix[(waypoint, next_waypoint)][0],weight=self._cost_matrix[(waypoint, next_waypoint)][2])
                                left_found = True
                if left_found and right_found:
                    break

    def _localize(self,location): 
        """
        This function finds the road segment that a given location
        is part of, returning the edge it belongs to
        """
        waypoint = self._map.get_waypoint(location)
        edge = None
        try:
            edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            pass
        return edge
    
    def find_paths(self, G, start, end, path=[], max_depth=15):
        path = path + [start]
        if start == end:
            return [path]
        if len(path) > max_depth:
            return []
        paths = []
        for node in G.neighbors(start):
            if node not in path:
                newpaths = self.find_paths(G, node, end, path, max_depth)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def fitness_func(self, path):
        fitness = []

        #Provide max values why? because we want to use them as the scale component for each element. 
        max_time = 50 #Assume maximum time of 50 seconds 
        max_distance = 200 #Assume maximum distance between nodes. 
        max_weight = 10

        #Weights for each component 
        weight_time = 0.4 
        weight_ease = 0.4 
        weight_weight = 0.2 

        #Initialize metrics 
        travel_time = 0 
        ease_of_driving = 0
        weights = 0

        for i in range(len(path)-1):
            source_node = path[i]
            target_node = path[i+1]

            if self._hga_graph.has_edge(source_node,target_node):

                edge_attributes = self._hga_graph.get_edge_data(source_node,target_node)
                travel_time += edge_attributes['time']
                ease_of_driving += edge_attributes['ease_of_driving']
                weights += edge_attributes['weight']

            else: 
                #Handle the case for where the edge does not exist for example assign a penalty 
                travel_time += 200
                ease_of_driving += 500
                weights += 50

        #Normalize values based on factors 
        normalized_time = travel_time/max_time
        normalized_ease = ease_of_driving/max_distance
        normalized_weights = weights/max_weight

        fit = -(weight_time*normalized_time) + (weight_ease*normalized_ease) + (weight_weight*normalized_weights)
        fitness.append(fit)
            
        return fitness
    
    def initialize_population(self, num_individuals,nodes, min_length, max_length):
        #TODO: Change the paramteres needed. 
        population = [] 
        population = self._ACO._ant_colony_optimization()
        print(f"Initialized population with ACO and length {len(population)}")
        return population  

    def selection(self, fitness_values):
        #Selection step
        selected_indices = [] 
        self._tournament_size = self._population_size//2
        for _ in range(int(self._population_size/2)):
            tournament_indices = np.random.choice(range(len(fitness_values)), size=self._tournament_size, replace=False)
            tournament_fitness = [fitness_values[i] for i in tournament_indices]

            best_individual_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected_indices.append(best_individual_idx)

        return selected_indices

    def crossover(self,p1, p2):
        #Combines parts of two selected paths to create offsprings

        # Find common nodes between p1 and p2 that are also connected in the graph        
        common_nodes = [node for node in p1 if node in p2 and self._hga_graph.has_node(node)]
        connected_common_nodes = [node for i, node in enumerate(common_nodes[:-1]) if self._hga_graph.has_edge(node, common_nodes[i + 1])]

        if not connected_common_nodes:
        # If no connected common nodes are found, return one of the parents as the child
            return p1 if np.random.rand() > 0.5 else p2
        
        crossover_point = np.random.choice(connected_common_nodes)
        
        # Find the indices of the crossover point in both parents
        cp_index_p1 = p1.index(crossover_point)
        cp_index_p2 = p2.index(crossover_point)

        # Create offspring by combining the segment before the crossover point in p1 with the segment after the crossover point in p2
        # Ensure the crossover point is included only once in the child
        child = [p1[0]] + p1[:cp_index_p1 + 1] + p2[cp_index_p2 + 1:] + [p1[-1]]

        valid_child = [child[0]]
        max_iterations = 100 
        while valid_child[-1] != child[-1] and max_iterations>0:
            for i in range(1, len(child)-1):
                if self._hga_graph.has_edge(valid_child[-1], child[i]):
                    valid_child.append(child[i])
                if not self._hga_graph.has_edge(valid_child[-1], child[i+1]):
                    max_iterations = 0
            max_iterations -= 1 
        return valid_child
    
    def mutate(self, individual):
        #Randomly alters a path to introduce variability and explore unseen areas of the search space. 
        if np.random.rand() < self._mutation_rate:
            if len(individual) <= 2: 
                return individual
            swap_index1 = np.random.randint(1, len(individual) - 1)
            swap_index2 = np.random.randint(1, len(individual) - 1)
        
            individual[swap_index1], individual[swap_index2] = individual[swap_index2], individual[swap_index1]
            individual = [node for node in individual if node in self._hga_graph.nodes]
        return individual
    
    def is_non_dominated(self,candidate, pareto_front):
        #For Pareto
        candidate_attr = self.get_attributes(candidate)
        for member in pareto_front: 
            member_data = self.get_attributes(member)
            if self.dominates(member_data, candidate_attr):
                return False
        return True

    def dominates(self,member , candidate):
        #For pareto
        better_in_one = False 
        for objective in ['time', 'ease_of_driving', 'weights']: 
            if member[objective] > candidate[objective]:
                return False 
            elif member[objective] < candidate[objective]:
                better_in_one = True
        return better_in_one
    
    def update_pareto_front(self,pareto, offspring):
        # print("Updating Pareto front...")
        #Combine current pareto front and offspring
        new_pareto_front = pareto[:]
        for child in offspring: 
            if self.is_non_dominated(child,new_pareto_front):
                new_pareto_front = [solution for solution in new_pareto_front if not self.dominates(child, solution)]
                new_pareto_front.append(child)
        return new_pareto_front

    def get_attributes(self,path):
        travel_time = 0
        ease_of_driving = 0 
        weights = 0 
        for i in range(len(path)-1):
            edge_data = self._hga_graph.get_edge_data(path[i], path[i+1])
            if edge_data: 
                travel_time += edge_data['time']
                ease_of_driving += edge_data['ease_of_driving']
                weights += edge_data['weights']
            else: 
                travel_time += 200
                ease_of_driving += 500
                weights += 50

        return {'time':travel_time, 'ease_of_driving':ease_of_driving, 'weights':weights}

    def is_valid_path(self,path):
        for i in range(len(path)-1):
            sn = path[i]
            tn = path[i+1]
            if not self._hga_graph.has_edge(sn,tn):
                return False
        return True
    
    def find_best_path(self, final_population):
        # Assuming fitness_values is a list of fitness scores corresponding to each path in population
        # and that a higher fitness score is better.
        index_of_best = 0
        best_paths = {}

        for generation in final_population:
            # keys = list(final_population[generation])
            values = list(final_population[generation])
            index_of_best = values.index(min(values))
            best_paths[generation] = values[index_of_best]

        best_fitness = list(best_paths.values())
        index_of_best = best_fitness.index(min(best_fitness))
        best_path = list(best_paths.keys())[index_of_best]

        return best_path

    def check_duplicate_paths(self, offsprings):
        unique_paths_set = set(tuple(path) for path in offsprings)
        unique_paths = [list(path) for path in unique_paths_set]  
        return unique_paths  

    def Pareto_GA(self, replacement_rate):
    
        for node in ((self._hga_graph.nodes)):
            self._hga_graph.nodes[node]['order'] = 0
            if node == self._start:
                self._hga_graph.nodes[node]['order'] = -1000

        maximul_path_len = len(self._hga_graph.nodes)
        minimum_path_len = 2
        nodes = list(self._hga_graph.nodes()) 

        #Initialization
        population = self.initialize_population(self._population_size, nodes,minimum_path_len,maximul_path_len)
    
        # TODO: Create the initial Pareto front with the first solution
        # pareto_front = [population[0]]

        final_population = {}
        for generation in range(self._generations):
            
            #Output progress bar 
            percent = generation / self._generations * 100
            filled_length = int(percent / 2)
            bar = '=' * filled_length + '-' * (50 - filled_length)
            print(f'\r|{bar}| {percent:.1f}%', end='\r')
            time.sleep(0.1)
            
            #Upper limit to avoid infinite loop 
            max_attempts = 10 * int(replacement_rate* self._population_size)
            attempts = 0
            
            #Evaluation Step - fitness for each individual
            fitness_values = [self.fitness_func(individual + [self._finish]) for individual in population]

            #Selection Step
            selected_indices = self.selection(fitness_values=fitness_values)            
            parents = [population[i] for i in selected_indices]
  
            # Create offspring using crossover and mutation
            offspring = []
            while len(offspring) < int(replacement_rate * self._population_size)and attempts < max_attempts:

                #Crossover and mutation
                index1, index2 = np.random.choice(len(parents), 2, replace=False)
                parent1, parent2 = parents[index1], parents[index2]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)

                #Validity check for temporary path 
                if self.is_valid_path(child):
                    offspring.append(child)
                attempts += 1

            #Duplication deletion 
            offspring = self.check_duplicate_paths(offspring)
            # final_population.append(offspring)
            
            for path in offspring:
                # print(f"Path {path}")
                final_population[tuple(path)] = self.fitness_func(path)
            final_population = dict(sorted(final_population.items(), key=lambda item: item[1]))
            # TODO: Maybe fix the pareto evaluation
            # pareto_front = self.update_pareto_front(pareto_front, offspring)
        
        #Final processing for best path 
        # best_route = self.find_best_path(final_population)
        # if not self.is_valid_path(best_route):
        #     print("Path generated by the HGA is not valid!\nReturning empty route.")
        #     return []
        # else:
        #     print("Path generated by the HGA is valid!\nReturning the best route.")
        #     print(best_route)
 
        #Return the best routes 
        best_routes = [] 
        for path in final_population:
            path = list(path)
            duplicates = set(path)
            if path[-1] == self._finish[0]:
                if len(path) == len(duplicates):
                    if not self.is_valid_path(path):
                        # print("Path generated by the HGA is not valid!\nReturning empty route.")
                        continue
                    else:
                        # print("Path generated by the HGA is valid!\nReturning the best route.")
                        best_routes.append(path)
        print(best_routes)
        # TODO: Select the best route from the final Pareto front
        # best_route = max(pareto_front, key=lambda x: self.calculate_fitness(x+ [self._finish] ))
       
        return best_routes

    