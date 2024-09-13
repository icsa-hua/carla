from agents.navigation.dummy_application.dummy_utils.interface.HGA import HGA
from agents.navigation.dummy_application.dummy_utils.PPAs.ant_colony import  AntColony
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import vector
import carla 
import numpy as np 
import networkx as nx 
import pdb 
import time

class HybridGeneticAlgorithm(HGA):

    def __init__(self, origins, map, specs):
        super().__init__(origins, map, specs) 

        self.source = self.origins['start']
        self.destination = self.origins['end']

        self.cost_matrix = {}
        self.speed_lms = {} 
        self.topology = []
        self.graph = None
        self.id_map = {}
        
        self.build_topology()
        self.create_graph()
        self.find_loose_ends()
        self.lane_change_link()


    def build_topology(self):

        for segment in self.map.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self.specs['sampling_resolution']:
                w = wp1.next(self.specs['sampling_resolution'])[0]
                while w.transform.location.distance(endloc) > self.specs['sampling_resolution']:
                    seg_dict['path'].append(w)
                    next_ws = w.next(self.specs['sampling_resolution'])
                    if len(next_ws) == 0:
                        break
                    w = next_ws[0]
            else:
                next_wps = wp1.next(self.specs['sampling_resolution'])
                if len(next_wps) == 0:
                    continue
                seg_dict['path'].append(next_wps[0])
            self.topology.append(seg_dict) 


    def create_graph(self):
        speed_limits = self.map.get_all_landmarks_of_type('274')
        self.speed = {lm.road_id: lm for lm in speed_limits}

        self.graph = nx.DiGraph()
        self.road_id_to_edge = {}

        for segment in self.topology:
            enter_pos, exit_pos = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            enter_wp, exit_wp = segment['entry'], segment['exit']
            intersection = enter_wp.is_junction
            road_id, section_id, lane_id = enter_wp.road_id, enter_wp.section_id, enter_wp.lane_id
            edge_distance = (np.sqrt((exit_pos[1]-enter_pos[1])**2 + (exit_pos[0]-enter_pos[0])**2 + (exit_pos[2]-enter_pos[2])**2))
            
            if road_id not in self.speed_lms: 
                self.speed_lms[road_id] = 30

            if type(self.speed_lms[road_id]) == carla.Landmark:
                self.speed_lms[road_id] = self.speed_lms[road_id].value

            weight = np.random.randint(1,5)
            time_for_distance = edge_distance / self.speed_lms[road_id]
            self.cost_matrix[(enter_wp,exit_wp)] = [edge_distance, time_for_distance, weight]

            for vertex in enter_pos, exit_pos:
                if vertex not in self.id_map:
                    new_id = len(self.id_map)
                    self.id_map[vertex] = new_id
                    self.graph.add_node(new_id, vertex=vertex)

            n1 = self.id_map[enter_pos]
            n2 = self.id_map[exit_pos]

            if road_id not in self.road_id_to_edge:
                self.road_id_to_edge[road_id] = dict()

            if section_id not in self.road_id_to_edge[road_id]:
                self.road_id_to_edge[road_id][section_id] = dict()

            self.road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_carla_vector = enter_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()
            
            self.graph.add_edge(
                n1,n2,
                length=len(path)+1,
                path=path,
                entry_waypoint=enter_wp,
                exit_waypoint=exit_wp,
                entry_vector=np.array([entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
                exit_vector=np.array([exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),
                net_vector=vector(enter_wp.transform.location, exit_wp.transform.location),
                intersection=intersection,
                type=RoadOption.LANEFOLLOW,
                time=self.cost_matrix[(enter_wp, exit_wp)][1],
                ease_of_driving=self.cost_matrix[(enter_wp, exit_wp)][0],
                weight=self.cost_matrix[(enter_wp, exit_wp)][2]
            )


    def find_loose_ends(self):
       
        count_loose_ends = 0

        for segment in self.topology: 
            exit_wp = segment['exit']
            exit_pos = segment['exitxyz']
            road_id, section_id, lane_id = exit_wp.road_id, exit_wp.section_id, exit_wp.lane_id

            if road_id in self.road_id_to_edge \
                    and section_id in self.road_id_to_edge[road_id] \
                    and lane_id in self.road_id_to_edge[road_id][section_id]:
                
                continue
            
            else: 
                count_loose_ends += 1 
                
                if road_id not in self.road_id_to_edge: 
                    self.road_id_to_edge[road_id] = dict()

                if section_id not in self._road_id_to_edge[road_id]:
                    self._road_id_to_edge[road_id][section_id] = dict()

                n1 = self.id_map[exit_pos]
                n2 = -1*count_loose_ends

                self.road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
                next_wp = exit_wp.next(self.specs['sampling_resolution'])
                
                path = [] 

                while next_wp is not None and next_wp \
                        and next_wp[0].road_id == road_id \
                        and next_wp[0].section_id == section_id \
                        and next_wp[0].lane_id == lane_id:
                    path.append(next_wp[0])

                    next_wp = next_wp[0].next(self.specs['sampling_resolution'])

                if path: 
                    n2_xyz = (path[-1].transform.location.x,
                              path[-1].transform.location.y,
                              path[-1].transform.location.z)
                    
                    self.graph.add_node(n2, vertex=n2_xyz)

                    if (exit_wp,path[-1]) not in self.cost_matrix:
                        edge_distance = (np.sqrt((n2_xyz[1]-exit_pos[1])**2 + (n2_xyz[0]-exit_pos[0])**2 + (n2_xyz[2]-exit_pos[2])**2))
                        weight = np.random.randint(1,5)
                        time_for_distance = edge_distance/self.speed_lms[exit_wp.road_id]
                        self.cost_matrix[(exit_wp,path[-1])] = [edge_distance, time_for_distance, weight]


                    self.graph.add_edge(
                        n1, n2, 
                        length=len(path) + 1,
                        path=path,
                        entry_waypoint=exit_wp,
                        exit_waypoint=path[-1],
                        entry_vector=None,
                        exit_vector=None,
                        net_vector=None,
                        intersection=exit_wp.is_junction,
                        type=RoadOption.LANEFOLLOW,
                        time=self.cost_matrix[(exit_wp,path[-1])][1],
                        ease_of_driving=self.cost_matrix[(exit_wp,path[-1])][0],
                        weight=self.cost_matrix[(exit_wp,path[-1])][2]
                    )


    def try_lane_change(self, waypoint, lane_direction, segment): 
        lane_found = False 
        next_waypoint = getattr(waypoint, f'get_{lane_direction}_lane')()
        
        if not next_waypoint or next_waypoint.lane_type != carla.LaneType.Driving \
             or waypoint.road_id != next_waypoint.road_id:
            return lane_found

        next_segment = self._localize(next_waypoint.transform.location)
        if not next_segment:
            return lane_found
        
        if (waypoint, next_waypoint) not in self.cost_matrix: 
            edge_distance = (np.sqrt((next_waypoint.transform.location.x-waypoint.transform.location.x)**2 + (next_waypoint.transform.location.y-waypoint.transform.location.y)**2 + (next_waypoint.transform.location.z-waypoint.transform.location.z)**2))
            weight = np.random.randint(1,5)
            time_for_distance = edge_distance/self.speed_lms[waypoint.road_id]
            self.cost_matrix[(waypoint, next_waypoint)] = [edge_distance, time_for_distance, weight]

        next_road_option = RoadOption.CHANGELANERIGHT if lane_direction == 'right' else RoadOption.CHANGELANELEFT
        self.graph.add_edge(
            self.id_map[segment['entryxyz']],
            next_segment[0], 
            entry_waypoint=waypoint,
            exit_waypoint=next_waypoint,
            intersection=False,
            exit_vector=None,
            path=[],
            length=0, 
            type=next_road_option, 
            change_waypoint=next_waypoint,
            time=self.cost_matrix[(waypoint, next_waypoint)][1],
            ease_of_driving=self.cost_matrix[(waypoint, next_waypoint)][0],
            weight=self.cost_matrix[(waypoint, next_waypoint)][2],
        )

        return True
    

    def lane_change_link(self): 

        for segment in self.topology: 
            left_found, right_found = False, False

            for waypoint in segment['path']: 
               
                if segment['entry'].is_junction:
                    continue

                if not right_found and waypoint.right_lane_marking and \
                    (waypoint.right_lane_marking.lane_change & carla.LaneChange.Right):
                    
                    right_found = self.try_lane_change(waypoint, 'right', segment)

                if not left_found and waypoint.left_lane_marking and \
                    (waypoint.left_lane_marking.lane_change & carla.LaneChange.Left):

                    left_found = self.try_lane_change(waypoint, 'left', segment)

                if left_found and right_found:
                    break

    
    def _localize(self, location):
        waypoint = self.map.get_waypoint(location)
        edge = None
        try:
            edge = self.road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            pass
        return edge
    
    
    def find_valid_paths(self, graph: nx.Graph, origins: dict, path: list, max_depth: int) -> list:
        return super().find_valid_paths(graph, origins, path, max_depth)
    
    
    def fitness_calculation(self, path: list) -> list:
        fitness = [] 

        max_time = 100 #max seconds between two nodes
        max_distance = 500 #max distance between two nodes
        max_weight = 5 #max weight of a node

        weight_for_time = 0.25 
        weight_for_distance = 0.7
        weight_for_weight = 0.05

        travel_time = 0 
        ease_of_driving = 0
        weight = 0

        for i in range(len(path)-1):
            source_node = path[i]
            target_node = path[i+1]

            if self.graph.has_edge(source_node,target_node):
                edge_attributes = self.graph.get_edge_data(source_node,target_node)
                travel_time += edge_attributes['time']
                ease_of_driving += edge_attributes['ease_of_driving']
                weight += edge_attributes['weight']

            else:
                travel_time+=100
                ease_of_driving+=1000
                weight+=10

        normalized_time = travel_time/max_time
        normalized_ease = ease_of_driving/max_distance
        normalized_weights = weight/max_weight
        fit = (normalized_time*weight_for_time) + (normalized_ease*weight_for_distance) + (normalized_weights*weight_for_weight)
        fitness.append(fit)

        return fitness


    def initialize_population(self) -> list:
        
        specs = {
            "number_of_ants":10, 
            "iterations": 20, 
            "initial_pheromone":0.1, 
            "alpha": 1.0, 
            "beta": 1.0, 
            "gamma":1.0, 
            "evaporation_rate": 0.5, 
            "pheromone_deposit": 100.0, 
            "population_limit": 60, 
            "max_recursions":5
        }

        aco = AntColony(self.origins, self.graph, specs)
        population = aco.run()

        if not population: 
            return []
        
        self.specs['population_size'] = len(population)

        for i in range(self.specs['population_size'])[::-1]:
            path = population[i]
            if path[0] != self.source[0] or path[-1] != self.destination[0]:
                population.pop(i)
                self.specs['population_size'] -=1 
            
            if path[0] not in self.graph.nodes or path[-1] not in self.graph.nodes:
                msg = f"Either source {path[0]} or target {path[-1]} is not in G"
                raise nx.NodeNotFound(msg)
        return population
    
    def selection(self, fitness_values: list) -> list:
                
        selected_indices = [] 
        self.specs["tournament_size"] = int(self.specs["population_size"]//2)
        for _ in range(self.specs["tournament_size"]):
            tournament_indices = np.random.choice(range(len(fitness_values)), size=self.specs["tournament_size"], replace=False)
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            best_individual_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected_indices.append(best_individual_idx)

        return selected_indices
    
    
    def crossover(self, ind1, ind2):
        
        if len(ind1) < 2 or len(ind2) < 2:
            return ind1 if len(ind1) > len(ind2) else ind2 
        
        common_nodes = [node for node in ind1 if node in ind2 and self.graph.has_node(node)]
        connected_common_nodes = [node for i, node in enumerate(common_nodes[:-1]) if self.graph.has_edge(node, common_nodes[i + 1]) and self.graph.has_edge(common_nodes[i + 1], node)]

        if not connected_common_nodes:
            return ind1 if np.random.rand() > 0.65 else ind2 #Allagh ap;o 0.5 se 0.65
        
        crossover_point = np.random.choice(connected_common_nodes)

        cp_index_p1 = ind1.index(crossover_point)
        cp_index_p2 = ind2.index(crossover_point)

        child = [ind1[0]] + ind1[:cp_index_p1 + 1] + ind2[cp_index_p2 + 1:] + [ind1[-1]]

        seen = set()
        valid_child = []
        for node in child:
            if node not in seen and (not valid_child or self.graph.has_edge(valid_child[-1], node)):
                valid_child.append(node)
                seen.add(node)
    
        return valid_child


    def mutation(self, ind):
                
        if np.random.rand() < self.specs['mutation_rate'] and len(ind) > 2: 
            swap_index1 = np.random.randint(1, len(ind) - 1)
            swap_index2 = np.random.randint(1, len(ind) - 1)

            if (self.graph.has_edge(ind[swap_index1-1],ind[swap_index2]) and 
                self.graph.has_edge(ind[swap_index2], ind[swap_index1 + 1]) and
                self.graph.has_edge(ind[swap_index2-1], ind[swap_index1]) and 
                self.graph.has_edge(ind[swap_index1], ind[swap_index2 + 1])):

                ind[swap_index1], ind[swap_index2] = ind[swap_index2], ind[swap_index1]
        
        return ind 
    
    
    def run(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]['order'] = 0
            if node == self.source:
                self.graph.nodes[node]['order'] = -1000
        population = self.initialize_population()
        final_population = self.evolve_population(population) 
        self.best_routes = [path for path in final_population if self.check_duplicate_nodes(path)]
        if not self.best_routes:
            import warnings 
            warnings.warn("No valid routes found.")
            return self.best_routes 
        
        return self.best_routes
    

    def is_valid_path(self,path):
        for i in range(len(path)-1):
            sn = path[i]
            tn = path[i+1]
            if not self.graph.has_edge(sn,tn):
                return False
        return True
    
    
    def check_duplicate_paths(self, offsprings):
        unique_paths_set = set(tuple(path) for path in offsprings)
        unique_paths = [list(path) for path in unique_paths_set]  
        return unique_paths  
    

    def check_duplicate_nodes(self, path): 
        
        path = list(path)
        duplicates = set(path)

        if path[-1] != self.destination[0]:    
            return False 
        
        if len(path) <= 2: 
            return False 

        if len(path) != len(duplicates):
            return False
        
        if not self.is_valid_path(path): 
            return False 
        
        return True
    

    def display_progress(self, generation): 
        percent = generation / self.specs['generations'] * 100
        filled_length = int(percent / 2)
        bar = '=' * filled_length + '-' * (50 - filled_length)
        print(f'\r|{bar}| {percent:.1f}%', end='\r')
        time.sleep(0.1)


    def generate_offspring(self, population): 
        offspring = []
        max_attempts = 10 * int(self.specs['replacement_rate'] * self.specs['population_size'])
        count_attempts = 0
        fitness_values = [self.fitness_calculation(individual + [self.destination]) for individual in population]

        parents = [population[i] for i in self.selection(fitness_values)]
        while len(offspring) < int(self.specs['replacement_rate'] * self.specs['population_size']) and count_attempts < max_attempts:
            index_1 , index_2 = np.random.choice(len(parents), 2, replace=False)
            parent1, parent2 = parents[index_1], parents[index_2]

            child = self.mutation(self.crossover(parent1, parent2))

            if self.is_valid_path(child):
                offspring.append(child)
            count_attempts += 1
        return self.check_duplicate_paths(offspring)


    def evolve_population(self, population): 
        final_population = {}
    
        for generation in range(self.specs['generations']):
            self.display_progress(generation)
            offspring = self.generate_offspring(population)
            final_population.update({tuple(path): self.fitness_calculation(path) for path in offspring})
        
        return dict(sorted(final_population.items(), key=lambda item: item[1]))


    def get_best_routes(self): 
        return self.best_routes