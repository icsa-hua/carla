###### Hybrid Genetic Algorithm to find the best possible route based on time travel , ease of driving and route distance. #######
import numpy as np 
import networkx as nx
import time
import carla 
from agents.tools.misc import vector
from agents.navigation.local_planner import RoadOption


class HybridGA():

    def __init__(self, start, finish,map,graph,  population_size, generations, mutation_rate):
        self._map = map 
        self._start = start 
        self._finish= finish
        self._population = population_size
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

        self._build_topology()
        self._create_graph()
        self._find_loose_ends()
        self._lane_change_link()

        self._routes = self.Pareto_GA(replacement_rate=0.2)
        print(self._routes)
        time.sleep(1000)

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
        self._hga_graph = nx.Graph()
        self._id_map = dict()  # Map with structure {(x,y,z): id, ... }
        self._road_id_to_edge = dict()  # Map with structure {road_id: {lane_id: edge, ... }, ... }
        # print(self._topology)
        # print("Topology")
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
            weight = np.random.randint(1,100)
            self._cost_matrix[(enter_wp,exit_wp)] = [edge_distance, time_distance, weight]
            print(f"Waypoints {(enter_wp,exit_wp)} with time {time_distance} and distance {edge_distance} and weight {weight}")
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
                        weight = np.random.randint(1,100)
                        self._cost_matrix[(exit_wp,path[-1])] = [edge_distance, time_distance, weight]
                        print("Position inside the find looseee")

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
                                    weight = np.random.randint(1,100)
                                    self._cost_matrix[(waypoint, next_waypoint)] = [edge_distance, time_distance,weight]
                                    print("Position inside the right junction")

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
                                    weight = np.random.randint(1,100)
                                    self._cost_matrix[(waypoint, next_waypoint)] = [edge_distance, time_distance,weight]
                                    print("Position inside the left junction")
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
    
    def calculate_fitness(self,path):
        travel_time = 0 
        ease_of_driving = 0
        weights = 0
        scaling_factor = 0.75
        for i in range(len(path)-1):
            source_node = path[i]
            target_node = path[i+1]

            if self._hga_graph.has_edge(source_node,target_node):
                # edge_attributes = self._graph[source_node][target_node]
                edge_attributes = self._hga_graph.get_edge_data(source_node,target_node)
                travel_time += edge_attributes['time']
                ease_of_driving += edge_attributes['ease_of_driving']
                weights += edge_attributes['weight']
                # print("Is a link ", travel_time, ease_of_driving, weights) 
            else: 
                #Handle the case for where the edge does not exist for example assign a penalty 
                travel_time +=1000
                ease_of_driving += 0
                weights += 50
                # print("Is  not a link ", travel_time, ease_of_driving, weights)  

        return -(travel_time + (ease_of_driving*scaling_factor) + (weights*(1-scaling_factor)))    
    
    def initialize_population(self,nodes):
        # print("Initializing population completing...")
        #Randomly generated individuals 
        return [list(np.random.permutation(nodes)) for _ in range(self._population)]
    

    def crossover(self,p1, p2):
        #Combines parts of two selected paths to create offsprings
        # print("Crossover processing...")
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
        child = p1[:cp_index_p1 + 1] + p2[cp_index_p2 + 1:]

        valid_child = [child[0]]
        for i in range(1, len(child)):
            if self._hga_graph.has_edge(valid_child[-1], child[i]):
                valid_child.append(child[i])
        
        return valid_child
    
    def mutate(self, individual):
        # print("Mutation processing...")
        #Randomly alters a path to introduce variability and explore unseen areas of the search space. 
        if np.random.rand() < self._mutation_rate:
            print(f"Lentgth of individual {len(individual)}")
            if len(individual) <= 1: 
                return individual
            mutation_point1, mutation_point2 = np.random.choice(len(individual), 2, replace=False)
            individual[mutation_point1], individual[mutation_point2] = individual[mutation_point2] , individual[mutation_point1]
        
        individual = [node for node in individual if node in self._hga_graph.nodes]
        return individual
    
    def update_pareto_front(self,pareto, offspring):
        # print("Updating Pareto front...")
        #Combine current pareto front and offspring
        combined_front = pareto + offspring

        #Identify non-dominated solutions using Pareto dominance 
        non_dominated_front = [] 
        for sol in combined_front:
            if all(self.calculate_fitness(sol) >= self.calculate_fitness(sol2) for sol2 in combined_front):
                non_dominated_front.append(sol)
        return non_dominated_front


    def is_valid_path(self,path):
        for i in range(len(path)-1):
            sn = path[i]
            tn = path[i+1]
            if not self._hga_graph.has_edge(sn,tn):
                return False
        return True
    
    def Pareto_GA(self, replacement_rate):
    
        for node in ((self._hga_graph.nodes)):
            self._hga_graph.nodes[node]['order'] = 0
            if node == self._start:
                self._hga_graph.nodes[node]['order'] = -1000

        #Nodes are fine
        nodes = list(self._hga_graph.nodes()) 
        
        #Population is fine
        population = self.initialize_population(nodes)

        # Create the initial Pareto front with the first solution
        pareto_front = [population[0]]

        for generation in range(self._generations):
            percent = generation / self._generations * 100
            filled_length = int(percent / 2)
            bar = '=' * filled_length + '-' * (50 - filled_length)
            print(f'\r|{bar}| {percent:.1f}%', end='\r')
            time.sleep(0.1)
            max_attempts = 10 * int(replacement_rate* self._population)
            attempts = 0
            #Evaluate fitness for each individual
             
            fitness_value = [self.calculate_fitness(individual + [self._finish]) for individual in population]
            # print(f"Fitness value: {fitness_value}")

            selected_indices = np.argsort(fitness_value)[-int(self._population/2):]
            # print("Selected indices: ", selected_indices)
            parents = [population[i] for i in selected_indices]
            # print("parents: ", parents)
  
            # Create offspring using crossover and mutation
            offspring = []
            while len(offspring) < int(replacement_rate * self._population)and attempts < max_attempts:
                index1, index2 = np.random.choice(len(parents), 2, replace=False)
                parent1, parent2 = parents[index1], parents[index2]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                # print("Child ::: ", child)
                if self.is_valid_path(child):
                    print("Child ::: ", child)
                    offspring.append(child)
                attempts += 1

        #103,2,96,88,89,38,39,108,109,111,116
        # Update Pareto front
        pareto_front = self.update_pareto_front(pareto_front, offspring)
        
        best_route = max(pareto_front,key=lambda x:self.calculate_fitness(x + [self._finish]))

        print(best_route)
        time.sleep(100)

        if best_route[-1] == self._finish:
            return best_route
        
        replacement_indices = selected_indices[:int(replacement_rate * self._population)]
        for i, index in enumerate(replacement_indices):
            population[index] = offspring[i]
        
        # Select the best route from the final Pareto front
        best_route = max(pareto_front, key=lambda x: self.calculate_fitness(x+ [self._finish] ))
        return best_route

    def GA(self):
        nodes = list(self._hga_graph.nodes())
        population = self.initialize_population(nodes)

        for generation in range(self._generations):

            fitness_value = [self.calculate_fitness(individual + [self._finish])for individual in population]
            selected_indices = np.argsort(fitness_value)[-int(self._population/2):]
            parents = [population[i] for i in selected_indices]
            
            offspring = []
            while len(offspring) < self._population - len(parents):
                ind1,ind2 = np.random.choice(len(parents),2,replace=False)
                p1,p2 = parents[ind1],parents[ind2]
                child = self.crossover(p1,p2)
                child = self.mutate(child)
                offspring.append(child)

            population = parents+offspring
        
        best_route = max(population, key=lambda x: self.calculate_fitness(x+[self._finish]))
        return best_route + [self._finish]
    
# G = nx.DiGraph()
# G.add_edges_from([(1, 2, {'time': 10, 'ease_of_driving': 8}),
#                   (2, 3, {'time': 15, 'ease_of_driving': 7}),
#                   (3, 4, {'time': 20, 'ease_of_driving': 6}),
#                   (4, 1, {'time': 5, 'ease_of_driving': 9}),
#                   (2, 4, {'time': 25, 'ease_of_driving': 5})])


# start = 3
# finish = 1 

# for node in ((G.nodes)):
#     G.nodes[node]['order'] = 0
#     if node == start:
#         G.nodes[node]['order'] = -1000
   
# ga = HybridGA(start, finish, G, population_size=50, generations=100, mutation_rate=0.2)
# best_route = ga.steady_state_GA(replacement_rate=0.2)
# sorted_best_route = sorted(best_route, key=lambda x:G.nodes[x]['order'])
# if sorted_best_route[0]!= start and sorted_best_route[-1]!= finish:
#     print("Error: Start or finish node is not in the best route")
# elif sorted_best_route[0] == start :
#     sorted_best_route = [(sorted_best_route[i],sorted_best_route[i+1]) for i in range(len(sorted_best_route)-1) if sorted_best_route[i] != finish]
#     print(sorted_best_route)
#     print([(sorted_best_route[i],sorted_best_route[i+1]) for i in range(len(sorted_best_route)-1)])
#     total_time = 0 
#     ease_of = 0
#     for u in sorted_best_route:
#         total_time += G[u[0]][u[1]]['time']
#         ease_of += G[u[0]][u[1]]['ease_of_driving']
#     print("Best Route:", sorted_best_route)
#     print("Total Time:", total_time)
#     print("Total Ease of Driving:",ease_of)
