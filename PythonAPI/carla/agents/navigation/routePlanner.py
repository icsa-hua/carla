# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


#Module to have the A* implementation and how to get the topology 
#Following the global_router example from python api examples. 

from collections import deque
import math
from tracemalloc import stop
import numpy as np
import networkx as nx
import copy
from statistics import mean

import carla
from agents.navigation.local_planner import RoadOption
from agents.navigation.energy_estimation import EnergyModelEstimation
from agents.navigation.hybridga import HybridGA

#from agents.navigation.pso_pfa import PSO
from agents.tools.misc import vector


class RoutePlanner(object): 
    def __init__(self,wmap,sampling_resolution): 
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap
        self._topology = None
        self._graph = None
        self._id_map = None
        self._road_id_to_edge = None
        self.copy_graph = None
        self._intersection_end_node = -1
        self._previous_decision = RoadOption.VOID 
        self._vehicle = None
        self._vehicle_coeffs = []
        self._energy_cost = []

        # Build the graph
        self._build_topology()
        self._build_graph()
        self._find_loose_ends()
        self._lane_change_link()

    def trace_route(self,origin,destination,vehicle,vehicle_coefficients,verbose): 
        """
        This method returns list of (carla.Waypoint, RoadOption)
        from origin to destination
        """
        self._vehicle = vehicle
        self._vehicle_coeffs = vehicle_coefficients
        self._verbose = verbose 
        route_trace = []
        best_route = []
        best_paths = []
        npaths = 5 
        self.copy_graph = nx.DiGraph(self._graph)
        current_waypoint = self._wmap.get_waypoint(origin)
        destination_waypoint = self._wmap.get_waypoint(destination)
        for _ in range(npaths):
            
            route = self._path_search(origin, destination) #Returns a list of nodes for the best possible path based on copy graph. 
            
            if not route: 
               print("Not another path available")
               break
            else: 
                print(f"The path to follow {route}")
            self.copy_graph.remove_edges_from(zip(route,route[1:]))
            route_trace.clear()

            for i in range(len(route) - 1):
                road_option = self._turn_decision(i, route)
                edge = self._graph.edges[route[i], route[i+1]]
                path = []

                if edge['type'] != RoadOption.LANEFOLLOW and edge['type'] != RoadOption.VOID:
                    route_trace.append((current_waypoint, road_option))
                    exit_wp = edge['exit_waypoint']
                    n1, n2 = self._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]
                    next_edge = self._graph.edges[n1, n2]
                    if next_edge['path']:
                        closest_index = self._find_closest_in_list(current_waypoint, next_edge['path'])
                        closest_index = min(len(next_edge['path'])-1, closest_index+5)
                        current_waypoint = next_edge['path'][closest_index]
                    else:
                        current_waypoint = next_edge['exit_waypoint']
                    route_trace.append((current_waypoint, road_option))

                else:
                    path = path + [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                    closest_index = self._find_closest_in_list(current_waypoint, path)
                    for waypoint in path[closest_index:]:
                        current_waypoint = waypoint
                        route_trace.append((current_waypoint, road_option))
                        if len(route)-i <= 2 and waypoint.transform.location.distance(destination) < 2*self._sampling_resolution:
                            break
                        elif len(route)-i <= 2 and current_waypoint.road_id == destination_waypoint.road_id and current_waypoint.section_id == destination_waypoint.section_id and current_waypoint.lane_id == destination_waypoint.lane_id:
                            destination_index = self._find_closest_in_list(destination_waypoint, path)
                            if closest_index > destination_index:
                                break
            
            best_paths.append(route_trace)
    
        tmp = best_paths.copy()
        EME = EnergyModelEstimation(vehicle=self._vehicle, vehicle_coeffs=self._vehicle_coeffs,possible_routes=tmp,wmap=self._wmap,origin=origin,road_ids=self._id_map, verbose=self._verbose)
        self._energy_cost,total_distance,total_travel_time,number_of_stops = EME.energy_estimation()
        final_factor = 1000000
        index = 0 
        for i in range(0,len(self._energy_cost)):
            print(f"Total number of stops is {number_of_stops}")
            print(f"The energy cost for {i + 1} possible paths is  | {self._energy_cost[i]} |")
            print(f"Total Distance Of Route | {total_distance[i]} | for path {i + 1}")
            print(f"The total time of the link is | {total_travel_time[i]} | for path {i + 1}")
            if final_factor > (int(self._energy_cost[i]+total_travel_time[i]+total_distance[i])):
                final_factor = (int(self._energy_cost[i]+total_travel_time[i]+total_distance[i]))
                index = i
        
        best_route = best_paths[index]

        return best_route
    

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
        for segment in self._wmap.get_topology():
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


    def _build_graph(self): 
        """
        This function builds a networkx graph representation of topology, creating several class attributes:
        - graph (networkx.DiGraph): networkx graph representing the world map, with:
            Node properties:
                vertex: (x,y,z) position in world map
            Edge properties:
                entry_vector: unit vector along tangent at entry point
                exit_vector: unit vector along tangent at exit point
                net_vector: unit vector of the chord from entry to exit
                intersection: boolean indicating if the edge belongs to an  intersection
        - id_map (dictionary): mapping from (x,y,z) to node id
        - road_id_to_edge (dictionary): map from road id to edge in the graph
        """
        self._graph = nx.DiGraph()
        self._id_map = dict()  # Map with structure {(x,y,z): id, ... }
        self._road_id_to_edge = dict()  # Map with structure {road_id: {lane_id: edge, ... }, ... }

        for segment in self._topology:
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

            for vertex in entry_xyz, exit_xyz:
                # Adding unique nodes and populating id_map
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    self._graph.add_node(new_id, vertex=vertex)
            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()

            # Adding edge with attributes
            self._graph.add_edge(
                n1, n2,
                length=len(path) + 1, path=path,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                entry_vector=np.array(
                    [entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
                exit_vector=np.array(
                    [exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),
                net_vector=vector(entry_wp.transform.location, exit_wp.transform.location),
                intersection=intersection, type=RoadOption.LANEFOLLOW)
            
            
    def _find_loose_ends(self):
        """
        This method finds road segments that have an unconnected end, and
        adds them to the internal graph representation
        """
        count_loose_ends = 0
        hop_resolution = self._sampling_resolution
        for segment in self._topology:
            end_wp = segment['exit']
            exit_xyz = segment['exitxyz']
            road_id, section_id, lane_id = end_wp.road_id, end_wp.section_id, end_wp.lane_id
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
                n1 = self._id_map[exit_xyz]
                n2 = -1*count_loose_ends
                self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
                next_wp = end_wp.next(hop_resolution)
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
                    self._graph.add_node(n2, vertex=n2_xyz)
                    self._graph.add_edge(
                        n1, n2,
                        length=len(path) + 1, path=path,
                        entry_waypoint=end_wp, exit_waypoint=path[-1],
                        entry_vector=None, exit_vector=None, net_vector=None,
                        intersection=end_wp.is_junction, type=RoadOption.LANEFOLLOW)
                    

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
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint)
                                right_found = True
                    if waypoint.left_lane_marking and waypoint.left_lane_marking.lane_change & carla.LaneChange.Left and not left_found:
                        next_waypoint = waypoint.get_left_lane()
                        if next_waypoint is not None \
                                and next_waypoint.lane_type == carla.LaneType.Driving \
                                and waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANELEFT
                            next_segment = self._localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, intersection=False, exit_vector=None,
                                    path=[], length=0, type=next_road_option, change_waypoint=next_waypoint)
                                left_found = True
                if left_found and right_found:
                    break


    def _localize(self,location): 
        """
        This function finds the road segment that a given location
        is part of, returning the edge it belongs to
        """
        waypoint = self._wmap.get_waypoint(location)
        edge = None
        try:
            edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            pass
        return edge
    

    def _distance_heuristic(self,n1,n2):
        """
        Distance heuristic calculator for path searching
        in self._graph
        """
        #print(edges)
        l1 = np.array(self._graph.nodes[n1]['vertex'])
        l2 = np.array(self._graph.nodes[n2]['vertex'])
        return np.linalg.norm(l1-l2)
    
    def _path_search(self,origin,destination): 
        """
        This function finds the shortest path connecting origin and destination
        using A* search with distance heuristic.
        origin      :   carla.Location object of start position
        destination :   carla.Location object of of end position
        return      :   path as list of node ids (as int) of the graph self._graph
        connecting origin and destination
        """

        start, end = self._localize(origin), self._localize(destination)
        route = []
        routeB = []

        #Deep copy of the original graph. 
        try:
            route = nx.astar_path(self.copy_graph, source=start[0], target=end[0],heuristic=self._distance_heuristic, weight='length')
            print("Best path based on A* is :", route)
            hga = HybridGA(start=start, finish=end ,map=self._wmap,graph=self._graph, population_size=50, generations=100, mutation_rate=0.2)
            #pso = PSO(self.copy_graph,start,end,num_particles=10,num_iterations=100)
            #routeB = pso.pso()
            
        except(KeyError, nx.exception.NetworkXNoPath):
            print("Got Exception so not another path available")
           
        #Here the energy estimation will occur for every possible path. 
        #print("Best path based on A* is :", route)
        #print("Best path based on PSO is :", routeB)

        return route
    
    def _successive_last_intersection_edge(self,index,route): 
        """
        This method returns the last successive intersection edge
        from a starting index on the route.
        This helps moving past tiny intersection edges to calculate
        proper turn decisions.
        """

        last_intersection_edge = None
        last_node = None
        for node1, node2 in [(route[i], route[i+1]) for i in range(index, len(route)-1)]:
            candidate_edge = self._graph.edges[node1, node2]
            if node1 == route[index]:
                last_intersection_edge = candidate_edge
            if candidate_edge['type'] == RoadOption.LANEFOLLOW and candidate_edge['intersection']:
                last_intersection_edge = candidate_edge
                last_node = node2
            else:
                break

        return last_node, last_intersection_edge

    def _turn_decision(self, index, route, threshold=math.radians(35)):
        """
        This method returns the turn decision (RoadOption) for pair of edges
        around current index of route list
        """

        decision = None
        previous_node = route[index-1]
        current_node = route[index]
        next_node = route[index+1]
        next_edge = self._graph.edges[current_node, next_node]
        if index > 0:
            if self._previous_decision != RoadOption.VOID \
                    and self._intersection_end_node > 0 \
                    and self._intersection_end_node != previous_node \
                    and next_edge['type'] == RoadOption.LANEFOLLOW \
                    and next_edge['intersection']:
                decision = self._previous_decision
            else:
                self._intersection_end_node = -1
                current_edge = self._graph.edges[previous_node, current_node]
                calculate_turn = current_edge['type'] == RoadOption.LANEFOLLOW and not current_edge[
                    'intersection'] and next_edge['type'] == RoadOption.LANEFOLLOW and next_edge['intersection']
                if calculate_turn:
                    last_node, tail_edge = self._successive_last_intersection_edge(index, route)
                    self._intersection_end_node = last_node
                    if tail_edge is not None:
                        next_edge = tail_edge
                    cv, nv = current_edge['exit_vector'], next_edge['exit_vector']
                    if cv is None or nv is None:
                        return next_edge['type']
                    cross_list = []
                    for neighbor in self._graph.successors(current_node):
                        select_edge = self._graph.edges[current_node, neighbor]
                        if select_edge['type'] == RoadOption.LANEFOLLOW:
                            if neighbor != route[index+1]:
                                sv = select_edge['net_vector']
                                cross_list.append(np.cross(cv, sv)[2])
                    next_cross = np.cross(cv, nv)[2]
                    deviation = math.acos(np.clip(
                        np.dot(cv, nv)/(np.linalg.norm(cv)*np.linalg.norm(nv)), -1.0, 1.0))
                    if not cross_list:
                        cross_list.append(0)
                    if deviation < threshold:
                        decision = RoadOption.STRAIGHT
                    elif cross_list and next_cross < min(cross_list):
                        decision = RoadOption.LEFT
                    elif cross_list and next_cross > max(cross_list):
                        decision = RoadOption.RIGHT
                    elif next_cross < 0:
                        decision = RoadOption.LEFT
                    elif next_cross > 0:
                        decision = RoadOption.RIGHT
                else:
                    decision = next_edge['type']

        else:
            decision = next_edge['type']

        self._previous_decision = decision
        return decision 

    def _find_closest_in_list(self, current_waypoint, waypoint_list):
        min_distance = float('inf')
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(
                current_waypoint.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index
