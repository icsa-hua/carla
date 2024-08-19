from agents.navigation.dummy_application.dummy_utils.interface.ECE import EnergyEstimator
from agents.navigation.local_planner import RoadOption
from agents.navigation.dummy_application.dummy_utils.EEM.douglas_peucker import DouglasPeuckerModel
from collections import deque
from statistics import mean 

import carla 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TkAgg')
import logging
import warnings 
import pdb 


class EnergyModel(EnergyEstimator):
    
    def __init__(self, vehicle, entity, map, origins, possible_routes, road_ids, verbose):
        super().__init__(vehicle, entity, map, origins)
        
        self.verbose = verbose
        self.possible_routes = possible_routes
        self.road_ids = road_ids
        self.physical_controls = self.vehicle.get_physics_control()
        self.world = self.vehicle.get_world()
        self.waypoint_queue = deque(maxlen=10000) 
        self.total_mechancal_losses = [] 
        self.total_net_power = [] 
        self.links = [] 
        self.coef_coefficient()
        self.get_map_landmarks()


    def get_map_landmarks(self):

        self.time_travel = self.vehicle_specs.vc.acc_time
        self.traffic_lights = [13.80, 13.84, 21.31,34.01,47.64,47.83,47.97] # Insert into JSON
        self.stop_times =  {"traffic_light" : mean(self.traffic_lights), "stop" : 3.2}
        self.speed_lms = {lm.road_id: lm for lm in self.map.get_all_landmarks_of_type('274')}
        self.stop_lms =  {lm.road_id: lm for lm in self.map.get_all_landmarks_of_type('206')}
        self.highway_lms = {lm.road_id : lm for lm in self.map.get_all_landmarks_of_type('330')}
        self.target_waypoint, self.target_road_option = (self.map.get_waypoint(self.origins["destination"]), RoadOption.LANEFOLLOW)
        self.waypoint_queue.append((self.target_waypoint, self.target_road_option))

        self.TFlights = {} 
        for actor in self.vehicle.get_world().get_actors():
            if isinstance(actor, carla.TrafficLight):
                tlwp = self.map.get_waypoint(actor.get_transform().location)
                self.TFlights[tlwp.road_id] = tlwp


    def coef_coefficient(self): 
        self.vehicle_specs.vc.A_coef = self.vehicle_specs.vc.A_coef * 4.44822162
        self.vehicle_specs.vc.B_coef = self.vehicle_specs.vc.B_coef * (4.44822162/0.44704)
        self.vehicle_specs.vc.C_coef = self.vehicle_specs.vc.C_coef * (4.44822162/(0.44704**2))


    def remake_queue(self, current_plan, clean_queue=True): 
        if clean_queue: 
            self.waypoint_queue.clear()

        new_plan_length = len(current_plan) + len(self.waypoint_queue)

        if new_plan_length > self.waypoint_queue.maxlen:
            new_waypoint_queue = deque(max_len=new_plan_length)
            for wp in self.waypoint_queue:
                new_waypoint_queue.append(wp)
            self.waypoint_queue = new_waypoint_queue

        for elem in current_plan:
            self.waypoint_queue.append(elem)


    def get_average_acceleration(self, initial_velocity=0, final_velocity=27.78, time_of_travel=5.7, distance_of_travel=0):

        if time_of_travel is None and distance_of_travel!=0:
            time_of_travel = (2*distance_of_travel)/(final_velocity+initial_velocity)

        return (final_velocity-initial_velocity)/time_of_travel
    

    def phase_calculation(self, phase_acc:float, phase_speed:float, phase_distance:float, phase_time:float, phase_avg_end_speed:float, avg_slope:float)->float:
       
        phase_avg_end_speed_ms = phase_avg_end_speed*0.27778

        if phase_speed == 0.0:
           rolling_resistance = 0.0
        else: 
           rolling_resistance = self.vehicle_specs.vc.A_coef * np.cos(avg_slope)

        friction = self.vehicle_specs.vc.B_coef * phase_avg_end_speed
        drag = self.vehicle_specs.vc.C_coef * (phase_avg_end_speed**2)    

        mass = self.physical_controls.mass

        total_friction = rolling_resistance + friction + drag 
        equiv_weight = mass * self.vehicle_specs.vc.g_coef * np.sin(avg_slope)
        inertia_force = (mass * self.vehicle_specs.vc.d_coef * phase_acc)
        force_wheels = total_friction + equiv_weight + inertia_force

        total_wheel_net_power = (equiv_weight + inertia_force) * phase_avg_end_speed
        mechanical_losses = total_friction*phase_avg_end_speed
        
        self.total_mechancal_losses.append(mechanical_losses) 
        self.total_net_power.append(total_wheel_net_power)

        Pwh = (force_wheels * phase_avg_end_speed_ms)
        Pbat = 0 

        if Pwh > 0: 
            Pbat = (Pwh/self.vehicle_specs.vc.ndis) + (self.vehicle_specs.vc.Paux/self.vehicle_specs.vc.naux_move)

        elif Pwh < 0: 
            Pbat = (Pwh*self.vehicle_specs.vc.nchg) + (self.vehicle_specs.vc.Paux/self.vehicle_specs.vc.naux_move)

        else: 
            Pbat = (self.vehicle_specs.vc.Paux/self.vehicle_specs.vc.naux_move)

        energy = (Pbat * phase_time)/3600

        if self.verbose: 
            logging.info(f"Force in Newtons {force_wheels}")
            logging.info(f"Power in J/s or Watts {Pwh}")
            logging.info(f"Power in Watts {Pbat}")
            logging.info(f"Energy is {energy}")

        return energy 
    

    def link_creation(self) :

        waypoints = self.ensure_min_link_length(self.waypoint_queue)
        
        wp_road_ids = dict()
        
        for wp in waypoints: 
            if wp.road_id not in wp_road_ids: 
                wp_road_ids[wp.road_id] = []
            wp_road_ids[wp.road_id].append(wp)
        

        self.display_links_if_verbose(waypoints, "Starting Road Links")        
        self.format_road_waypoints(wp_road_ids)
        
        self.display_links_if_verbose(self.links, "After Douglas Peucker")
        self.create_links() 
        self.display_links_if_verbose(self.links, "After Link Creation")
        return self.fix_loop_route()
        

    def create_links(self):
        links = [] 
        num_links = len(self.links)

        if num_links > 0: 
            links.append((self.links[0][0], self.links[0][-1]))

        for ii in range(1, num_links): 
            current_link = self.links[ii]
            prev_link = self.links[ii-1][-1]

            if current_link[0] != current_link[-1]: 
                links.append((current_link[0], current_link[1]))
                links.append((current_link[1], current_link[-1]))

            links.append((prev_link, current_link[0]))

        filtered_links = [link for link in links if self.condition(link)]
        self.links.clear()
        self.links = filtered_links


    def condition(self, link): 
        p1 = link[0].transform.location
        p2 = link[-1].transform.location
        if p1.distance(p2) == 0 or p1.distance(p2)<0.1: 
            return False
        return True
    

    def format_road_waypoints(self, road_wps):
        formatted_road_wps = {}
        for key in road_wps.keys(): 
            length = len(road_wps[key])
            if length == 1: 
                points = [road_wps[key][0]]
            else: 
                points = road_wps[key]
        
            formatted_road_wps[key] = self.Douglas_Peucker_algorithm(points, epsilon=1, curvature=np.pi/95, epsilon_curved=25, epsilon_straight=5) 
        
        for key in formatted_road_wps.keys(): 
            for ii in range(len(formatted_road_wps[key])-1):
                wp1 = formatted_road_wps[key][ii]
                wp2 = formatted_road_wps[key][ii+1]
                self.links.append((wp1, wp2))


    def ensure_min_link_length(self, waypoints, min_link_length=3): 

        if not waypoints: 
            return [] 
        
        filtered_waypoints = [waypoints[0][0]]
        last_included_waypoint = waypoints[0][0] 

        for ii in range(len(waypoints)-1): 
            current_wp = waypoints[ii][0]
            if self.calculate_distance(last_included_waypoint.transform.location, current_wp.transform.location) >= min_link_length:
                filtered_waypoints.append(current_wp)
                last_included_waypoint = current_wp
        
        if filtered_waypoints[-1] != waypoints[-1][0]: 
            filtered_waypoints.append(waypoints[-1][0])

        return filtered_waypoints


    def fix_loop_route(self)->list:
        
        if not self.links: 
            logging.debug("No links between nodes found")
            return [] 
        
        final_links = self.process_links()
        final_links = self.handle_final_link(final_links) 
        self.display_links_if_verbose(final_links, "Map formed after fusion/filtering")
        self.update_links(final_links)

        return final_links 
        

    def decide_fusion(self, d1, d2, l1, l2, l3, link_index): 
        if d2 < self.min_distance : 
            if d1 + d2 < self.min_distance : 
                self.update_links_after_fusion(link_index, None)
                self.links[link_index+2] = (None,l3[-1])
                return l3[-1]
            
            self.update_links_after_fusion(link_index, l2[-1])
            return l2[-1]
        
        self.update_links_after_fusion(link_index, l2[-1])
        return l2[-1]
    

    def update_links_after_fusion(self, link_index, new_end):
        self.links[link_index] = (self.links[link_index][0], None)
        self.links[link_index+1] = (None, new_end)


    def process_links(self)->list:
        final_links = [] 

        for link_index in range(len(self.links)-2): 
            current_link, next_link, next_next_link = self.links[link_index:link_index+3]

            if not (current_link[0] and current_link[-1]): 
                continue

            dist1 = current_link[0].transform.location.distance(current_link[-1].transform.location)
            dist2 = next_link[0].transform.location.distance(next_link[-1].transform.location)
            
            if dist1 < self.min_distance : 
                new_end = self.decide_fusion(dist1,dist2, current_link, next_link, next_next_link)
                final_links.append((current_link[0], new_end))
                
            else: 
                if dist2  < self.min_distance: 
                    self.update_links_after_fusion(link_index, next_link[-1])
                    final_links.append((current_link[0], next_link[-1]))
                final_links.append((current_link[0], next_link[0]))

        return final_links    
        
    
    def update_links(self, final_links):
        self.links = [(x, y) for x, y in self.links if y is not None and x is not None]
        self.links = final_links + self.links


    def display_links_if_verbose(self, final_links, txt=""):
        if self.verbose and txt == "True":
            for link in final_links:
                x1, y1 = link[0].transform.location.x, link[0].transform.location.y
                x2, y2 = link[1].transform.location.x, link[1].transform.location.y
                plt.plot([-x1, -x2], [y1, y2], marker='o')
            plt.title(txt)
            plt.show()


    def handle_final_link(self, final_links):
        if self.links[-1][-1] != final_links[-1][-1]:
            if self.links[-1][0] == final_links[-1][-1]:
                final_links.append((final_links[-1][-1], self.links[-1][0]))
                final_links.append((self.links[-1][0], self.links[-1][-1]))
            else:
                final_links.append((final_links[-1][-1], self.links[-1][-1]))
        return final_links


    def calculate_slope(self, source_node:int, target_node:int)->float:
        delta_x = target_node[0] - source_node[0]
        delta_y = target_node[1] - source_node[1]
        delta_z = target_node[2] - source_node[2]
        
        horizontal_distance = np.sqrt(delta_x**2 + delta_y**2)
        
        slope = delta_z / horizontal_distance if horizontal_distance != 0 else 0
        slope_radians = np.arctan(slope)
        slope_degrees = np.degrees(slope_radians)

        return slope_degrees


    def calculate_distance(self, source_node:int, target_node:int)->float:
        return np.linalg.norm(np.array([target_node.x - source_node.x, target_node.y - source_node.y, target_node.z - source_node.z]))


    def Douglas_Peucker_algorithm(self, points:list, epsilon:float, curvature:float, epsilon_curved:float, epsilon_straight:float)->list:
        dpa = DouglasPeuckerModel(points, epsilon, curvature, epsilon_curved, epsilon_straight)
        return dpa.run()


    def _run_init(self): 
        self.phases = {"accelerate":self.phase_accelerate, "steadyspeed":self.phase_steadyspeed, "decelerate":self.phase_deceleration, "stopstill":self.phase_stopstill}
        self.phase_matrix = {key: {"acc":1.0, "speed":1.0, "dist":1.0, "time":1.0, "avg_end_speed":1.0} for key in self.phases}
        self.phase_energy = []
        self.total_distance = 0 
        self.total_distance_list = []
        self.total_travel_time = 0 
        self.total_travel_time_list = []
        self.energy_cost_routes = [] 
        self.total_energy_link = [] 
        self.number_of_stops = [] 

        
    def process_route(self, route): 
        self.display_links_if_verbose(route)
        self.remake_queue(route)
        self.min_acc = self.get_average_acceleration(0,30, self.time_travel[(0,30)], distance_of_travel=0)
        self.min_distance = (1/2)*(5*0.27778)*self.time_travel[(0,5)]

        links = self.link_creation()

        self.display_links_if_verbose(links, "Links Created at Process Route")
        return links
    

    def set_initial_phase_state(self): 
        return  {
            "no_stop_at_wp": True, 
            "traffic_stop": False, 
            "junction":False, 
            "stop_flag":False, 
            "highway":False, 
            "v_end_current_km": 0, 
            "v_end_prev_km":0, 
            "v_max_current_km":0,
            "v_max_next_km":0,
            "v_end_current_ms":0,
            "v_max_current_ms":0,
            "init_velocity_km":0,
            "init_velocity_ms":0,
            "number_of_stops":0,
            "avg_slope":0,
            "skip_phase_1":False, 
            "skip_phase_2":False,
            "skip_phase_3":False,
            "skip_phase_4":False,
        }


    def energy_estimate(self): 
        while self.possible_routes:
            
            route = self.possible_routes.pop(0)
            trip = self.process_route(route)
            self.phase_state = self.set_initial_phase_state()
            num_links = len(trip)
            visited_wp = [] 
            for wps in range(num_links-1): 


                if num_links <= 2: 
                    self.energy_cost_routes.append(0)
                    break 

                waypoint = trip[wps][0]
                next_waypoint = trip[wps][-1]
                next_link_wp = trip[wps+1][-1]

                if waypoint in visited_wp:
                    continue
                visited_wp.append(waypoint)

                if not waypoint or not next_waypoint:
                    warnings.warn(F"At current iteration {wps} no waypoint was found. \nCheck the Start and Finish point locations. ")
                    break

                self.run_phase(wps, waypoint, next_waypoint, next_link_wp, trip)

                for phase in self.phases.keys():
                    
                    self.execute_phases(phase, waypoint, next_waypoint)
                    self.display_phase_stats(phase)

                self.get_phases_sum()            

            self.get_route_metrics()

        self.total_mechancal_losses = [sum(self.total_mechancal_losses[i:i+4]) for i in range(0, len(self.total_mechancal_losses), 4)]
        self.total_net_power = [sum(self.total_net_power[i:i+4]) for i in range(0, len(self.total_net_power), 4)]

        self.display_mech_losses()
        

    def get_phase_parameters(self, waypoint, next_waypoint, next_link_wp, wps, links):

        if next_waypoint.road_id!=waypoint.road_id:
                    if self.highway_lms.get(next_waypoint.road_id):
                        self.phase_state["highway"] = True

        if self.speed_flag: 
            self.speed_lms[waypoint.road_id] = 30
            self.speed_lms[next_waypoint.road_id] = 30
            self.speed_lms[next_link_wp.road_id] = 30
        
        self.phase_state["v_max_current_km"] = self.speed_lms.get(waypoint.road_id, 30)

        if isinstance(self.phase_state["v_max_current_km"], carla.Landmark):
            self.phase_state["v_max_current_km"] = self.phase_state["v_max_current_km"].value

        if next_waypoint.road_id == next_link_wp.road_id:
            self.phase_state["v_max_next_km"] = self.phase_state["v_max_current_km"]
        else: 
            self.phase_state["v_max_next_km"] = self.speed_lms.get(next_link_wp.road_id, 30)

        if isinstance(self.phase_state["v_max_next_km"], carla.Landmark):
            self.phase_state["v_max_next_km"] = self.phase_state["v_max_next_km"].value

        next_location = next_waypoint.transform.location

        if next_waypoint.road_id in self.TFlights.keys(): 
            tl_location = self.TFlights[next_waypoint.road_id].transform.location
            if next_location.distance(tl_location) < 100:
                self.phase_state["traffic_stop"] = True
                self.phase_state["stop_flag"] = False
                self.phase_state["no_stop_at_wp"] = False
                # del self.TFlights[next_waypoint.road_id]

        if next_waypoint.road_id in self.stop_lms.keys() and not self.phase_state["no_stop_at_wp"]: 
            self.phase_state["no_stop_at_wp"] = False
            self.phase_state["stop_flag"] = True

        if wps + 1 == len(links)-1: 
            self.phase_state["no_stop_at_wp"] = False
            self.phase_state["stop_flag"] = True

        if next_waypoint.is_junction:
            self.phase_state["junction"] = True 
            self.phase_state["no_stop_at_wp"] = False

        if self.phase_state["stop_flag"] or self.phase_state["junction"] or self.phase_state["traffic_stop"]:
            
            self.phase_state["v_end_current_km"] = 0
            self.phase_state["number_of_stops"] += 1 

        else: 
            self.phase_state["v_end_current_km"] = min(self.phase_state["v_max_current_km"], self.phase_state["v_max_next_km"])

        self.phase_state["init_velocity_km"] = self.phase_state["v_end_prev_km"]
        self.phase_state["v_end_prev_km"] = self.phase_state["v_end_current_km"]
        self.phase_state["v_max_current_ms"] = self.phase_state["v_max_current_km"]*0.27778
        self.phase_state["v_end_current_ms"] = self.phase_state["v_end_current_km"]*0.27778
        self.phase_state["init_velocity_ms"] = self.phase_state["init_velocity_km"]*0.27778
        
        self.phase_state["skip_phase_1"] = self.phase_state["init_velocity_km"] >= self.phase_state["v_max_current_km"]
        self.phase_state["skip_phase_2"] = self.phase_state["skip_phase_3"] = self.phase_state["skip_phase_4"] = False
        

    def run_phase(self, wps, waypoint, next_waypoint, next_link_wp, links):
    
        current_location = waypoint.transform.location
        next_location = next_waypoint.transform.location
        self.vehicle.set_location(current_location)

        cur_loc = [current_location.x, current_location.y, current_location.z]
        next_loc = [next_location.x, next_location.y, next_location.z]
        self.phase_state['avg_slope'] = self.calculate_slope(cur_loc, next_loc)
        link_distance = current_location.distance(next_location)

        self.total_distance += link_distance

        self.get_phase_parameters(waypoint=waypoint, next_waypoint=next_waypoint, next_link_wp=next_link_wp, wps=wps, links=links)


    def run(self):
        
        self._run_init()

        if self.speed_lms:
            self.speed_flag = False
        else: 
            warnings.warn(f"No speed limit landmarks found on map {self.map}")
            self.speed_flag = True
        
        if not any(self.possible_routes): 
            return None, None, None, None 
    
        self.energy_estimate()
        return self.total_distance_list, self.total_travel_time_list, self.energy_cost_routes, self.number_of_stops


    def reset_phase_variables(self,phase, txt=""): 
        
        for attr in ["acc", "speed", "dist", "time", "avg_end_speed"]:
            self.phase_matrix[phase][attr] = 0
        self.phase_energy.append(0)
        
        warnings.warn(f"{txt} is not executed phase matrix reset")
        

    def find_max_velocity(self, distance, phase):
        max_vel = 0 
        best_acceleration = 0   

        for (v_init, v_final), time in self.time_travel.items():
            if phase == 'decelerate' and v_final > v_init:
                continue 
            
            if phase == 'accelerate' and v_final < v_init:
                continue

            v_init = v_init * 0.27778
            v_final = v_final * 0.27778

            a = self.get_average_acceleration(v_init, v_final, time, distance)
            distance_factor = np.abs(v_init * time + 0.5 * a * (time**2))

            if distance_factor <= distance and v_final >= max_vel:
                max_vel = v_final
                best_acceleration = a


        return max_vel, best_acceleration


    def execute_phases(self, phase, *args, **kwargs): 
        phase_method = self.phases.get(phase)
        if phase_method: 
            phase_method(*args, **kwargs)
        else: 
            warnings.warn("No valid phase was used.")
        

    def adjust_velocity_for_distance(self,link_dist):
        if self.phase_matrix['accelerate']['dist'] >= link_dist:
            distance_factor = link_dist/2 if (self.phase_state["stop_flag"] or self.phase_state["junction"] or self.phase_state["traffic_stop"]) else link_dist
            max_vel, best_acceleration = self.find_max_velocity(distance=distance_factor, phase='accelerate')
            self.phase_matrix['accelerate']['acc'] = best_acceleration 
            self.phase_matrix['accelerate']['speed'] = max_vel
            self.phase_state['v_max_current_ms'] = max_vel 
            self.phase_matrix['accelerate']['dist'] = distance_factor
            # self.phase_matrix['accelerate']['dist'] = (max_vel**2 - self.phase_state["init_velocity_ms"]**2) / (2 * self.phase_matrix['accelerate']['acc'])
            self.phase_state['skip_phase_2'] = True 
            self.phase_state['skip_phase_3'] = False  if (self.phase_state["stop_flag"] or self.phase_state["junction"] or self.phase_state["traffic_stop"]) else True 
            self.phase_state['skip_phase_4'] = False  if (self.phase_state["stop_flag"] or self.phase_state["junction"] or self.phase_state["traffic_stop"]) else True


    def phase_accelerate(self, waypoint, next_waypoint):

        if self.phase_state["skip_phase_1"]:
            self.reset_phase_variables(phase="accelerate", txt="Phase 1 / Acceleration")
            self.phase_state["skip_phase_1"] = False
            return 
        
        link_dist = waypoint.transform.location.distance(next_waypoint.transform.location) 
        time_travel = self.time_travel.get((self.phase_state["init_velocity_km"], self.phase_state["v_max_current_km"]), None)
        self.phase_matrix['accelerate']['acc'] = self.get_average_acceleration(self.phase_state["init_velocity_ms"],self.phase_state["v_max_current_ms"],time_travel,distance_of_travel=0)
        self.phase_matrix['accelerate']['dist'] = (self.phase_state["v_max_current_ms"]**2 - self.phase_state["init_velocity_ms"]**2)/(2*self.phase_matrix['accelerate']['acc'])
        
        self.adjust_velocity_for_distance(link_dist)

        self.phase_matrix['accelerate']['time'] = (self.phase_state["v_max_current_ms"]-self.phase_state["init_velocity_ms"])/self.phase_matrix['accelerate']['acc']
        self.phase_matrix['accelerate']['speed'] = self.phase_state["v_max_current_km"]
        self.phase_matrix['accelerate']['avg_end_speed'] = (0.25*self.phase_state["init_velocity_km"]) + (0.75*self.phase_matrix['accelerate']['speed'])

        energy = self.phase_calculation(
            phase_acc=self.phase_matrix['accelerate']['acc'],
            phase_distance=self.phase_matrix['accelerate']['dist'],
            phase_time=self.phase_matrix['accelerate']['time'],
            phase_speed=self.phase_matrix['accelerate']['speed'],
            phase_avg_end_speed=self.phase_matrix['accelerate']['avg_end_speed'],
            avg_slope=self.phase_state["avg_slope"],
        )

        self.phase_energy.append(energy)


    def handle_stop_phase(self,link_dist): 
        self.phase_state["v_end_current_km"] = 0
        self.phase_state['v_end_current_ms'] = 0

        temp = self.time_travel.get((self.phase_state["v_max_current_km"],self.phase_state["v_end_current_km"]))
        self.phase_matrix['decelerate']['acc'] = self.get_average_acceleration(
            self.phase_state["v_max_current_ms"],
            self.phase_state["v_end_current_ms"],
            temp,
            link_dist
        )
        self.phase_matrix['decelerate']['dist'] = np.abs((self.phase_state["v_max_current_ms"]**2 - self.phase_state["v_end_current_ms"]**2)/(2*self.phase_matrix['decelerate']['acc']))

        if self.phase_matrix['decelerate']['dist'] + self.phase_matrix['accelerate']['dist'] >= link_dist: 
            self.reset_phase_variables(phase="steadyspeed", txt="Phase 2 / Steady Speed")
            self.phase_state["skip_phase_2"] = False if not (self.phase_state["stop_flag"] or self.phase_state["junction"] or self.phase_state["traffic_stop"]) else True
            
        
        energy = self.phase_calculation(
            phase_acc=self.phase_matrix['steadyspeed']['acc'],
            phase_distance=self.phase_matrix['steadyspeed']['dist'],
            phase_time=self.phase_matrix['steadyspeed']['time'],
            phase_speed=self.phase_matrix['steadyspeed']['speed'],
            phase_avg_end_speed=self.phase_matrix['steadyspeed']['avg_end_speed'],
            avg_slope=self.phase_state["avg_slope"],
        )

        self.phase_energy.append(energy)

        self.phase_state['skip_phase_3'] = False 
        self.phase_state['skip_phase_4'] = False


    def handle_normal_phase(self,link_dist):
        self.phase_matrix['decelerate']['acc'] = 0 
        self.phase_matrix['decelerate']['dist'] = 0

        self.phase_matrix['steadyspeed']['acc'] = 0
        self.phase_matrix['steadyspeed']['dist'] = link_dist - self.phase_matrix['accelerate']['dist'] - self.phase_matrix['decelerate']['dist']
        self.phase_matrix['steadyspeed']['speed'] = self.phase_state["v_max_current_km"]
        self.phase_matrix['steadyspeed']['time'] = self.phase_matrix['steadyspeed']['dist']/self.phase_state['v_max_current_ms']
        self.phase_matrix['steadyspeed']['avg_end_speed'] = self.phase_matrix['steadyspeed']['speed']

        energy = self.phase_calculation(
            phase_acc=self.phase_matrix['steadyspeed']['acc'],
            phase_distance=self.phase_matrix['steadyspeed']['dist'],
            phase_time=self.phase_matrix['steadyspeed']['time'],
            phase_speed=self.phase_matrix['steadyspeed']['speed'],
            phase_avg_end_speed=self.phase_matrix['steadyspeed']['avg_end_speed'],
            avg_slope=self.phase_state["avg_slope"],
        )
        self.phase_energy.append(energy)
        self.phase_state['skip_phase_3'] = True 
        self.phase_state['skip_phase_4'] = True


    def phase_steadyspeed(self, waypoint, next_waypoint):

        if self.phase_state["skip_phase_2"]:
            self.reset_phase_variables(phase="steadyspeed", txt="Phase 2 / Steady Speed")
            self.phase_state["skip_phase_2"] = False if not (self.phase_state["stop_flag"] or self.phase_state["junction"] or self.phase_state["traffic_stop"]) else True
            return 

        link_dist = waypoint.transform.location.distance(next_waypoint.transform.location) 

        if self.phase_state["junction"] or self.phase_state["traffic_stop"] or self.phase_state["stop_flag"]:
            self.handle_stop_phase(link_dist)
            return 

        self.handle_normal_phase(link_dist)

    
    def handle_stop_phase_deceleration(self,link_dist):
        self.phase_state["v_end_current_km"] = 0
        self.phase_state['v_end_current_ms'] = 0

        if self.phase_state["skip_phase_2"]: 
            self.phase_state["skip_phase_2"] = True 
            
            max_vel, best_deceleration = self.find_max_velocity(distance=link_dist-self.phase_matrix['accelerate']['dist'], phase="decelerate")
            
            self.phase_state["v_max_current_ms"] = max_vel
            self.phase_matrix['decelerate']['acc'] = best_deceleration
            self.phase_matrix['decelerate']['dist'] = np.abs((self.phase_state["v_max_current_ms"]**2 - self.phase_state["v_end_current_km"]**2)/(2*self.phase_matrix['decelerate']['acc']))
        
        self.phase_matrix['decelerate']['time'] = np.abs((self.phase_state["v_max_current_ms"]-self.phase_state["v_end_current_ms"])/(2*self.phase_matrix['decelerate']['acc']))
        self.phase_matrix['decelerate']['speed'] = self.phase_state["v_max_current_km"]
        self.phase_matrix['decelerate']['avg_end_speed'] = (0.25*self.phase_state["v_end_current_km"]) +  (0.75*self.phase_state["v_max_current_km"])
       
        energy = self.phase_calculation(
            phase_acc=self.phase_matrix['decelerate']['acc'],
            phase_distance=self.phase_matrix['decelerate']['dist'],
            phase_time=self.phase_matrix['decelerate']['time'],
            phase_speed=self.phase_matrix['decelerate']['speed'],
            phase_avg_end_speed=self.phase_matrix['decelerate']['avg_end_speed'],
            avg_slope=self.phase_state["avg_slope"],
        )
        self.phase_energy.append(energy)


    def handle_normal_phase_deceleration(self,link_dist):
        logging.debug("No deceleration is enabled...")
        self.phase_matrix['decelerate']['time'] = 0
        self.phase_matrix['decelerate']['speed'] = 0
        self.phase_matrix['decelerate']['avg_end_speed'] = 0

        energy = self.phase_calculation(
            phase_acc=self.phase_matrix['decelerate']['acc'],
            phase_distance=self.phase_matrix['decelerate']['dist'],
            phase_time=self.phase_matrix['decelerate']['time'],
            phase_speed=self.phase_matrix['decelerate']['speed'],
            phase_avg_end_speed=self.phase_matrix['decelerate']['avg_end_speed'],
            avg_slope=self.phase_state["avg_slope"],
        )

        self.phase_energy.append(energy)


    def phase_deceleration(self, waypoint, next_waypoint):
        if self.phase_state["skip_phase_3"]:
            self.reset_phase_variables(phase="decelerate", txt="Phase 3 / Deceleration")
            self.phase_state["skip_phase_3"] = False
            return
        
        link_dist = waypoint.transform.location.distance(next_waypoint.transform.location) 
        
        if self.phase_state["junction"] or self.phase_state["traffic_stop"] or self.phase_state["stop_flag"]:
            self.handle_stop_phase_deceleration(link_dist)
            return
        
        self.handle_normal_phase_deceleration(link_dist)


    def handle_stop_phase_stopstill(self, stop_time):
        self.phase_matrix['stopstill']['time'] = self.stop_times[stop_time]
        self.phase_matrix['stopstill']['speed'] = 0
        self.phase_matrix['stopstill']['avg_end_speed'] = 0
        self.phase_matrix['stopstill']['dist'] = 0
        self.phase_matrix['stopstill']['acc'] = 0
        energy = self.phase_calculation(
            phase_acc=self.phase_matrix['stopstill']['acc'],
            phase_distance=self.phase_matrix['stopstill']['dist'],
            phase_time=self.phase_matrix['stopstill']['time'],
            phase_speed=self.phase_matrix['stopstill']['speed'],
            phase_avg_end_speed=self.phase_matrix['stopstill']['avg_end_speed'],
            avg_slope=self.phase_state["avg_slope"],
        )
        self.phase_energy.append(energy)


    def phase_stopstill(self, waypoint, next_waypoint):
        if self.phase_state["skip_phase_4"]:
            self.reset_phase_variables(phase="stopstill", txt="Phase 4 / Stop Still")
            self.phase_state["skip_phase_4"] = False
            return

        if self.phase_state['highway']: 
            self.reset_phase_variables(phase="stopstill", txt="Phase 4 / Stop Still")
            return

        if self.phase_state["stop_flag"] or self.phase_state["junction"]: 
            self.handle_stop_phase_stopstill(stop_time="stop")
            return

        if self.phase_state["traffic_stop"]:
            self.handle_stop_phase_stopstill(stop_time="traffic_light")
            return

        self.reset_phase_variables(phase="stopstill", txt="Phase 4 / Stop Still")
        

        
    def get_phases_sum(self): 

        self.phase_state['traffic_stop'] = self.phase_state['stop_flag'] = self.phase_state['junction'] = self.phase_state['highway'] = False
        self.phase_state['no_stop_at_wp'] = True 

        total_phase_energy = sum(self.phase_energy)
        self.total_travel_time += sum([self.phase_matrix[phase]['time'] for phase in self.phases.keys()])
        self.total_energy_link.append(total_phase_energy)
        self.phase_energy.clear()
        

    def get_route_metrics(self): 
        rec = sum(self.total_energy_link)
        self.total_energy_link.clear() 
        self.energy_cost_routes.append(rec)
        self.total_distance_list.append(self.total_distance)
        self.total_travel_time_list.append(self.total_travel_time)
        self.number_of_stops.append(self.phase_state['number_of_stops'])
        self.vehicle.set_location(self.origins['start'])
        

    def display_mech_losses(self):

        if self.verbose: 
            for ii in range(len(self.total_mechancal_losses)): 
                x1, y1 = ii, self.total_mechancal_losses[ii]
                x2, y2 = ii, self.total_net_power[ii] 
                plt.plot(x1, y1, marker='o', color='r' )
                plt.plot(x2, y2, marker='o', color='b' )
            plt.legend(["Total Mechanical Losses","Total Wheel Net Power"])
            plt.xlabel("Time Steps")
            plt.title("Evaluation Diagram")
            plt.show()
            

    def display_phase_stats(self, phase=""): 
        if self.verbose:
            details = {
                "Phase ": phase,
                "self._A_coef": self.vehicle_specs.vc.A_coef,
                "avg_slope": self.phase_state["avg_slope"],
                "self._B_coef": self.vehicle_specs.vc.B_coef,
                "phase_avg_end_speed": self.phase_matrix[phase]["avg_end_speed"],
                "self._C_coef": self.vehicle_specs.vc.C_coef,
                "self._mass": self.physical_controls.mass,
                "self._d_coef": self.vehicle_specs.vc.d_coef,
                "phase_acc": self.phase_matrix[phase]["acc"],
                "self._g_coef": self.vehicle_specs.vc.g_coef,
                "np.sin(avg_slope)": np.sin(self.phase_state["avg_slope"]),
                "np.cos(avg_slope)": np.cos(self.phase_state["avg_slope"]),
                "phase_speed": self.phase_matrix[phase]["speed"],
                "phase_time": self.phase_matrix[phase]["time"],
                "phase_distance": self.phase_matrix[phase]["dist"]
            }

            print("***************************************************")
            for key, value in details.items():
                print(f"{key}: {value}")
            print("***************************************************")

