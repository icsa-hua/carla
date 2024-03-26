import numpy as np
import networkx as nx
from statistics import mean
from agents.navigation.local_planner import RoadOption
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt
import networkx as nx 
import warnings
import math
import carla 
import time 



class RoutePhase():
    def __init__(self): 
        self._phase1 = False 
        self._phase2 = False 
        self._phase3 = False 
        self._phase4 = False 
        self._stop = False 
        self._speed_lim = 0.0
        self._traffic_light = False  
        self._distance_link = 0.0
        self._highway = False 
        self._v_start = 0.0 
        self._v_max = 0.0 
        self._v_end = 0.0 
        self._v_max_next = 0.0 
        self._v_end_prev = 0.0 
        self._stop_time = 0.0 

class EnergyModelEstimation():
    def __init__(self, vehicle=None, vehicle_coeffs=None, possible_routes=None, wmap=None,origin=None,road_ids=None, verbose=None):
        
        self._vehicle_coefs = vehicle_coeffs
        self._A_coef = self._vehicle_coefs._A_coef
        self._B_coef = self._vehicle_coefs._B_coef
        self._C_coef = self._vehicle_coefs._C_coef
        self._d_coef = self._vehicle_coefs._d_coef
        self._g_coef = self._vehicle_coefs._g_coef
        self._ndis = self._vehicle_coefs._ndis
        self._nchg =self._vehicle_coefs._nchg
        self._naux_move = self._vehicle_coefs._naux_move
        self._naux_idle = self._vehicle_coefs._naux_idle
        self._P_aux =self._vehicle_coefs._Paux
        self._nd = self._vehicle_coefs._nd
        self._n_aux = self._vehicle_coefs._n_aux
        
        self._vehicle = vehicle
        self._wmap = wmap 
        self._verbose = verbose
        self._world = self._vehicle.get_world()
        self._physical_controls = self._vehicle.get_physics_control()
        self._mass = self._physical_controls.mass
        
        self._min_acc = 0
        self._min_dist = 0
        self._rng = np.random.default_rng(seed=42)
        self.time_travel = vehicle_coeffs._acc_time
        self.traffic_lights = [13.80, 13.84, 21.31,34.01,47.64,47.83,47.97] 
        self._stop_times = {"traffic_light" : mean(self.traffic_lights), "stop" : 3.2}
        self._number_of_stops = []
        speed_lms = self._wmap.get_all_landmarks_of_type('274')
        self._speed_lms = {lm.road_id: lm for lm in speed_lms}
        stop_lms = self._wmap.get_all_landmarks_of_type('206')
        self._stop_lms =  {lm.road_id: lm for lm in stop_lms}
        highway_lms = self._wmap.get_all_landmarks_of_type('330')
        self._highway_lms = {lm.road_id : lm for lm in highway_lms}
        
        self._traffic_lights = {}
        list_actors = self._world.get_actors()
        for actor_ in list_actors:
            if isinstance(actor_, carla.TrafficLight):
                tlwp = self._wmap.get_waypoint(actor_.get_transform().location)
                road_id_tl = tlwp.road_id
                self._traffic_lights[road_id_tl] = tlwp

        self.target_waypoint, self.target_road_option = (self._wmap.get_waypoint(origin), RoadOption.LANEFOLLOW)
        self._waypoint_queue = deque(maxlen=10000) 
        self._waypoint_queue.append((self.target_waypoint, self.target_road_option))
        self._road_ids = road_ids
        self._possible_routes = possible_routes
        self._origin = origin
        self.total_energy_link = [] 
        self._energy_cost_routes = []
        self._total_distance = [] 
        self._total_travel_time = []
        self._links = []
        
    def _remake_queue(self,current_plan, clean_queue=True):
        if clean_queue:
            self._waypoint_queue.clear()
        new_plan_length = len(current_plan) + len(self._waypoint_queue)
        
        if new_plan_length > self._waypoint_queue.maxlen:
            new_waypoint_queue = deque(max_len=new_plan_length)
            for wp in self._waypoint_queue:
                new_waypoint_queue.append(wp)
            self._waypoint_queue = new_waypoint_queue
         
        for elem in current_plan:
            self._waypoint_queue.append(elem)
        
    def _average_acceleration(self, v_initial=0, v_final=27.78, time_travel=5.7, dist_travel = 0):
            
        if time_travel is None and dist_travel!=0:
            time_travel = (2*dist_travel)/(v_final+v_initial)

        if time_travel is None:
            print(f"{v_initial}, {v_final}, {dist_travel}")

        return (v_final-v_initial)/time_travel

    def _phase_calculation(self, phase_acc, phase_speed, phase_dist,phase_time,phase_avg_end_speed,avg_slope):
        
        phase_avg_end_speed_ms = phase_avg_end_speed*0.27778
        # print("***************************************************")
        # print(f"self._A_coef {self._A_coef}")
        # print(f"avg_slope {avg_slope}")
        # print(f"self._B_coef {self._B_coef }")
        # print(f"phase_avg_end_speed {phase_avg_end_speed}")
        # print(f"self._C_coef {self._C_coef}")
        # print(f"self._mass {self._mass}")
        # print(f"self._d_coef {self._d_coef}")
        # print(f"phase_acc {phase_acc}")
        # print(f"self._g_coef {self._g_coef}")
        # print(f"np.sin(avg_slope) {np.sin(avg_slope)}")
        # print(f"np.cos(avg_slope) {np.cos(avg_slope)}")
        # print(f"phase_avg_end_speed_ms {phase_avg_end_speed_ms}")
        # print(f"phase_speed {phase_speed}")
        # print(f"phase_time {phase_time}")
        # print(f"phase_dist {phase_dist}")
        # print("***************************************************")

        force_wheels = (self._A_coef*np.cos(avg_slope)) + (self._B_coef*phase_avg_end_speed) + (self._C_coef*(phase_avg_end_speed**2))\
                      +(self._mass*self._d_coef*phase_acc) + (self._mass*self._g_coef*np.sin(avg_slope))
        Pwh = (force_wheels*phase_avg_end_speed_ms)
        Pbat = 0 

        if Pwh > 0:
            Pbat = (Pwh/self._ndis) + (self._P_aux/self._naux_move)
            
        elif Pwh < 0:
            Pbat = (Pwh*self._nchg) + (self._P_aux/self._naux_move) 

        else: 
            Pbat = (self._P_aux/self._naux_idle)

        energy = (Pbat*phase_time)/3600

        if self._verbose:
            print(f"Force in Newtons {force_wheels}")
            print(f"Power in J/s or Watts {Pwh}")
            print(f"Power in Watts {Pbat}")
            print(f"Energy is {energy}")

        # energy = ((Pwh*phase_time)/(self._nd) + (self._P_aux*phase_time)/(self._n_aux))/3600
        return energy

    def waypoints_roads(self):
        waypoints = self._wmap.get_topology()
        if waypoints:
            wp_road_ids = dict()
            for wp,_ in waypoints:
                road_id = wp.road_id
                if road_id not in wp_road_ids:
                    wp_road_ids[road_id]=[]
                wp_road_ids[road_id].append(wp)
        else:
            return 
        
        return wp_road_ids
        
    def loop_route(self):
        
        if not self._links:
            print("Error when converting the links. Exit program. ")
            return 
        
        #flag = False
        ind = True
        flinks = []
        fnal = self._links[-1][-1]
        for link in range(len(self._links)-2):
            link1 = self._links[link]
            if not link1[0] or not link1[-1]:
                continue
            link2 = self._links[link+1]
            link3 = self._links[link+2]
            p1 = link1[0].transform.location 
            p2 = link1[-1].transform.location 
            p3 = link2[0].transform.location
            p4 = link2[-1].transform.location
            d1 = p1.distance(p2) 
            d2 = p3.distance(p4)
            
            if d1 < self._min_dist: 
                if ind: 
                    flag = True 
                    ind = False
                
                if d2 < self._min_dist:
                    if d1 + d2 < self._min_dist:
                        flinks.append((link1[0],link3[-1]))
                        self._links[link] = (link1[0],None)
                        self._links[link+1] = (None,None)
                        self._links[link+2] = (None,link3[-1])
                    else: 
                        flinks.append((link1[0],link2[-1]))
                        self._links[link] = (link1[0],None)
                        self._links[link+1] = (None,link2[-1])
                else:
                    flinks.append((link1[0],link2[-1]))
                    self._links[link] = (link1[0],None)
                    self._links[link+1] = (None,link2[-1])                    
            else:
                if d2 < self._min_dist:
                    if ind: 
                        flag = True 
                        ind = False
                    flinks.append((link1[0],link2[-1]))
                    self._links[link] = (link1[0],None)
                    self._links[link+1] = (None,link2[-1])
                else:
                    flinks.append((link1[0],link2[0]))
            link+=1

        if fnal != flinks[-1][-1]: 
                flinks.append((flinks[-1][-1],self._links[-1][-1]))

        if self._verbose==True:
            for link in range(len(flinks)): 
                x1, y1 = flinks[link][0].transform.location.x, flinks[link][0].transform.location.y    
                x2, y2 = flinks[link][1].transform.location.x, flinks[link][1].transform.location.y 
                plt.plot([-x1,-x2], [y1,y2], marker = 'o')
            plt.title("Map formed inside the loop_route function")
            plt.show()
            
        self._links = [(x,y) for x,y in self._links if y is not None and x is not None]
        self._links = flinks + self._links

        return flinks
    
    def condition(self, link):
        p1 = link[0].transform.location
        p2 = link[-1].transform.location
        if p1.distance(p2) == 0 or p1.distance(p2)<0.1: 
            return False
        return True

    def link_creation(self):

        waypoints = self._waypoint_queue
        wp_road_ids = dict()
    
        for wp,_ in waypoints: 
            road_id = wp.road_id
            if road_id not in wp_road_ids:
                wp_road_ids[road_id]=[]
            wp_road_ids[road_id].append(wp)

        formatted_wp_roads = {}
        for key in wp_road_ids.keys():

            length = len(wp_road_ids[key])
            if length == 1: 
                points = [wp_road_ids[key][0]]
            else:
                points = wp_road_ids[key]

            formatted_wp_roads[key] = self._douglas_pucker(points, epsilon_curved=25, epsilon_straight=5)

        for key in formatted_wp_roads.keys():
            for ii in range(len(formatted_wp_roads[key])-1):
                wayp1 = formatted_wp_roads[key][ii]
                wayp2 = formatted_wp_roads[key][ii+1]
                self._links.append((wayp1,wayp2))

        # if self._verbose == True:
        #     for link in self._links: 
        #         x1, y1 = link[0].transform.location.x, link[0].transform.location.y    
        #         x2, y2 = link[1].transform.location.x, link[1].transform.location.y 
        #         # x3, y3 = link[2].transform.location.x, link[2].transform.location.y 
        #         plt.plot([-x1,-x2], [y1,y2], marker = 'o')
        #     plt.title("This is the one straingth after the appendind")
        #     plt.show()


        #Find and create a path with all the points
        links = []
        for link in range(len(self._links)-1):
            if (link==0):
                links.append((self._links[0][0],self._links[0][-1]))
            elif(link==len(self._links)-1):
                links.append((self._links[link][0],self._links[link][-1]))
                continue
            
            elif not (self._links[link][0] == self._links[link][-1]):
                links.append((self._links[link][0],self._links[link][1]))
                links.append((self._links[link][1],self._links[link][-1]))

            p2 = self._links[link][-1] #the final point of the tuple 
            p3 = self._links[link+1][0] # the first point of the next link 
            links.append((p2,p3))

        if self._verbose==True:
            for link in links: 
                x1, y1 = link[0].transform.location.x, link[0].transform.location.y    
                x2, y2 = link[1].transform.location.x, link[1].transform.location.y 
                plt.plot([-x1,-x2], [y1,y2], marker = 'o')
            plt.title("Graph topology after separation and link creation.")
            plt.show()

        #print the elements in the route in a diagram for clarity. 
        filtered_links = [link for link in links if self.condition(link)]
        links.clear()
        self._links.clear()         
        self._links = filtered_links
       
        #return self._links
        return self.loop_route()
        
    def calculate_incline_angle(self, cur_location, next_location):

        dot_product = np.dot(cur_location, next_location)
        magnitude1 = np.linalg.norm(cur_location)
        magnitude2 = np.linalg.norm(next_location)       
        cosine_angle = dot_product / (magnitude1 * magnitude2)
        angle_rad = np.arccos(cosine_angle)
        angle_degree =  np.degrees(angle_rad)
        return angle_degree

    def avg_calculate_incline_angle(self, start_point, end_point):

        delta_z = end_point[2] - start_point[2]
        horizontal_distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        angle_radians = np.arctan2(delta_z, horizontal_distance)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees
    
    def calculate_average_slope(self, start_point, end_point):

        delta_x = end_point[0] - start_point[0]
        delta_y = end_point[1] - start_point[1]
        delta_z = end_point[2] - start_point[2]
        
        horizontal_distance = np.sqrt(delta_x**2 + delta_y**2)
        
        # Υπολογισμός της κλίσης
        slope = delta_z / horizontal_distance if horizontal_distance != 0 else 0
        slope_radians = np.arctan(slope)
        slope_degrees = np.degrees(slope_radians)
        return slope_degrees

    def point_line_distance_3d(self, point, line_start, line_end):
        AP = np.array(point) - np.array(line_start)
        AB = np.array(line_end) - np.array(line_start)
        cross_product = np.cross(AP,AB) 
        distance = np.linalg.norm(cross_product)/np.linalg.norm(AB)
        return distance 
    
    def calculate_curvature(self, waypoints):
        p1 = waypoints[0].transform.location
        x1,y1,z1 =  p1.x, p1.y, p1.z 
        p2 = waypoints[1].transform.location
        x2,y2,z2 =  p2.x, p2.y, p2.z
        p3 = waypoints[-1].transform.location
        x3,y3,z3 =  p3.x, p3.y, p3.z
        
        v1 = np.array([x2-x1, y2-y1, z2-z1])
        v2 = np.array([x3-x2, y3-y2, z3-z2])

        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1) 
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        
        cos_theta = dot / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta,-1,1)
        angle = np.arccos(cos_theta)

        return angle    

    def key_points_of_curvature(self, points, curvature_thr):
        segment_classification = ['straight'] * (len(points)-1)
        key_points = [points[0]]
        for i in range(1, len(points)-1): 
            curvature = self.calculate_curvature(points[i-1:i+2])
            if curvature > curvature_thr:
                segment_classification[i-1] = 'curved'
                segment_classification[i] = 'curved'
                key_points.append(points[i])
        key_points.append(points[-1])
        return key_points 

    def _douglas_pucker(self,points, epsilon_curved, epsilon_straight): 

        segment_classification = self.key_points_of_curvature(points ,np.pi/95)
       
        def simplify_segment(points, epsilon):
            start_point = points[0].transform.location.x, points[0].transform.location.y, points[0].transform.location.z
            end_point = points[-1].transform.location.x, points[-1].transform.location.y, points[-1].transform.location.z
        
            dmax = 0.0 
            index = 0
            for ii in range(1,len(points)-1):
                current_point = points[ii].transform.location.x, points[ii].transform.location.y,points[ii].transform.location.z,
                d = self.point_line_distance_3d(current_point,start_point,end_point)
                if d > dmax: 
                    index = ii
                    dmax = d 
            if dmax > epsilon: 
                rec_result1 = self._douglas_pucker(points[:index+1], epsilon_curved,epsilon_straight)
                rec_result2 = self._douglas_pucker(points[index:], epsilon_curved,epsilon_straight)
                return rec_result1[:-1] + rec_result2
            else:     
                return [points[0], points[-1]]

        results = [] 
        start_index = 0 
        for i in range(1, len(segment_classification)):
            if segment_classification[i] != segment_classification[start_index] or i == len(segment_classification)-1:
                segment_points = points[start_index:i+1]
                epsilon = epsilon_curved if segment_classification[start_index] == 'curved' else epsilon_straight
                simplified_segment = simplify_segment(segment_points, epsilon)
                results += simplified_segment[:-1]
                start_index = i 
        results.append(points[-1])
        return results 
                
    def find_max_velocity(self, distance): 
        max_velocity = 0 
        best_acceleration = None 
        for(v_initial_km, v_final_km), time in self.time_travel.items():
            v_initial_ms = v_initial_km*0.27778
            v_final_ms = v_final_km*0.27778 
            acceleration = self._average_acceleration(v_initial=v_initial_ms, v_final=v_final_ms, time_travel=time, dist_travel=0)
            
            #Calculate the distance neede to reach v_final from v_initial 
            distance_needed = v_initial_ms * time + 0.5 * acceleration * (time ** 2)
            if distance_needed <= distance and v_final_ms>max_velocity: 
                max_velocity = v_final_ms 
                best_acceleration = acceleration
        return max_velocity, best_acceleration

    def energy_estimation(self):

        phases = {"accelerate":0, "steadyspeed":1, "decelerate":2, "stopstill":3}
        numP = len(phases)
        phase_acc = [1.0] * numP
        phase_speed = [1.0] * numP
        phase_dist = [1.0] * numP
        phase_time = [1.0] * numP
        phase_avg_end_speed = [1.0] * numP
        phase_energy = []
        total_distance = 0
        total_travel_time = 0
        traffic_stop_same_road = False  

        if not self._speed_lms: 
            print("No speed limit sign found. Due to urban environment the speed limit is set to 30 km/h")
            speed_flag = True
        else: 
            speed_flag = False

        while len(self._possible_routes) != 0:
            number_stops = 0

            route = self._possible_routes[0]

            if self._verbose==True:
                for link in range(len(route)-1): 
                    x1, y1 = route[link][0].transform.location.x, route[link][0].transform.location.y    
                    x2, y2 = route[link+1][0].transform.location.x, route[link+1][0].transform.location.y 
                    plt.plot([-x1,-x2], [y1,y2], marker = 'o')
                plt.title("Original graph topology generated by the pfa selection")
                plt.show()

            self._remake_queue(route)
            self._min_acc = self._average_acceleration(0,30,self.time_travel[(0,30)],dist_travel=0)
            self._min_dist = (1/2)*(30*0.27778) * self.time_travel[(0,30)] #distance travelled to reach 30km/h
            links = self.link_creation()

            #Τελική αποτύπωση της διαδρομής μετά τον υπολογισμό και την μείωση των ακμών.
            if self._verbose==True: 
                for link in range(len(links)): 
                    cur_link = links[link]
                    x1, y1 = cur_link[0].transform.location.x, cur_link[0].transform.location.y    
                    x2, y2 = cur_link[-1].transform.location.x, cur_link[-1].transform.location.y 
                    plt.plot([-x1,-x2], [y1,y2], marker = 'o')
                plt.title("Final Processed Path")
                plt.show()

            not_stop_wp = True
            traffic_stop = False 
            stop_flag = False 
            highway = False
            v_end_current_km = 0 
            v_end_prev_km = 0
            v_max_current_km = 0
            v_max_next_km = 0
            v_end_current_ms = 0 
            # v_end_prev_ms = 0
            v_max_current_ms = 0
            # v_max_next_ms = 0
            init_velocity_km = 0
            init_velocity_ms = 0

            for wps in range(len(links)-1): #Link loop

                if len(links) < 2 :
                    self._energy_cost_routes.append(0)
                    break

                waypoint = links[wps][0]
                next_waypoint = links[wps][-1]
                next_link_wp = links[wps+1][-1]

                if speed_flag:
                    self._speed_lms[waypoint.road_id] = 30
                    self._speed_lms[next_waypoint.road_id] = 30
                    self._speed_lms[next_link_wp.road_id] = 30

                if not waypoint or not next_waypoint:
                    warnings.warn(F"At current iteration {wps} no waypoint was found. \nCheck the Start and Finish point locations. ")
                    break
                
                if next_waypoint.road_id!=waypoint.road_id:
                    if self._highway_lms.get(next_waypoint.road_id):
                        highway = True
            
                current_loc = waypoint.transform.location
                next_location = next_waypoint.transform.location
                self._vehicle.set_location(current_loc)
                cur_loc = [current_loc.x, current_loc.y, current_loc.z]
                next_loc = [next_location.x, next_location.y, next_location.z]
                # incline_angle = self.calculate_incline_angle(cur_loc, next_loc)
                # avg_incline = self.avg_calculate_incline_angle(cur_loc, next_loc)
                avg_slope = self.calculate_average_slope(cur_loc,next_loc)
                link_dist = current_loc.distance(next_location) 
                total_distance += link_dist

                if next_waypoint.road_id == waypoint.road_id:
                    v_max_current_km = self._speed_lms.get(waypoint.road_id)
                    if next_waypoint.road_id == next_link_wp.road_id:
                        v_max_next_km = v_max_current_km  
                    else:
                        v_max_next_km = self._speed_lms.get(next_link_wp.road_id) 
                    v_max_current_km = self._speed_lms.get(next_waypoint.road_id) 
                    if next_waypoint.road_id == next_link_wp.road_id: 
                        v_max_next_km = v_max_current_km 
                    else: 
                        v_max_next_km = self._speed_lms.get(next_link_wp.road_id) 
                
                if not v_max_current_km: 
                    v_max_current_km = 30 

                if not v_max_next_km:
                    v_max_next_km = 30
               
                if next_waypoint.road_id in list(self._traffic_lights.keys()):
                    tl_location = self._traffic_lights[next_waypoint.road_id].transform.location
                    if next_location.distance(tl_location) < 100:
                        traffic_stop = True 
                        not_stop_wp = False
                        stop_flag = False
                        del self._traffic_lights[next_waypoint.road_id] 
                    
                stop_link = self._stop_lms.get(next_waypoint.id)
                if stop_link and not not_stop_wp: 
                    stop_flag = True
                    not_stop_wp = True
                    traffic_stop = False
                elif wps + 1 == (len(links)-1):
                    stop_flag = True
                    not_stop_wp = True
                    traffic_stop = False    
                
                if stop_flag or traffic_stop: 
                    v_end_current_km = 0
                    number_stops += 1 
                else: 
                    if v_max_next_km >= v_max_current_km:
                        v_end_current_km = v_max_current_km 
                    else: 
                        v_end_current_km = v_max_next_km 
                
                init_velocity_km = v_end_prev_km 
                v_end_prev_km = v_end_current_km 
                v_max_current_ms = v_max_current_km * 0.27778
                v_end_current_ms = v_end_current_km * 0.27778
                # v_max_next_ms = v_max_next_km * 0.27778
                init_velocity_ms = init_velocity_km * 0.27778
                # v_end_prev_ms = v_end_prev_km * 0.27778 
            
                skip_phase1 = False
                skip_phase2 = False
                skip_phase3 = False 
                skip_phase4 = False  
                if init_velocity_km >= v_max_current_km : 
                    skip_phase1 = True

                for phase_key, _ in phases.items():

                    if phase_key == "accelerate":

                        if skip_phase1:

                            phase_acc[0] = 0 
                            phase_dist[0] = 0 
                            phase_time[0] = 0
                            phase_speed[0] = 0
                            phase_avg_end_speed[0] = 0
                            energy = 0
                            phase_energy.append(energy)
                            skip_phase1=False

                        else: 

                            # print(f"Phase 1 in progress. Acceleration from initial velocity {init_velocity_km} to max velocity {v_max_current_km}")
                            
                            temp = self.time_travel.get((init_velocity_km,v_max_current_km))#\/
                            phase_acc[0] = self._average_acceleration(init_velocity_ms,v_max_current_ms,temp,dist_travel=0)#\/
                            phase_dist[0] = (v_max_current_ms**2 - init_velocity_ms**2)/(2*phase_acc[0])#\/

                            if phase_dist[0] >= link_dist and not (stop_flag or traffic_stop): #\/
                                max_vel, best_acceleration = self.find_max_velocity(distance=link_dist)
                                v_max_current_ms = max_vel
                                phase_acc[0] = best_acceleration

                                # v_max_current_km = v_max_current_km/2
                                # v_max_current_ms = v_max_current_km * 0.27778
                                # phase_acc[0] = self._average_acceleration(init_velocity_ms,v_max_current_ms,None,link_dist)

                                phase_dist[0] = (v_max_current_ms**2 - init_velocity_ms**2)/(2*phase_acc[0])
                                skip_phase2 = True 
                                skip_phase3 = True 
                                skip_phase4 = True

                            elif phase_dist[0] >= link_dist and (stop_flag or traffic_stop): #\/
                                max_vel, best_acceleration = self.find_max_velocity(distance=link_dist/2)
                                v_max_current_ms = max_vel
                                phase_acc[0] = best_acceleration

                                # print(max_vel, best_acceleration, "Fuck ywahhhhhhhh222222222222")
                                # v_max_current_km = 10
                                # v_max_current_ms = 10 * 0.27778
                                # phase_acc[0] = self._average_acceleration(init_velocity_ms,v_max_current_ms,None,link_dist/2)
                                phase_dist[0] = (v_max_current_ms**2 - init_velocity_ms**2)/(2*phase_acc[0])
                                skip_phase2 = True 
                                skip_phase3 = False 
                                skip_phase4 = False 

                            phase_time[0] = (v_max_current_ms-init_velocity_ms)/(phase_acc[0]) #\/
                            phase_speed[0] = v_max_current_km #\/ 
                            phase_avg_end_speed[0] = (0.25*init_velocity_km) + (0.75*v_max_current_km) #\/

                            energy = self._phase_calculation(phase_acc=phase_acc[0], phase_speed=phase_speed[0],
                                                             phase_dist=phase_dist[0], phase_time=phase_time[0],
                                                             phase_avg_end_speed=phase_avg_end_speed[0], avg_slope=avg_slope)
                            phase_energy.append(energy) 

                    elif phase_key == "steadyspeed" :
                        if skip_phase2:
                            phase_acc[1] = 0
                            phase_dist[1] = 0
                            phase_time[1] =0
                            phase_speed[1] = 0
                            phase_avg_end_speed[1] = 0
                            energy = 0 
                            phase_energy.append(energy)
                            skip_phase2 = False 
                            
                        elif stop_flag or traffic_stop:

                            v_end_current_km = 0
                            v_end_current_ms = 0 
                            temp = self.time_travel.get((v_max_current_km,v_end_current_km)) #\/
                            phase_acc[2] = self._average_acceleration(v_max_current_ms,v_end_current_ms,temp,dist_travel=0)#\/
                            phase_dist[2] = np.abs((v_max_current_ms**2 - v_end_current_ms**2)/(2*phase_acc[2])) #\/
                            if phase_dist[2] + phase_dist[0]> link_dist:
                                # print(f"The distance in phase 3 {phase_dist[2]} is bigger than the link's distance {link_dist}")
                                phase_acc[1] = 0
                                phase_dist[1] = 0
                                phase_time[1] = 0
                                phase_speed[1] = 0
                                phase_avg_end_speed[1] = 0
                                energy = 0 
                                phase_energy.append(energy)
                            skip_phase3 = False 
                            skip_phase4 = False 
                            energy = self._phase_calculation(phase_acc=phase_acc[1],
                                                             phase_speed=phase_speed[1],
                                                             phase_dist=phase_dist[1],
                                                             phase_time=phase_time[1],
                                                             phase_avg_end_speed=phase_avg_end_speed[1],
                                                             avg_slope=avg_slope) 
                            phase_energy.append(energy)

                        else: 

                            phase_acc[2] = 0
                            phase_dist[2] = 0  
                            # print("Phase 2 == > Entered the normal execution without final stop")
                            phase_acc[1] = 0 # a_2(k)=0
                            phase_dist[1] = link_dist - phase_dist[0] - phase_dist[2] #d2_2 = link_lenght(k) - ds_1(k) - ds_3(k)
                            phase_time[1] = phase_dist[1]/v_max_current_ms # ds_2(k) / v_max(k)
                            phase_speed[1] = v_max_current_km # v_2(k) = v_max(k)
                            phase_avg_end_speed[1] = v_max_current_km # v_avg,2(k) = v_max(k)
                            energy = self._phase_calculation(phase_acc=phase_acc[1],
                                                             phase_speed=phase_speed[1],
                                                             phase_dist=phase_dist[1],
                                                             phase_time=phase_time[1],
                                                             phase_avg_end_speed=phase_avg_end_speed[1],
                                                             avg_slope=avg_slope)       
                            phase_energy.append(energy)
                            skip_phase3 = True 
                            skip_phase4 = True 

                    elif phase_key == "decelerate":
                        if skip_phase3:
                            phase_acc[2] = 0
                            phase_dist[2] = 0
                            phase_time[2] = 0
                            phase_speed[2] = 0
                            phase_avg_end_speed[2] = 0
                            energy = 0
                            phase_energy.append(energy)
                            skip_phase3 = False

                        elif stop_flag or traffic_stop :
                            v_end_current_km = 0
                            v_end_current_ms = 0
                            # print(f"Decreasing speed as stop node was found... acceleration is {phase_acc[2]} and distance {phase_dist[2]}")
                            phase_time[2] = np.abs((v_max_current_ms-v_end_current_ms)/(2*phase_acc[2]))
                            phase_speed[2] = v_end_current_km
                            phase_avg_end_speed[2] = (0.25*v_end_current_km) +  (0.75*v_max_current_km) #vavg_3(k) = 0.25*v_end(k) + 0.75*v_max(k)
                            energy = self._phase_calculation(phase_acc=phase_acc[2],
                                                             phase_speed=phase_speed[2],
                                                             phase_dist=phase_dist[2],
                                                             phase_time=phase_time[2],
                                                             phase_avg_end_speed=phase_avg_end_speed[2],
                                                             avg_slope=avg_slope)
                            phase_energy.append(energy)
                            # print("Phase 3 == > Entered the normal deceleration")
                            
                        else: 
                            # print("Phase 3 == > No deceleration needed")
                            phase_time[2] = 0
                            phase_speed[2] = 0
                            phase_avg_end_speed[2] = 0
                            energy = self._phase_calculation(phase_acc=phase_acc[2],
                                                             phase_speed=phase_speed[2],
                                                             phase_dist=phase_dist[2],
                                                             phase_time=phase_time[2],
                                                             phase_avg_end_speed=phase_avg_end_speed[2],
                                                             avg_slope=avg_slope)
                            phase_energy.append(energy)

                    elif phase_key == "stopstill":
                        if skip_phase4: 
                            phase_acc[3] = 0
                            phase_dist[3] = 0
                            phase_time[3] = 0
                            phase_speed[3] = 0
                            phase_avg_end_speed[3] = 0
                            energy = self._phase_calculation(phase_acc=phase_acc[3],
                                                             phase_speed=phase_speed[3],
                                                             phase_dist=phase_dist[3],
                                                             phase_time=phase_time[3],
                                                             phase_avg_end_speed=phase_avg_end_speed[3],
                                                             avg_slope=avg_slope)
                            phase_energy.append(energy) 
                            skip_phase4 = False 

                        elif highway:
                            # print("Phase 4 == > Highway")
                            phase_acc[3] = 0
                            phase_dist[3] = 0
                            phase_time[3] = 0
                            phase_speed[3] = 0
                            phase_avg_end_speed[3] = 0
                            energy = self._phase_calculation(phase_acc=phase_acc[3],
                                                             phase_speed=phase_speed[3],
                                                             phase_dist=phase_dist[3],
                                                             phase_time=phase_time[3],
                                                             phase_avg_end_speed=phase_avg_end_speed[3],
                                                             avg_slope=avg_slope)
                            phase_energy.append(energy)

                        elif stop_flag: 
                            # print("Phase 2 == > Stop sign stop ")
                            phase_acc[3] = 0
                            phase_dist[3] = 0 
                            phase_time[3] = self._stop_times["stop"]
                            phase_speed[3] = 0
                            phase_avg_end_speed[3] = 0 
                            energy = self._phase_calculation(phase_acc=phase_acc[3],
                                                             phase_speed=phase_speed[3],
                                                             phase_dist=phase_dist[3],
                                                             phase_time=phase_time[3],
                                                             phase_avg_end_speed=phase_avg_end_speed[3],
                                                             avg_slope=avg_slope)
                            phase_energy.append(energy)
                        elif traffic_stop:
                            # print("Phase 4 == > Traffic light Stop ")
                            phase_acc[3] = 0
                            phase_dist[3] = 0
                            phase_time[3] = self._stop_times["traffic_light"]
                            phase_speed[3] = 0
                            phase_avg_end_speed[3] = 0 
                            energy = self._phase_calculation(phase_acc=phase_acc[3],
                                                             phase_speed=phase_speed[3],
                                                             phase_dist=phase_dist[3],
                                                             phase_time=phase_time[3],
                                                             phase_avg_end_speed=phase_avg_end_speed[3],
                                                             avg_slope=avg_slope)
                            phase_energy.append(energy) 
                        else:
                            # print("Phase 4 == > No stopping")
                            phase_acc[3] = 0
                            phase_dist[3] = 0
                            phase_time[3] = 0
                            phase_speed[3] = 0
                            phase_avg_end_speed[3] = 0
                            energy = self._phase_calculation(phase_acc=phase_acc[3],
                                                             phase_speed=phase_speed[3],
                                                             phase_dist=phase_dist[3],
                                                             phase_time=phase_time[3],
                                                             phase_avg_end_speed=phase_avg_end_speed[3],
                                                             avg_slope=avg_slope)
                            phase_energy.append(energy)

                        traffic_stop = False
                        stop_flag = False 
                        not_stop_wp = True 
                        highway = False 
                        tmp = phase_energy[0] + phase_energy[1] + phase_energy[2] + phase_energy[3] 
                        total_travel_time = total_travel_time + (phase_time[0] + phase_time[1] + phase_time[2] + phase_time[3])
                        self.total_energy_link.append(tmp)

                        #phase_acc.clear()
                        #phase_dist.clear()
                        #phase_speed.clear()
                        #phase_time.clear()
                        #phase_avg_end_speed.clear()
                        phase_energy.clear()

            
            rec = 0 
            for eelink in self.total_energy_link:
                rec = rec + eelink
            self.total_energy_link = []
            self._energy_cost_routes.append(rec)
            self._total_distance.append(total_distance)
            self._number_of_stops.append(number_stops)
            self._total_travel_time.append(total_travel_time)
            self._possible_routes.pop(0)
            self._vehicle.set_location(self._origin)

        return self._energy_cost_routes,self._total_distance,self._total_travel_time, self._number_of_stops
                