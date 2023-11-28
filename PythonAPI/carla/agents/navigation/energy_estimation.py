#energy.py

import math
import numpy as np
import networkx as nx
from statistics import mean
from agents.navigation.local_planner import RoadOption
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt
import networkx as nx 

import carla
class energy_model():
    def __init__(self, vehicle, possible_routes, wmap,origin,destination,road_ids):
        
        #print(f"Vehicle {vehicle} , Map {wmap}, Origin {origin}, Possible routes {possible_routes}")
        self._energy_cost_routes = []
        self._A_coef = 35.745
        self._B_coef = 0.38704
        self._C_coef = 0.018042
        self._d_coef = 1.1 
        self._g_coef = 9.81 
        self._ndis = 0.92 
        self._nchg = 0.63 
        self._naux_move = 0.9 
        self._naux_idle = 0.9 
        self._P_aux = 520
        self._nd = 0
        self._n_aux = 0 
        self.total_energy_link = [] 
        self._mass = 2565
        self._wmap = wmap 
        self._vehicle = vehicle
        self._possible_routes = possible_routes
        self._origin = origin
        self._min_acc = 0
        self._min_dist = 0
        speed_30 = 30 * 0.27778
        speed_60 = 60 * 0.27778 
        speed_90 = 90 * 0.27778
        speed_100 = 100 * 0.27778
        self.time_travel = {(0,speed_30): 2.40, (0,speed_60): 3.90, (0,speed_90): 6.08, (0,speed_100): 6.90, (speed_30,speed_60): 1.28, (speed_30,speed_90): 3.81,
                       (speed_60,speed_90): 1.99, (speed_90,speed_60): 1.03, (speed_90,speed_30): 2.12, (speed_90,0): 2.84, (speed_60,speed_30): 1.28, (speed_60,0): 1.78, (speed_30,0): 1.08}
        self.traffic_lights = [13.80, 13.84, 21.31,34.01,47.64,47.83,47.97]
        self._stop_times = {"traffic_light" : mean(self.traffic_lights), "stop" : 3.2}

        speed_lms = self._wmap.get_all_landmarks_of_type('274')
        self._speed_lms = {lm.road_id: lm for lm in speed_lms}
        stop_lms = self._wmap.get_all_landmarks_of_type('206')
        self._stop_lms =  {lm.road_id: lm for lm in stop_lms}
        highway_lms = self._wmap.get_all_landmarks_of_type('330')
        self._highway_lms = {lm.road_id : lm for lm in highway_lms}

        self.target_waypoint, self.target_road_option = (self._wmap.get_waypoint(origin), RoadOption.LANEFOLLOW)
        self._waypoint_queue = deque(maxlen=10000) 
        self._waypoint_queue.append((self.target_waypoint, self.target_road_option))
        self._road_ids = road_ids
        #print(road_ids)
        self._total_distance = [] 
        self._total_travel_time = []
        self._links = []
        #print(f"Stops for the map -> {self._stop_lms}")
        #print(f"Speed limits in the map -> {self._speed_lms}")


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
        #v_initial = v_initial*0.27778
        #v_final = v_final*0.27778 #turn Km/h to m/s for this link
        if time_travel is None and dist_travel!=0:
            time_travel = (2*dist_travel)/(v_final+v_initial)
        else:
            s = (1/2)*(v_final+v_initial) * time_travel 
        return (v_final-v_initial)/time_travel

    def _phase_calculation(self, phase_acc, phase_speed, phase_dist,phase_time,phase_avg_end_speed,slope_angle):
        force_wheels = (self._A_coef*np.cos(slope_angle)) + (self._B_coef*phase_avg_end_speed) + (self._C_coef*(phase_avg_end_speed**2))\
                        +(self._mass*self._d_coef*phase_acc) + (self._mass*self._g_coef*np.sin(slope_angle))
        
        Pwh = (force_wheels *phase_avg_end_speed)/3.6
        if Pwh >= 0 :
            self._nd = self._ndis
        else: 
            self._nd = (1/self._nchg)

        if phase_avg_end_speed>0:
            self._n_aux = self._naux_move
        elif phase_avg_end_speed == 0:
            self._n_aux = self._naux_idle
        
        energy = ((Pwh*phase_time)/(self._nd) + (self._P_aux*phase_time)/(self._n_aux))*(1/3600)
        return energy

    def waypoints_roads(self):
        #waypoints = self._wmap.generate_waypoints(distance = 10)
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
        
        flag = False
        ind = True

        for link in self._links: 
            x1, y1 = link[0].transform.location.x, link[0].transform.location.y    
            x2, y2 = link[-1].transform.location.x, link[-1].transform.location.y 
            print(f"Distance of link {link} is {link[0].transform.location.distance(link[-1].transform.location)}")
            plt.plot([-x1,-x2], [y1,y2], marker = 'o')
        plt.show()

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
                    print("Recursion Initialized")
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
                    print(f"Reached here to second link less than ten")
                    if ind: 
                        print("Recursion Initialized")
                        flag = True 
                        ind = False
                    flinks.append((link1[0],link2[-1]))
                    self._links[link] = (link1[0],None)
                    self._links[link+1] = (None,link2[-1])
                else:
                    flinks.append(link)
            link+=1
        if fnal != flinks[-1][-1]: 
                flinks.append((flinks[-1][-1],self._links[-1][-1]))
        #G = nx.DiGraph()
        #for link in flinks: G.add_edge(link[0],link[-1])
        #links = list(nx.topological_sort(G))
        #flinks = []
        #flinks = [(links[link],links[link+1]) for link in range(len(links)-1)]
        #del G
        #links.clear()

        for link in flinks: 
            x1, y1 = link[0].transform.location.x, link[0].transform.location.y    
            x2, y2 = link[-1].transform.location.x, link[-1].transform.location.y 
            plt.plot([-x1,-x2], [y1,y2], marker = 'o')
        plt.show()
            
        self._links = [(x,y) for x,y in self._links if y is not None and x is not None]

        #self._links.clear()
        self._links = flinks + self._links
        #if flag: self.loop_route()
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
    
        #Create the list of waypoints only. 
        #Each element is a list of wp with the same road_id. 
        for wp,_ in waypoints: 
            road_id = wp.road_id
            if road_id not in wp_road_ids:
                wp_road_ids[road_id]=[]
            wp_road_ids[road_id].append(wp)

        #Get the first and last elements in every road_id.
        #Some road_ids only have one element
        for key in wp_road_ids.keys(): 
            self._links.append((wp_road_ids[key][0],wp_road_ids[key][-1]))
        
        links = []
        #Prepei prwta na enosv ola ta shmeia
        for link in range(len(self._links)-1):
            if not (self._links[link][0] == self._links[link][-1]):
                links.append(self._links[link])

            p2 = self._links[link][-1] #the final point of the tuple 
            p3 = self._links[link+1][0] # the first point of the next link 
            links.append((p2,p3))

        #print the elements in the route in a diagram for clarity. 

        filtered_links = [link for link in links if self.condition(link)]
        links.clear()
        self._links.clear()         
        self._links = filtered_links
       
        #return self._links
        return self.loop_route()
        
        

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
        wp_road_ids = self.waypoints_roads()

        while len(self._possible_routes) != 0:
            route = self._possible_routes[0]
            self._remake_queue(route)
            self._min_acc = self._average_acceleration(0,30*0.27778,self.time_travel[(0,30*0.27778)],dist_travel=0)
            self._min_dist = (1/2)*(30*0.27778) * self.time_travel[(0,30*0.27778)] #distance travveled to reach 100km/h
            links = self.link_creation()

            print(links)
            for link in range(len(links)): 
                cur_link = links[link]
                x1, y1 = cur_link[0].transform.location.x, cur_link[0].transform.location.y    
                x2, y2 = cur_link[-1].transform.location.x, cur_link[-1].transform.location.y 
                print(cur_link[0].transform.location.distance(cur_link[-1].transform.location))
                plt.plot([-x1,-x2], [y1,y2], marker = 'o')
            
            plt.show()
        

            not_stop_wp = True
            traffic_stop = False 
            stop_flag = False 
            highway = False
            v_end_current = 0 
            v_end_prev = 0
            v_max_current = 0
            v_max_next = 0

            for wps in range(len(links)-1): #Link loop
                if len(links) < 2 :
                    self._energy_cost_routes.append(0)
                    break

                waypoint = links[wps][0]
                next_waypoint = links[wps][-1] #This is the same for both links
                
                next_link_wp = links[wps+1][-1]
                if not waypoint or not next_waypoint:
                    print(F"AT ITERATION {wps} no waypoint was found")
                    break

                if next_waypoint.road_id!=waypoint.road_id:
                    not_stop_wp = False
                    if self._highway_lms.get(next_waypoint.road_id):
                        highway = True
            

                current_loc = waypoint.transform.location
                next_location = next_waypoint.transform.location

                self._vehicle.set_location(current_loc)

                if (next_waypoint==waypoint) or (current_loc.x == next_location.x and current_loc.y == next_location.y):
                    continue

                incline = math.atan2((next_location.y-current_loc.y),(next_location.x-current_loc.x))
                slope_angle = math.degrees(incline)
                link_dist = current_loc.distance(next_location)
                total_distance += link_dist


                if next_waypoint.road_id == waypoint.road_id:
                    v_max_current = self._speed_lms.get(waypoint.road_id)
                    if next_waypoint.road_id == next_waypoint.road_id:
                        v_max_next = v_max_current
                    else:
                        v_max_next = self._speed_lms.get(next_link_wp.road_id)
                else: 
                    v_max_current = self._speed_lms.get(next_waypoint.road_id)
                    if next_waypoint.road_id == next_link_wp.road_id:
                        v_max_next = v_max_current
                    else: 
                        v_max_next = self._speed_lms.get(next_link_wp.road_id)
                
                if not v_max_current:
                    v_max_current = 30 
                else: 
                    v_max_current = v_max_current.value

                if not v_max_next:
                    v_max_next = 30 
                else: 
                    v_max_next = v_max_next.value

                v_max_current = v_max_current*0.27778
                v_max_next = v_max_next*0.27778

                if self._vehicle.get_traffic_light():
                    if next_waypoint.get_junction():
                        if self._vehicle.is_at_traffic_light() and not not_stop_wp: # and not not_stop_wp:
                            traffic_stop = True 
                            not_stop_wp = True 
                            stop_flag = False 

                stop_link = self._stop_lms.get(next_waypoint.id)
                if stop_link and not not_stop_wp: #and not_stop_wp==False :
                    print(f"Stop sign spotted at {next_waypoint.road_id}")
                    stop_flag = True
                    not_stop_wp = True
                    traffic_stop = False


                if stop_flag or traffic_stop:
                    v_end_current = 0
                else:
                    if v_max_next >= v_max_current:
                        v_end_current = v_max_current
                    else:
                        v_end_current = v_max_next
                init_velocity = v_end_prev
                v_end_prev = v_end_current
                
                #print(f" Vmac_cur = {v_max_current} | Next V_max = {v_max_next} | End speed of link = {v_end_current} | end speed of previous link = {v_end_prev}")
                print(f" Initial_speed {init_velocity} and Max speed of link {v_max_current}")

                for phase_key, _ in phases.items():
                    if phase_key == "accelerate":
                        if init_velocity == v_max_current:
                            phase_acc[0] = 0 
                            phase_dist[0] = 0 
                            phase_time[0] = 0
                            phase_speed[0] = 0
                            phase_avg_end_speed[0] = 0
                            phase_energy.append(0)
                        else: 
                            
                            temp = self.time_travel.get((init_velocity,v_max_current))
                            phase_acc[0] = self._average_acceleration(init_velocity,v_max_current,temp,dist_travel=0)
                            phase_dist[0] = (v_max_current**2 - init_velocity**2)/(2*phase_acc[0])
                            if phase_dist[0] >= link_dist and not (stop_flag or traffic_stop):
                                print(f"The length of the distance for phase 1 {phase_dist[0]} is bigger than the link's distance{link_dist}")
                                phase_acc[0] = self._average_acceleration(init_velocity,v_max_current,None,link_dist)
                                phase_dist[0] = link_dist
                            phase_time[0] = (v_max_current-init_velocity)/(2*phase_acc[0])
                            phase_speed[0] = v_max_current
                            phase_avg_end_speed[0] = (0.25*init_velocity) + (0.75*v_max_current)
                            energy = self._phase_calculation(phase_acc=phase_acc[0],
                                                             phase_speed=phase_speed[0],
                                                             phase_dist=phase_dist[0],
                                                             phase_time=phase_time[0],
                                                             phase_avg_end_speed=phase_avg_end_speed[0],
                                                             slope_angle=slope_angle)
                            phase_energy.append(energy)
                    elif phase_key == "steadyspeed" :
                        if stop_flag or traffic_stop:
                            temp = self.time_travel.get((v_max_current,v_end_current))
                            phase_acc[2] = self._average_acceleration(v_max_current,v_end_current,temp,dist_travel=0)
                            phase_dist[2] = (v_max_current**2 - v_end_current**2)/(2*phase_acc[2])
                            if phase_dist[2] > link_dist:
                                print(f"The distance in phase 3 {phase_dist[2]} is bigger than the link's distance {link_dist}")
                            
                        else: 
                            phase_acc[2] = 0
                            phase_dist[2] = 0  

                        if (phase_dist[0] + phase_dist[2]) > link_dist:
                            print(f"The sum distance of phases 1 and 3 {phase_dist[0] + phase_dist[2]} is bigger thatn the link's distance{link_dist}. ")
                            phase_energy.append(0)
                        else: 
                            phase_acc[1] = 0
                            phase_dist[1] = link_dist - phase_dist[0] - phase_dist[2]
                            phase_time[1] = phase_dist[1]/v_max_current
                            phase_speed[1] = v_max_current 
                            phase_avg_end_speed[1] = v_max_current
                            energy = self._phase_calculation(phase_acc=phase_acc[1],
                                                             phase_speed=phase_speed[1],
                                                             phase_dist=phase_dist[1],
                                                             phase_time=phase_time[1],
                                                             phase_avg_end_speed=phase_avg_end_speed[1],
                                                             slope_angle=slope_angle)       
                            phase_energy.append(energy)
                    elif phase_key == "decelerate":
                        if stop_flag or traffic_stop:
                            phase_time[2] = (v_max_current-v_end_current)/(2*phase_acc[2])
                            phase_speed[2] = v_end_current
                            phase_avg_end_speed[2] = (0.25*v_end_current) +  (0.75*v_max_current)
                            energy = self._phase_calculation(phase_acc=phase_acc[2],
                                                             phase_speed=phase_speed[2],
                                                             phase_dist=phase_dist[2],
                                                             phase_time=phase_time[2],
                                                             phase_avg_end_speed=phase_avg_end_speed[2],
                                                             slope_angle=slope_angle)
                            phase_energy.append(energy)
                        else: 
                            phase_time[2] = 0
                            phase_speed[2] = 0
                            phase_avg_end_speed[2] = 0
                            phase_energy.append(0)
                    elif phase_key == "stopstill":
                        if highway:
                            phase_acc[3] = 0
                            phase_dist[3] = 0
                            phase_time[3] = 0
                            phase_speed[3] = 0
                            phase_avg_end_speed[3] = 0
                            phase_energy.append(0)
                        elif stop_flag: 
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
                                                             slope_angle=slope_angle)
                            phase_energy.append(energy)
                        elif traffic_stop:
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
                                                             slope_angle=slope_angle)
                            phase_energy.append(energy) 
                        else:
                            phase_acc[3] = 0
                            phase_dist[3] = 0
                            phase_time[3] = 0
                            phase_speed[3] = 0
                            phase_avg_end_speed[3] = 0
                            phase_energy.append(0)
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
            self._energy_cost_routes.append(rec)
            self._total_distance.append(total_distance)
            self._total_travel_time.append(total_travel_time)
            self._possible_routes.pop(0)
            self._vehicle.set_location(self._origin)

        return self._energy_cost_routes,self._total_distance,self._total_travel_time
                
