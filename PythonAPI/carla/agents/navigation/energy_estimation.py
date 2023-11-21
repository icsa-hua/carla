#energy.py

import math
import numpy as np
import networkx as nx
from statistics import mean
from agents.navigation.local_planner import RoadOption
from collections import deque


import carla
class energy_model():
    def __init__(self, vehicle, possible_routes, wmap,origin,destination):
        
        print(f"Vehicle {vehicle} , Map {wmap}, Origin {origin}, Possible routes {possible_routes}")
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
        self.time_travel = {(0,30): 2.40, (0,60): 3.90, (0,90): 6.08, (0,100): 6.90, (30,60): 1.28, (30,90): 3.81,
                       (60,90): 1.99, (90,60): 1.03, (90,30): 2.12, (90,0): 2.84, (60,30): 1.28, (60,0): 1.78, (30,0): 1.08}
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

        print(f"Stops for the map -> {self._stop_lms}")
        print(f"Speed limits in the map -> {self._speed_lms}")


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


    def _average_acceleration(self, v_initial=0, v_final=27.78, time_travel=5.7):
        print(f"Vinitital {v_initial} and V_final {v_final} with time P{time_travel}")
        v_initial = v_initial*0.27778
        v_final = v_final*0.27778 #turn Km/h to m/s for this link

        s = (1/2)*(v_final+v_initial) * time_travel #distance travveled to reach 100km/h

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

    def energy_estimation(self):
        phases = {"accelerate":0, "steadyspeed":1, "decelerate":2, "stopstill":3}
        numP = len(phases)
        phase_acc = [1.0] * numP
        phase_speed = [1.0] * numP
        phase_dist = [1.0] * numP
        phase_time = [1.0] * numP
        phase_avg_end_speed = [1.0] * numP
        phase_energy = []
        end_speed_link = 0
        end_speed_prev = 0
        init_velocity = end_speed_prev

        while len(self._possible_routes) != 0:
            route = self._possible_routes[0]
            self._remake_queue(route)

            for wp in range(len(self._waypoint_queue)-1):
                if len(self._waypoint_queue) < 2: 
                    break 

                waypoint,_=self._waypoint_queue[wp]
                next_waypoint,_ = self._waypoint_queue[wp + 1]
                cur_loc = waypoint.transform.location 
                next_loc = next_waypoint.transform.location
                #print(f"Current Position {cur_loc} and next Position {next_loc}")
                if cur_loc.x == next_loc.x or cur_loc.y == next_loc.y: 
                    continue 
                
                self._vehicle.set_location(cur_loc)
                link_dist = cur_loc.distance(next_loc)

                max_speed_lim = self._speed_lms.get(waypoint.road_id)
                if next_waypoint.road_id == waypoint.road_id:
                    next_max_speed_lim = max_speed_lim
                else: 
                    next_max_speed_lim = self._speed_lms.get(next_waypoint.road_id) 

                if not max_speed_lim: 
                    max_speed_lim = 30 
                else: 
                    max_speed_lim = max_speed_lim.value 
                
                if not next_max_speed_lim:
                    next_max_speed_lim = 30 
                else: 
                    next_max_speed_lim = next_max_speed_lim.value 
                
                traffic_stop = False 
                if waypoint.get_junction():
                    if(self._vehicle.is_at_traffic_light()): 
                        traffic_stop = True 

                incline = (next_loc.y - cur_loc.y)/(next_loc.x - cur_loc.x)
                slope_angle = math.degrees(math.atan(incline))

                stop_flag = False 
                stop_link = self._stop_lms.get(waypoint.road_id)
                if stop_link:
                    stop_link = True
                
                if stop_flag:
                        end_speed_link = 0
                else: 
                    if next_max_speed_lim >= max_speed_lim:
                        end_speed_link = max_speed_lim
                    else: 
                        end_speed_link = next_max_speed_lim
                init_velocity = end_speed_prev

                for phase_key, phase_value in phases.items(): 
                    if phase_key == "accelerate": 
                        if init_velocity == max_speed_lim:
                            phase_acc[phase_value] = 0
                            phase_dist[phase_value] = 0 
                            phase_time[phase_value] = 0
                            phase_speed[phase_value] = 0
                            phase_avg_end_speed[phase_value] = 0
                            phase_energy.append(0)
                            continue
                        temp = self.time_travel.get((init_velocity,max_speed_lim))
                        phase_acc[phase_value] = self._average_acceleration(init_velocity,max_speed_lim,temp)
                        phase_dist[phase_value] = (max_speed_lim**2 - init_velocity**2)/(2*phase_acc[phase_value])
                        phase_time[phase_value] = (max_speed_lim - init_velocity)/(2*phase_acc[phase_value])
                        phase_speed[phase_value] = max_speed_lim
                        phase_avg_end_speed[phase_value] = (0.25*init_velocity) + (0.75*max_speed_lim)
                        energy = self._phase_calculation(phase_acc[phase_value],phase_speed[phase_value],phase_dist[phase_value],phase_time[phase_value],phase_avg_end_speed[phase_value],slope_angle)
                        phase_energy.append(energy)
                        init_velocity = max_speed_lim
                    elif phase_key == "steadyspeed":
                        if stop_flag or traffic_stop:
                            temp = self.time_travel.get((max_speed_lim,end_speed_link))
                            phase_acc[phase_value+1] = self._average_acceleration(max_speed_lim,end_speed_link,temp)
                            phase_dist[phase_value+1] = (max_speed_lim**2 - end_speed_link)/(2*phase_acc[phase_value+1])
                        else:
                            phase_acc[phase_value+1] = 0 
                            phase_dist[phase_value+1] = 0
                        
                        phase_acc[phase_value] = 0
                        phase_dist[phase_value] = link_dist - phase_dist[phase_value-1] - phase_dist[phase_value+1]
                        phase_time[phase_value] = phase_dist[phase_value] / max_speed_lim
                        phase_speed[phase_value] = max_speed_lim
                        phase_avg_end_speed[phase_value] = phase_speed[phase_value] 
                        energy = self._phase_calculation(phase_acc[phase_value],phase_speed[phase_value],phase_dist[phase_value],phase_time[phase_value],phase_avg_end_speed[phase_value],slope_angle)
                        phase_energy.append(energy)

                        init_velocity = max_speed_lim
                    elif phase_key == "decelerate":
                        if stop_flag or traffic_stop:
                            phase_time[phase_value] = (max_speed_lim-end_speed_link)/(2*phase_acc[phase_value])
                            phase_speed[phase_value] = end_speed_link
                            phase_avg_end_speed[phase_value] = (0.25*phase_speed[phase_value]) + (0.75*max_speed_lim)
                            energy = self._phase_calculation(phase_acc[phase_value],phase_speed[phase_value],phase_dist[phase_value],phase_time[phase_value],phase_avg_end_speed[phase_value],slope_angle)
                            phase_energy.apped(energy)
                            init_velocity = end_speed_link
                        else:
                            phase_acc[phase_value] = 0
                            phase_dist[phase_value] = 0 
                            phase_time[phase_value] = 0 
                            phase_speed[phase_value] = 0 
                            phase_avg_end_speed[phase_value] = 0
                            phase_energy.append(0)
                    elif phase_key == "stopstill":
                        if stop_flag : 
                            phase_acc[phase_value] = 0 
                            phase_dist[phase_value] = 0
                            phase_time[phase_value] = self._stop_times["stop"]
                            phase_speed[phase_value] = 0 
                            phase_avg_end_speed[phase_value] = 0
                            energy = self._phase_calculation(phase_acc[phase_value],phase_speed[phase_value],phase_dist[phase_value], phase_time[phase_value],phase_avg_end_speed[phase_value],slope_angle)
                            phase_energy.append(energy)
                            init_velocity = 0 
                        elif traffic_stop : 
                            phase_acc[phase_value] = 0 
                            phase_dist[phase_value] = 0
                            phase_time[phase_value] = self._stop_times["stop"]
                            phase_speed[phase_value] = 0 
                            phase_avg_end_speed[phase_value] = 0
                            energy = self._phase_calculation(phase_acc[phase_value],phase_speed[phase_value],phase_dist[phase_value], phase_time[phase_value],phase_avg_end_speed[phase_value],slope_angle)
                            phase_energy.append(energy)
                            init_velocity = 0 
                        else: 
                            phase_acc[phase_value] = 0
                            phase_dist[phase_value] = 0 
                            phase_time[phase_value] = 0 
                            phase_speed[phase_value] = 0 
                            phase_avg_end_speed[phase_value] = 0
                            phase_energy.append(0)

                        tmp = phase_energy[0] + phase_energy[1] + phase_energy[2] + phase_energy[3] 
                        self.total_energy_link.append(tmp)
                        phase_energy.clear() 
                end_speed_prev = end_speed_link   

            rec = 0 
            for eelink in self.total_energy_link:
                rec = rec + eelink
            self._energy_cost_routes.append(rec)
            self._possible_routes.pop(0)
        self._vehicle.set_location(self._origin)
        return self._energy_cost_routes
                