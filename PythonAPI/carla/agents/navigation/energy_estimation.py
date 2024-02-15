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
import random
#Κλάση το ενεργειακό μοντέλο
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
        self._world = self._vehicle.get_world()
        self._possible_routes = possible_routes
        self._origin = origin
        self._min_acc = 0
        self._min_dist = 0
        speed_30 = 30 * 0.27778
        speed_60 = 60 * 0.27778 
        speed_90 = 90 * 0.27778
        speed_100 = 100 * 0.27778

        #Η επιτάχυνση είναι παραγώμενη από τον χρόνο. 
        #ToDo => Get the acceleration from the the car dynamics and the distance covered. 
        self.time_travel = {(0,speed_30): 2.40, (0,speed_60): 3.90, (0,speed_90): 6.08, (0,speed_100): 6.90, (speed_30,speed_60): 1.28, (speed_30,speed_90): 3.81,
                            (speed_60,speed_90): 1.99, (speed_90,speed_60): 1.03, (speed_90,speed_30): 2.12, (speed_90,0): 2.84, (speed_60,speed_30): 1.28, (speed_60,0): 1.78, (speed_30,0): 1.08}
        self.traffic_lights = [13.80, 13.84, 21.31,34.01,47.64,47.83,47.97] #Traffic Stops in seconds
        self._stop_times = {"traffic_light" : mean(self.traffic_lights), "stop" : 3.2}

        #Από εδώ μπορούμε να πάρουμε όλα τα σημεία που μπορεί να σταματήσει το όχημα μαζί με το όριο ταχύτητας κάθε δρόμου. 
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

                print(f'Traffic light location {tlwp.road_id}')


        #Απλή αρχικοποίηση του waypoint_queue που θα έχει στην συνέχεια όλα τα σημεία από τα οποία θα περάσει το αυτοκίνητο. 
        self.target_waypoint, self.target_road_option = (self._wmap.get_waypoint(origin), RoadOption.LANEFOLLOW)
        self._waypoint_queue = deque(maxlen=10000) 
        self._waypoint_queue.append((self.target_waypoint, self.target_road_option))
        self._road_ids = road_ids
        self._total_distance = [] 
        self._total_travel_time = []
        self._links = []
        print(f"|Stops for the map ==> {self._stop_lms}|")
        print(f"|Speed limits in the map ==> {self._speed_lms}|")

    #Βοηθά όταν έχουμε πολλαπλά μονοπάτια. Αναδημιουργεί την ουρά με τα δεδομένα του μονοπατιού όπως αυτό δίνεται από τον PFA αλγόριθμο
    def _remake_queue(self,current_plan, clean_queue=True):
        if clean_queue:
            #Delete for every new route
            self._waypoint_queue.clear()
        new_plan_length = len(current_plan) + len(self._waypoint_queue)
        if new_plan_length > self._waypoint_queue.maxlen:
            new_waypoint_queue = deque(max_len=new_plan_length)
            for wp in self._waypoint_queue:
                new_waypoint_queue.append(wp)
            self._waypoint_queue = new_waypoint_queue
        #Pass the route elements into the queue
        for elem in current_plan:
            self._waypoint_queue.append(elem)

    #Υπολογισμός της επιτάχυνσης (μέσω του χρόνου, όπως αυτός υπολογίστηκε στο περιβάλλον)
    def _average_acceleration(self, v_initial=0, v_final=27.78, time_travel=5.7, dist_travel = 0):
        #v_initial = v_initial*0.27778
        #v_final = v_final*0.27778 #turn Km/h to m/s for this link
        if time_travel is None and dist_travel!=0:
            time_travel = (2*dist_travel)/(v_final+v_initial)
        else:
            s = (1/2)*(v_final+v_initial) * time_travel 
        return (v_final-v_initial)/time_travel

    #Υπολογισμός της ενέργειας που θα καταναλωθεί σε κάθε φάση ενός στάδιου. 
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

    #Εύκολος τρόπος για να πάρουμε τα ids για τους δρόμους των σημείων
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
        

    #Όταν έχουμε πολλαπλά μονοπάτια που γυρίζουν από τους αλγόριθμους τότε θα πρέπει να τα περάσουμε όλα και να δείξουμε τις διαφορές τους. 
    def loop_route(self):
        #Η συνάρτηση περιλαμβάνει και σχηματική αναπαράσταση των μονοπατιών σε δισδιάστατο διάγραμμα για κάθε μονοπάτι. 
        
        if not self._links:
            print("Error when converting the links. Exit program. ")
            return 
        
        #flag = False
        ind = True

        #Δημιούργησε ξεκάθαρες ακμές για τα σημεία, ώστε να είναι περισσότερο ξεκάθαρες οι φάσεις στον υπολογισμό και 
        #ελάττωσε τα προβλήματα όταν οι αποστάσεις είναι πολύ μικρές. 
        flinks = []
        fnal = self._links[-1][-1]
        print(len(self._links))
        for link in range(len(self._links)-2):
            link1 = self._links[link]
            if not link1[0] or not link1[-1]:
                continue
            link2 = self._links[link+1]
            link3 = self._links[link+2]
            p1 = link1[0].transform.location #Αρχικός κόμβος της ακμής
            p2 = link1[-1].transform.location #Τελικός κόμβος της ακμής 
            p3 = link2[0].transform.location
            p4 = link2[-1].transform.location
            d1 = p1.distance(p2) #Απόσταση για την πρώτη ακμή
            d2 = p3.distance(p4) #Απόσταση για την δεύτερη ακμή
            
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

        #Τρόπος για να μετατρέψουμε τα δεδομένα σε γράφο
        #G = nx.DiGraph()
        #for link in flinks: G.add_edge(link[0],link[-1])
        #links = list(nx.topological_sort(G))
        #flinks = []
        #flinks = [(links[link],links[link+1]) for link in range(len(links)-1)]
        #del G
        #links.clear()
        
        # Διάγραμμα με τις αλλαγές που προκύπτουν από την παραπάνω διαδικασία
        for link in range(len(flinks)): 
            x1, y1 = flinks[link][0].transform.location.x, flinks[link][0].transform.location.y    
            x2, y2 = flinks[link][1].transform.location.x, flinks[link][1].transform.location.y 
            plt.plot([-x1,-x2], [y1,y2], marker = 'o')
        plt.title("This is inside the loop function")
        plt.show()
            
        self._links = [(x,y) for x,y in self._links if y is not None and x is not None]

        self._links = flinks + self._links
        #if flag: self.loop_route() #Επιλογή αναδρομής (Νοt fully functional)

        return flinks
    
    #Έλεγχος αν τα σημεία της ακμής είναι πολύ κοντά
    def condition(self, link):
        p1 = link[0].transform.location
        p2 = link[-1].transform.location
        if p1.distance(p2) == 0 or p1.distance(p2)<0.1: 
            return False
        return True

    #Δημιουργία ακμής
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
            
        #also try the douglas peucker way 
        formatted_wp_roads = {}
        for key in wp_road_ids.keys():

            length = len(wp_road_ids[key])
            if length == 1: 
                points = [wp_road_ids[key][0]]
            else:

                include_points = random.sample(range(1, length-1), int(np.ceil(length/2)))
                include_points.append(0)
                include_points.append(length-1)
                include_points = sorted(include_points)
                points = [wp_road_ids[key][ii] for ii in include_points]

            formatted_wp_roads[key] = self._douglas_pucker(points, epsilon_curved=0.25, epsilon_straight=3)


        for key in formatted_wp_roads.keys():
            for ii in range(len(formatted_wp_roads[key])-1):

                wayp1 = formatted_wp_roads[key][ii]
                # wayp1 = carla.Location(x=wayp1[0], y=wayp1[1], z=wayp1[2])
                wayp2 = formatted_wp_roads[key][ii+1]
                # wayp2 = carla.Location(x=wayp2[0], y=wayp2[1], z=wayp2[2])
                # wayp1 = self._wmap.get_waypoint(wayp1)
                # wayp2 = self._wmap.get_waypoint(wayp2)
                self._links.append((wayp1,wayp2))
            # wayp1 = formatted_wp_roads[key][0]
            # wayp1 = carla.Location(x=wayp1[0], y=wayp1[1], z=wayp1[2])
            # wayp2 = formatted_wp_roads[key][-1]
            # wayp2 = carla.Location(x=wayp2[0], y=wayp2[1], z=wayp2[2])
            # wayp1 = self._wmap.get_waypoint(wayp1)
            # wayp2 = self._wmap.get_waypoint(wayp2)
            # self._links.append((wayp1,wayp2)) 


        #Create the list of waypoints only.
        # for key in wp_road_ids.keys(): 
        #     middle_pack = int(len(wp_road_ids[key])/2)
        #     self._links.append((wp_road_ids[key][0], wp_road_ids[key][middle_pack], wp_road_ids[key][-1]))
        
        for link in self._links: 
            x1, y1 = link[0].transform.location.x, link[0].transform.location.y    
            x2, y2 = link[1].transform.location.x, link[1].transform.location.y 
            # x3, y3 = link[2].transform.location.x, link[2].transform.location.y 
            plt.plot([-x1,-x2], [y1,y2], marker = 'o')
        plt.title("This is the one straingth after the appendind")
        plt.show()


        links = []
        #Find and create a path with all the points
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

        for link in links: 
            x1, y1 = link[0].transform.location.x, link[0].transform.location.y    
            x2, y2 = link[1].transform.location.x, link[1].transform.location.y 
            plt.plot([-x1,-x2], [y1,y2], marker = 'o')
        plt.title("This is the one straingth after Seperattions of links")
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
        incline_angle_rad = np.arccos(cosine_angle)
        incline_angle_degree = np.degrees(incline_angle_rad)
        return incline_angle_degree


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

        segment_classification = self.key_points_of_curvature(points ,np.pi/30)
       
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
                rec_result1 = self._douglas_pucker(points[:index+1], epsilon)
                rec_result2 = self._douglas_pucker(points[index:], epsilon)
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
                

        # key_points = self.key_points_of_curvature(points ,np.pi/18)
        
        # for ii in range(1, len(points)-1):
        #     if points[ii] in key_points:
        #         continue
        #     current_point = points[ii].transform.location.x, points[ii].transform.location.y,points[ii].transform.location.z,
            
        #     d = self.point_line_distance_3d(current_point,start_point,end_point)
        #     if d > dmax: 
        #         index = ii
        #         dmax = d 
        #     print(f'Distance {d}')
        # if dmax > epsilon: 
        #     rec_result1 = self._douglas_pucker(points[:index+1], epsilon)
        #     rec_result2 = self._douglas_pucker(points[index:], epsilon)

        #     results = rec_result1[:-1] + rec_result2
        # else:
        #     results = [start_point] + [[p.transform.location.x,p.transform.location.y,p.transform.location.z] for p in key_points if p in points[1:-1]] + [end_point]        
        # return results 



    #Υπολογισμός ενέργειας
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
        #wp_road_ids = self.waypoints_roads()
        if not self._speed_lms: 
            print("No speed limit sign found. Due to urban environment the speed limit is set to 30 km/h")
            speed_flag = True
        else: 
            speed_flag = False

        while len(self._possible_routes) != 0:

            route = self._possible_routes[0]

            #Διαγραμματική αποτύπωση της διαδρομής. 
            # for link in range(len(route)-1): 
            #     x1, y1 = route[link][0].transform.location.x, route[link][0].transform.location.y    
            #     x2, y2 = route[link+1][0].transform.location.x, route[link+1][0].transform.location.y 
            #     # print(f"Distance of link {link} is {route[link][0].transform.location.distance(route[link+1][0].transform.location)}")
            #     plt.plot([-x1,-x2], [y1,y2], marker = 'o')
            # plt.title("Original Path as generated by the PFA")
            # plt.show()

            self._remake_queue(route)
            self._min_acc = self._average_acceleration(0,30*0.27778,self.time_travel[(0,30*0.27778)],dist_travel=0)
            self._min_dist = (1/2)*(30*0.27778) * self.time_travel[(0,30*0.27778)] #distance travelled to reach 30km/h
            links = self.link_creation()

            #Τελική αποτύπωση της διαδρομής μετά τον υπολογισμό και την μείωση των ακμών. 
            for link in range(len(links)): 
                cur_link = links[link]
                x1, y1 = cur_link[0].transform.location.x, cur_link[0].transform.location.y    
                x2, y2 = cur_link[-1].transform.location.x, cur_link[-1].transform.location.y 
                print(cur_link[0].transform.location.distance(cur_link[-1].transform.location))
                plt.plot([-x1,-x2], [y1,y2], marker = 'o')
            plt.title("Final Processed Path")
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
                next_waypoint = links[wps][-1]
                next_link_wp = links[wps+1][-1]

                if speed_flag:
                    self._speed_lms[waypoint.road_id] = 30*0.27778
                    self._speed_lms[next_waypoint.road_id] = 30*0.27778
                    self._speed_lms[next_link_wp.road_id] = 30*0.27778


                if not waypoint or not next_waypoint:
                    warnings.warn(F"At current iteration {wps} no waypoint was found. \nCheck the Start and Finish point locations. ")
                    break

                if next_waypoint.road_id!=waypoint.road_id:
                    # not_stop_wp = False
                    if self._highway_lms.get(next_waypoint.road_id):
                        highway = True
            
                current_loc = waypoint.transform.location
                #print(current_loc)
                next_location = next_waypoint.transform.location
                print("Type of the vehicle object is " , type(self._vehicle))

                # self._vehicle.set_transform(waypoint.transform)
                print(self._vehicle.get_velocity())
                # if (next_waypoint==waypoint) or (current_loc.x == next_location.x and current_loc.y == next_location.y):
                #     continue

                # incline = math.atan2((next_location.y-current_loc.y),(next_location.x-current_loc.x))
                # slope_angle = math.degrees(incline)
                # print(f' Slope angle {slope_angle}')

                cur_loc = [current_loc.x, current_loc.y, current_loc.z]
                next_loc = [next_location.x, next_location.y, next_location.z]
                slope_angle = self.calculate_incline_angle(cur_loc, next_loc)
                # print(f'Incline angle {self.calculate_incline_angle(cur_loc, next_loc)}')

                link_dist = current_loc.distance(next_location) #The same reuslt as the below equation. 
                # (np.sqrt((next_location.y-current_loc.y)**2 + (next_location.x-current_loc.x)**2 + (next_location.z-current_loc.z)**2))

                total_distance += link_dist

                #Παίρνουμε την μέγιστη ταχύτητα της ακμής που αλλάζει και την τρέχουσα ταχύτητα. 
                if next_waypoint.road_id == waypoint.road_id: #Αν το επόμενο σημείο ανήκει στον ίδιο δρόμο τότε 
                    v_max_current = self._speed_lms.get(waypoint.road_id) #Η μέγιστη ταχύτητα είναι το όριο ταχύτητα του παρόν δρόμου. 
                    print(f'V_max_current 1 {v_max_current}')
                    if next_waypoint.road_id == next_link_wp.road_id: #Αν το μέθεπόμενο σημείο ανήκει και αυτό στον ίδιο δρόμο με το επόμενο
                        v_max_next = v_max_current #Η μέγιστη ταχύτητα του επόμενου δρόμου θα είναι ίση με την μέγιστη του παρόν δρόμου. 
                    else:
                        v_max_next = self._speed_lms.get(next_link_wp.road_id) #Αλλιώς πάρε την μέγιστη ταχύτητα του επόμενου δρόμου με βάση τις τιμές των speed limits. 
                else: #Αν η το επόμενο σημείο ανήκει σε άλλον δρόμο, ελέγχουμε το όριο τοαχύτητας εκείνου του δρόμου. 
                    v_max_current = self._speed_lms.get(next_waypoint.road_id) 
                    if next_waypoint.road_id == next_link_wp.road_id: #Αν το μέθεπόμενο σημείο ανήκει στον ίδιο δρόμο με το επόμενο
                        v_max_next = v_max_current #Η μέγιστη ταχύτητα του επόμενου θα είναι η ίση με την παρούσα. 
                    else: 
                        v_max_next = self._speed_lms.get(next_link_wp.road_id) #Παίρνουμε την μέγιστη ταχύτητα με βάση τον μέθεπόμενο δρόμο. 
                
                
                if not v_max_current: #Αν δεν υπάρχει μέγιστη ταχύτητα (κάτι πήγε λάθος)
                    v_max_current = 30 *0.27778
                # else: 
                #     v_max_current = v_max_current.value

                if not v_max_next:
                    v_max_next = 30*0.27778
                # else: 
                #     v_max_next = v_max_next.value
                

                #Ελέγχουμε για στάσεις κατά την διαδρομή ανάλογα με τους δρόμους που έχουν επιλεγεί για να ακολουθήσει το όχημα. 
                #Πρώτα θέλουμε να ελέγχουμε την επόμενη διασταύρωση για τυχόν φανάρια. 
                # if self._vehicle.get_traffic_light(): #Παίρνουμε το φανάρι που ανήκει στον δρόμο που είναι το αυτοκίνητο. 
                #     print("Traffic light foungd in road :::: ", waypoint.road_id )
                if next_waypoint.get_junction():
                    print("Traffic light foungd in road :::: ", self._vehicle.get_traffic_light())
                    if self._vehicle.is_at_traffic_light() and not not_stop_wp: # and not not_stop_wp:
                        print(f"Traffic light spotted at {next_waypoint.road_id}")
                        traffic_stop = True 
                        not_stop_wp = True 
                        stop_flag = False 

                stop_link = self._stop_lms.get(next_waypoint.id)
                if stop_link and not not_stop_wp: #and not_stop_wp==False :
                    stop_flag = True
                    not_stop_wp = True
                    traffic_stop = False
                elif wps + 1 == (len(links)-1):
                    print(f"Stop sign spotted at {next_waypoint.road_id}")
                    stop_flag = True
                    not_stop_wp = True
                    traffic_stop = False    
                
                if stop_flag or traffic_stop: #Αν υπάρχει οποιαδήποτε στάση
                    v_end_current = 0 #στο τέλος της παρούσας φάσης πρέπει η ταχύτητα να γίνει μηδέν (φρενάρισμα)
                else: #Αλλιώς θα πρέπει να ελέγξουμε την ταχύτητα στην επόμενη στάση με βάση την προηγούμενη διεργασία
                    if v_max_next >= v_max_current: #Αν η ταχύτητα του επόμενου δρόμου είναι μεγαλύτερη από την παρούσα
                        v_end_current = v_max_current # Η ταχύτητα στο τέλος της φάσης πρέπει να γίνει η επόμενη ώστε να μην υπάρξει επιτάχυνση
                    else: 
                        v_end_current = v_max_next # Η ταχύτητα στο τέλος της φάσης πρέπει να γίνει η μεθεπόμενη ώστε να υπάρξει επιτάχυνση
                
                init_velocity = v_end_prev #Η αρχική ταχύτητα για την πρώτη φάση είναι η ίση με την τελική ταχύτητα των προηγούμενων φάσεων. 
                v_end_prev = v_end_current 
                
                print(f" Initial_speed {init_velocity} and Max speed of link {v_max_current} with end speed {v_end_current}")
                print(f"Velocity of car {self._vehicle.get_velocity()}")
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
                            print("Decreasing speed as stop node was found...")
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
                
