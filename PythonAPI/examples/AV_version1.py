import glob 
import os 
import sys 
import random
import time
import argparse
import cv2


#Try and import numpy and what other package we are unsure will load correctly
try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

#Get the installation file path so that the program knows what carla version to execute upon
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

##Extremely necessary to be able to get the libraries we want from the carla repository
#Fixed a problem where the program could not find the agents directory under carla in PythonAPI
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from agents.navigation.Basic_RP_agent import Agent
from agents.navigation.Vehicle_design import VehicleDesign,PhysicalControls,VehicleCoefficients, VehicleParts

#Image specs this can be good for future projects. 
IM_WIDTH = 640
IM_HEIGHT = 480

#Proces the images and save them to disk 
def process_image(image):
    cc = carla.ColorConverter.LogarithmicDepth
    image.save_to_disk('_out/%06d.png' % image.frame, cc)
    i = np.array(image.raw_data)

    #the way the images come from carla are flatten(all bits are in one dimension)
    i2 = i.reshape((IM_HEIGHT,IM_WIDTH,4)) #The four is the 3 rgb and a for alpha information
    i3 = i2[:,:, :3] #Get the first three elements so take the height, the width and the rgb values 
    #cv2.imshow("",i3)
    #cv2.waitKey(1)
    return i3/255.0 #Normalize data to pass them to a neural network 

def main():

    #always create an actor's list to orderly destroy the actors after finalizing the program. 
    actor_list = []

    #attempt to connect to server and after everything destroy the clients / actors 
    try: 
        #This is to pass arguments in terminal in order to change the connection with the server. 
        argparser = argparse.ArgumentParser(
            description=__doc__)
        argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        argparser.add_argument(
                '--verbose', 
                default = False, 
                help='Select if you want to show the plots for the route establishment'
        )
        argparser.add_argument(
            '--osm-path',
            metavar='OSM_FILE_PATH',
            help='load a new map with a minimum physical road representation of the provided OpenStreetMaps',
            type=str)
        argparser.add_argument(
            '-s', '--seed',
            help='Set seed for repeating executions (default: None)',
            default=None,
            type=int)
        argparser.add_argument(
            '--sync',
            action='store_true',
            default = False,
            help='Synchronous mode execution')
        argparser.add_argument(
            '--json', 
            default=None,
            help='Provide absolute path of the json file')
        if len(sys.argv) < 1:
            argparser.print_help()
            return

        args = argparser.parse_args()

        #Get client connected with the host based on the above configuration
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        #use sim_world instead of world. 
        sim_world = client.get_world()

        #Get the world which is the environment (we can of course change that)
        map = sim_world.get_map()
        
        #We can get a traffic manager from the client. 
        traffic_manager = client.get_trafficmanager()
        
        if args.sync:
            settings = sim_world.get_settings() 
            delta = 0.5 #How many steps are in a second. so if delta = 0.05 it takes 20 steps for a second. 
            settings.fixed_delta_seconds = delta 
            settings.synchronous_mode = True
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True) 


        #Get spawn points and generate waypoints to be used later for the routeplanner. 
        spawn_points = map.get_spawn_points()
        road_waypoints = map.generate_waypoints(distance = 10.0)
        st_point = carla.Location(spawn_points[50].location)
        fin_point = carla.Location(spawn_points[100].location)
        #landmarks = sim_world.get_map().get_all_landmarks_of_type('274') #'1000001' return the signal lights. 
        
        #Actual spawning of the vehicle 
        #spawn_point_wp = map.get_waypoint(location)

        #Pick a car based on a list given in the documentation
        #Get the blueprint library which is used to give characteristics to the sensors and vehicles. 
        blueprint_library = sim_world.get_blueprint_library()
        bp = blueprint_library.filter("etron")[0] #Tesla Model 3
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color',color)
        bp.set_attribute('role_name','hero')

        #location = carla.Location(x=60.2, y = 123.4, z=0)
        sp_wp = map.get_waypoint(st_point)
        start_wp = sp_wp.transform.location 
        sp = carla.Transform(carla.Location(x = start_wp.x, y = start_wp.y, z = start_wp.z),sp_wp.transform.rotation)
        sp.location.z += 2.5 

        vehicle = sim_world.spawn_actor(bp,sp)
        print("Type of the vehicle object is " , type(vehicle))
        actor_list.append(vehicle)
        print('Created %s' % vehicle.type_id)


        # destination_location = carla.Location(x=-23.6,y=40,z=0)
        # dest_wp = map.get_waypoint(destination_location)
        # wp = dest_wp.transform.location
        # dest = carla.Transform(carla.Location(x=wp.x, y=wp.y, z = wp.z),dest_wp.transform.rotation)
        # dest.location.z +=2

        #We are able to configure vehicle's physics as below: 
        front_left_wheel = carla.WheelPhysicsControl(tire_friction = 2.0, damping_rate = 1.5, max_steer_angle = 70.0,long_stiff_value=1000)
        front_right_wheel = carla.WheelPhysicsControl(tire_friction = 2.0, damping_rate = 1.5, max_steer_angle = 70.0,long_stiff_value=1000)
        rear_left_wheel = carla.WheelPhysicsControl(tire_friction = 3.0, damping_rate = 1.5, max_steer_angle = 0.0,long_stiff_value=1000)
        rear_right_wheel = carla.WheelPhysicsControl(tire_friction = 3.0, damping_rate = 1.5, max_steer_angle = 0.0,long_stiff_value=1000)
        
        wheels = [front_left_wheel,front_right_wheel,rear_left_wheel,rear_right_wheel]

        physics_control = vehicle.get_physics_control()
        physics_control.torque_curve = [carla.Vector2D(x=0,y=325),
                                        carla.Vector2D(x=2000,y=325),
                                        carla.Vector2D(x=4000,y=325),
                                        carla.Vector2D(x=6000,y=255),
                                        carla.Vector2D(x=8000,y=177),
                                        carla.Vector2D(x=10000,y=120),
                                        carla.Vector2D(x=12000,y=77.5)]
        physics_control.max_rmp = 18000
        physics_control.moi = 1.0 
        physics_control.damping_rate_full_throttle = 0.0
        physics_control.use_gear_autobox = True 
        physics_control.gear_switch_time = 0.5
        physics_control.clutch_strength= 10
        physics_control.mass = 2490
        physics_control.drag_coefficient = 0.27 
        physics_control.steering_curve = [carla.Vector2D(x=0,y=1),carla.Vector2D(x=100,y=1), carla.Vector2D(x=300,y=1)]
        physics_control.use_sweep_wheel_collision = True 
        physics_control.wheels = wheels 

        vehicle.apply_physics_control(physics_control)
        #print(physics_control)
        #***************************************************#
        velocities = {'speed_10' : 10 ,
                      'speed_20' : 20 , 
                      'speed_30' : 30 ,
                      'speed_40' : 40 , 
                      'speed_50' : 50 , 
                      'speed_60' : 60 ,
                      'speed_70' : 70 ,
                      'speed_80' : 80 , 
                      'speed_90' : 90 ,
                      'speed_100' : 100} 

        time_travel = {(0,velocities['speed_10']): 1.41, (velocities['speed_10'],0): 0.49,
                       (0,velocities['speed_20']): 2.05, (velocities['speed_20'],0): 0.75,
                       (0,velocities['speed_30']): 2.40, (velocities['speed_30'],0): 1.08,
                       (0,velocities['speed_40']): 3.03, (velocities['speed_40'],0): 1.22, 
                       (0,velocities['speed_50']): 3.50, (velocities['speed_50'],0): 1.49,
                       (0,velocities['speed_60']): 3.90, (velocities['speed_60'],0): 1.78, 
                       (0,velocities['speed_70']): 4.58, (velocities['speed_70'],0): 2.07,
                       (0,velocities['speed_80']): 5.35, (velocities['speed_80'],0): 2.48,
                       (0,velocities['speed_90']): 6.08, (velocities['speed_90'],0): 2.84,
                       (0,velocities['speed_100']): 6.90, (velocities['speed_100'],0): 3.05,
                       (velocities['speed_10'],velocities['speed_20']): 0.88, (velocities['speed_20'],velocities['speed_10']): 0.41,
                       (velocities['speed_10'],velocities['speed_30']): 1.53, (velocities['speed_30'],velocities['speed_10']): 0.74,
                       (velocities['speed_10'],velocities['speed_40']): 2.12, (velocities['speed_40'],velocities['speed_10']): 0.98,
                       (velocities['speed_10'],velocities['speed_50']): 2.57, (velocities['speed_50'],velocities['speed_10']): 1.17,
                       (velocities['speed_10'],velocities['speed_60']): 2.89, (velocities['speed_60'],velocities['speed_10']): 1.30,
                       (velocities['speed_10'],velocities['speed_70']): 3.51, (velocities['speed_70'],velocities['speed_10']): 1.76,
                       (velocities['speed_10'],velocities['speed_80']): 3.82, (velocities['speed_80'],velocities['speed_10']): 2.11,
                       (velocities['speed_10'],velocities['speed_90']): 4.60, (velocities['speed_90'],velocities['speed_10']): 2.45,
                       (velocities['speed_10'],velocities['speed_100']): 5.59,(velocities['speed_100'],velocities['speed_10']): 2.89,
                       (velocities['speed_20'],velocities['speed_30']): 0.53, (velocities['speed_30'],velocities['speed_20']): 0.43,
                       (velocities['speed_20'],velocities['speed_40']): 1.26, (velocities['speed_40'],velocities['speed_20']): 0.87,
                       (velocities['speed_20'],velocities['speed_50']): 1.85, (velocities['speed_50'],velocities['speed_20']): 1.05,
                       (velocities['speed_20'],velocities['speed_60']): 2.42, (velocities['speed_60'],velocities['speed_20']): 1.36,
                       (velocities['speed_20'],velocities['speed_70']): 2.80, (velocities['speed_70'],velocities['speed_20']): 1.72,
                       (velocities['speed_20'],velocities['speed_80']): 3.39, (velocities['speed_80'],velocities['speed_20']): 2.01,
                       (velocities['speed_20'],velocities['speed_90']): 3.92, (velocities['speed_90'],velocities['speed_20']): 2.31,
                       (velocities['speed_20'],velocities['speed_100']): 4.87, (velocities['speed_100'],velocities['speed_20']): 2.60,
                       (velocities['speed_30'],velocities['speed_40']): 0.79, (velocities['speed_40'],velocities['speed_30']): 0.60,
                       (velocities['speed_30'],velocities['speed_50']): 1.37, (velocities['speed_50'],velocities['speed_30']): 0.98,
                       (velocities['speed_30'],velocities['speed_60']): 1.87, (velocities['speed_60'],velocities['speed_30']): 1.28,
                       (velocities['speed_30'],velocities['speed_70']): 2.26, (velocities['speed_70'],velocities['speed_30']): 1.49,
                       (velocities['speed_30'],velocities['speed_80']): 2.72, (velocities['speed_80'],velocities['speed_30']): 1.84,
                       (velocities['speed_30'],velocities['speed_90']): 3.81, (velocities['speed_90'],velocities['speed_30']): 2.12,
                       (velocities['speed_30'],velocities['speed_100']): 4.43, (velocities['speed_100'],velocities['speed_30']): 2.48,
                       (velocities['speed_40'],velocities['speed_50']): 0.84, (velocities['speed_50'],velocities['speed_40']): 0.55,
                       (velocities['speed_40'],velocities['speed_60']): 1.57, (velocities['speed_60'],velocities['speed_40']): 0.88,
                       (velocities['speed_40'],velocities['speed_70']): 1.96, (velocities['speed_70'],velocities['speed_40']): 1.10,
                       (velocities['speed_40'],velocities['speed_80']): 2.48, (velocities['speed_80'],velocities['speed_40']): 1.42,
                       (velocities['speed_40'],velocities['speed_90']): 3.10, (velocities['speed_90'],velocities['speed_40']): 1.79,
                       (velocities['speed_40'],velocities['speed_100']): 3.83, (velocities['speed_100'],velocities['speed_40']): 2.07,
                       (velocities['speed_50'],velocities['speed_60']): 0.69, (velocities['speed_60'],velocities['speed_50']): 0.44,
                       (velocities['speed_50'],velocities['speed_70']): 1.38, (velocities['speed_70'],velocities['speed_50']): 0.69,
                       (velocities['speed_50'],velocities['speed_80']): 1.92, (velocities['speed_80'],velocities['speed_50']): 0.96,
                       (velocities['speed_50'],velocities['speed_90']): 2.66, (velocities['speed_90'],velocities['speed_50']): 1.22,
                       (velocities['speed_50'],velocities['speed_100']): 3.43, (velocities['speed_100'],velocities['speed_50']): 1.64,
                       (velocities['speed_60'],velocities['speed_70']): 0.78, (velocities['speed_70'],velocities['speed_60']): 0.48,
                       (velocities['speed_60'],velocities['speed_80']): 1.34, (velocities['speed_80'],velocities['speed_60']): 0.77,
                       (velocities['speed_60'],velocities['speed_90']): 1.99, (velocities['speed_90'],velocities['speed_60']): 1.03,
                       (velocities['speed_60'],velocities['speed_100']): 2.63, (velocities['speed_100'],velocities['speed_60']): 1.2,
                       (velocities['speed_70'],velocities['speed_80']): 0.69, (velocities['speed_80'],velocities['speed_70']): 0.50,
                       (velocities['speed_70'],velocities['speed_90']): 1.41, (velocities['speed_90'],velocities['speed_70']): 0.69,
                       (velocities['speed_70'],velocities['speed_100']): 1.93, (velocities['speed_100'],velocities['speed_70']): 0.99,
                       (velocities['speed_80'],velocities['speed_90']): 0.78, (velocities['speed_90'],velocities['speed_80']): 0.40,
                       (velocities['speed_80'],velocities['speed_100']): 1.20, (velocities['speed_100'],velocities['speed_80']): 0.78,
                       (velocities['speed_90'],velocities['speed_100']): 0.68, (velocities['speed_100'],velocities['speed_90']): 0.45}


        vehicle_coeffs = VehicleCoefficients(A_coef=35.745,B_coef=0.38704,C_coef=0.018042,
                                               d_coef=1.1,P_aux=520, velocities=velocities, time_travel=time_travel)
        veh_parts = VehicleParts(vehicle)
        ph_objects = PhysicalControls(physics_control)
        vehicle_design = VehicleDesign(model="etron",
                                       company="Audi",
                                       physical_controls=ph_objects,
                                       vehicle_parts=veh_parts,
                                       vehicle_coefficients=vehicle_coeffs)
        print("Printing the vehicle desing in JSON form ")
        print(vehicle_design.to_dict())
        vehicle_design.save_to_json_file("vehicle.json"); 
        print("Saveing everything to a jason file...")
        time.sleep(60)

        #****************************************************#
        #To have a sort of self-driving car 
        #vehicle.set_autopilot(True)
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0.0,brake = 0.0 , hand_brake=False, reverse = False, manual_gear_shift = False))
        oct_dictionary = {'ignore_traffic_lights':False,
                          'ignore_stop_signs': False,
                          'ignore_vehicles': False}
        agent = Agent(vehicle=vehicle, vehicle_coeffs=vehicle_coeffs,target_speed=30,opt_dict=oct_dictionary,verbose=args.verbose) #The Basic RP planner will handle the control of the vehicle. 
        agent.follow_speed_limits(True)
        agent.ignore_stop_signs(False)
        #agent.set_destination(fin_point.location,st_point.location)
        agent.set_destination(fin_point,st_point)
        route_planner_agent = agent.get_route_planner() #Get the RoutePlanner object

        #Attach a sensor to a car 
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        cam_bp.set_attribute("fov", "110")
        cam_transform = carla.Transform((carla.Location(x=-4,z=2.5)))#)
        camera = sim_world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        spectator = sim_world.get_spectator()
        spectator.set_transform(camera.get_transform())
        actor_list.append(camera)

        #For different sensors different attributes. 
        spawn_cam_point = carla.Transform(carla.Location(x=2.5, z=0.7))#Relative position depending on the vehicle.
        sensor = sim_world.spawn_actor(cam_bp,spawn_cam_point,attach_to=vehicle)
        actor_list.append(sensor)
        print('Created %s' % sensor.type_id)

        # for tl in sim_world.get_actors().filter('traffic.traffic_light*'):
        #     # Trigger/bounding boxes contain half extent and offset relative to the actor.
        #     trigger_transform = tl.get_transform()
        #     trigger_transform.location += tl.trigger_volume.location
        #     trigger_extent = tl.trigger_volume.extent


        #how to get the data from sensor. 
        sensor.listen(lambda data: process_image(data))

        #This is to generate the npc and other vehicles giving life to the world. 
        transform = random.choice(spawn_points)
        transform.location += carla.Location(x=40,y=-3.2)
        transform.rotation.yaw = -180.0
        for _ in range(0,10):
            transform.location.x += 8.0 
            bp = random.choice(blueprint_library.filter('vehicle'))
            npc = sim_world.try_spawn_actor(bp,transform)
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                print('Created %s' % npc.type_id)

        #vehicle.apply_control(agent.run_step()) #
        time = 0 
        while True:
            time = time + 1         
            spectator.set_transform(camera.get_transform())







            #vehicle.apply_control(agent.run_step())
            if agent.done(): 
                #If we do not wish to finish the simulation when the destination is reached 
                #we can create an new random routeas: 
                #agent.set_destination(random.choice(spawn_points).location)
                print("The target has been reached, stopping the simulation")
                break
            control = agent.run_step()
            control.manual_gear_shift = False 
            vehicle.apply_control(control)
            #print(f" Vehicle Max speed {vehicle.get_speed_limit()}")

        print(f"Simulation Ended after {time} steps")

    finally: 
        #settings = sim_world.get_settings()
        #settings.synchronous_mode = False 
        #settings.fixed_delta_seconds = None 
        #sim_world.apply_setting(settings) 
        #traffic_manager.set_synchronous_mode(True)

        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print("All clients are cleaned")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except RuntimeError as e:
        print(e)
