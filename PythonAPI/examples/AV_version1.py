import glob 
import os 
import sys 
import random
import time
import numpy as np 
import cv2 
import argparse



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
        #argparser.add_argument(
        #    '-s', '--speed',
        #    metavar='FACTOR',
        #    default=1.0,
        #    type=float,
        #    help='rate at which the weather changes (default: 1.0)')
        argparser.add_argument(
            '-m', '--map',
            help='load a new map, use --list to see available maps')
        argparser.add_argument(
            '-r', '--reload-map',
            action='store_true',
            default=False,
            help='reload current map')
        argparser.add_argument(
            '--osm-path',
            metavar='OSM_FILE_PATH',
            help='load a new map with a minimum physical road representation of the provided OpenStreetMaps')
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
        #waypoints = map.generate_waypoints(distance = 10.0)
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
        location = carla.Location(x=60.2, y = 123.4, z = 0)
        sp_wp = map.get_waypoint(location)
        start_wp = sp_wp.transform.location 
        sp = carla.Transform(carla.Location(x = start_wp.x, y = start_wp.y, z = start_wp.z),sp_wp.transform.rotation)
        sp.location.z += 2 
        vehicle = sim_world.spawn_actor(bp,sp)
        actor_list.append(vehicle)
        print('Created %s' % vehicle.type_id)

        destination_location = carla.Location(x=-139.6,y=-134,z=0)
        dest_wp = map.get_waypoint(destination_location)
        wp = dest_wp.transform.location
        dest = carla.Transform(carla.Location(x=wp.x, y=wp.y, z = wp.z),dest_wp.transform.rotation)
        dest.location.z +=2

        ego_vehicle_speed = vehicle.get_velocity()
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
        physics_control.dampingr_rate_full_throttle = 0.0
        physics_control.use_gear_autobox = True 
        physics_control.gear_switch_time = 0.5
        physics_control.clutch_strength= 10
        physics_control.mass = 2.490
        physics_control.drag_coefficient = 0.27 
        physics_control.steering_curve = [carla.Vector2D(x=0,y=1),carla.Vector2D(x=100,y=1), carla.Vector2D(x=300,y=1)]
        physics_control.use_sweep_wheel_collision = True 
        physics_control.wheels = wheels 

        vehicle.apply_physics_control(physics_control)
        #print(physics_control)

        #To have a sort of self-driving car 
        #vehicle.set_autopilot(True)
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0.0,brake = 0.0 , hand_brake=False, reverse = False, manual_gear_shift = False))
        oct_dictionary = {'ignore_traffic_lights':False,
                          'ignore_stop_signs': False,
                          'ignore_vehicles': False}
        agent = Agent(vehicle,30,opt_dict=oct_dictionary) #The Basic RP planner will handle the control of the vehicle. 
        agent.follow_speed_limits(True)
        agent.ignore_stop_signs(False)
        agent.set_destination(dest.location,sp.location)
        route_planner_agent = agent.get_route_planner() #Get the RoutePlanner object

        #Attach a sensor to a car 
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}" )
        cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}" )
        cam_bp.set_attribute("fov", "110")

        #For different sensors different attributes. 
        spawn_cam_point = carla.Transform(carla.Location(x=2.5, z=0.7))#Relative position depending on the vehicle.
        sensor = sim_world.spawn_actor(cam_bp,spawn_cam_point,attach_to=vehicle)
        actor_list.append(sensor)
        print('Created %s' % sensor.type_id)

        #spectator_transform = carla.Transform(vehicle_transform.location, vehicle_transform.rotation)
        #spectator_transform.location += vehicle_transform.get_forward_vector() * 20
        #spectator_transform.rotation.yaw += 180
        #spectator = world.get_spectator()
        #spectator.set_transform(spectator_transform)


        for tl in sim_world.get_actors().filter('traffic.traffic_light*'):
            # Trigger/bounding boxes contain half extent and offset relative to the actor.
            trigger_transform = tl.get_transform()
            trigger_transform.location += tl.trigger_volume.location
            trigger_extent = tl.trigger_volume.extent


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
        #for actor in actor_list:
        #    actor.destroy()
        #or
        
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
