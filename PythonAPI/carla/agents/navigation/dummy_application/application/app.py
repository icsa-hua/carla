import os 
import glob 
import sys 
import random 
import argparse 
import pdb 
import numpy as np
import warnings
import logging
import carla

try: 
    if "Tkinter" not in sys.modules:
        import tkinter as tk
        from tkinter import *
    from PIL import Image, ImageTk
    import matplotlib.pyplot as plt
except ImportError: 
    raise RuntimeError('Something went wrong.')
 
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')

except IndexError as e:
    warnings.warn(f"Exception caught {e}... ")


import carla 
from agents.navigation.Basic_RP_agent import Agent 
from agents.navigation.dummy_application.dummy_vehicle.utils.Entity import Entity
from agents.navigation.dummy_application.dummy_vehicle.utils.Components import PCI, Parts, Mechanical
from dummy_utils.GUI.GraphicalInterface import WaypointInterface


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 


class Application():

    def __init__(self)->None:
        self.actor_list = [] 
        self.argparser = argparse.ArgumentParser(description=__doc__)
        self.client = None
        self.flag = None

    def get_arguments(self): 
        try : 
            logger.info("Getting the arguments")

            self.argparser.add_argument(
                '--host', 
                metavar='H',
                default='127.0.0.1',
                help='IP of the host server (default: 127.0.0.1)'
            )
            self.argparser.add_argument(
                '-p', '--port',
                metavar='P',
                default=2000,
                type=int,
                help='TCP port to listen to (default: 2000)'
            )
            self.argparser.add_argument(
                '--verbose', 
                default = False, 
                help='Select if you want to show the plots for the route establishment'
            )
            self.argparser.add_argument(
                '--osm-path',
                metavar='OSM_FILE_PATH',
                help='load a new map with a minimum physical road representation of the provided OpenStreetMaps',
                type=str
            )
            self.argparser.add_argument(
                '-s', '--seed',
                help='Set seed for repeating executions (default: None)',
                default=None,
                type=int
            )
            self.argparser.add_argument(
                '--sync',
                action='store_true',
                default = False,
                help='Synchronous mode execution'
            )
            self.argparser.add_argument(
                '--json', 
                default=None,
                help='Provide absolute path of the json file'
            )
            self.argparser.add_argument(
                '--pfa', 
                default = 'A*', 
                help='Select if you want to alernate the environment settings'
            )
            self.argparser.add_argument(
                '--gui',
                default=False, 
                help="Generate Interface",
            )
            
            if len(sys.argv) < 1:
                self.argparser.print_help()
                return
            
            self.args = self.argparser.parse_args()
            
            logger.info("Connecting to localhost...")
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(10.0)
            
            logger.info("Starting Application...")   
            self.verbose = True if self.args.verbose=='True' or self.args.verbose=='true' else False
            self.world = self.client.get_world()
            self.map = self.world.get_map() 
            self.topology = self.map.get_topology()
            self.traffic_manager = self.client.get_trafficmanager()
            
            self.synchronize_host(self.args) 
            
            self.start_point = None 
            self.destination_point = None
            
            if self.args.gui: 
                self.run_app_with_interface()
            else: 
                self.run_app_without_interface()
            
            self.bpl_library = self.world.get_blueprint_library()

            if self.args.json: 
                hero_entity = self.run_app_from_json()
            else: 
                hero_entity = self.run_app_from_topology()
            
            self.set_npc_agents()

            self.run_app(self.set_agent_spectator(hero_entity))

        except Exception as e:
            warnings.warn(f"Exception caught {e}...")

        finally: 
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            logger.info("All clients have been cleared")


    def synchronize_host(self, args): 
        if args.sync:
            settings = self.world.get_settings() 
            delta = 0.5 #How many steps are in a second. so if delta = 0.05 it takes 20 steps for a second. 
            settings.fixed_delta_seconds = delta 
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(True)


    def call_GUI(self): 
                
        directory = os.path.join(os.getcwd(), 'TownImages/')
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        image_path = self.save_if_verbose(directory)
        
        root=tk.Tk()
        app = WaypointInterface(root, self.map,image_path)
        root.mainloop()
        self.start_point = app.start_wp
        self.destination_point = app.end_wp


    def save_if_verbose(self, directory): 
        
        name = self.map.name.split("/")
        town_image = name[-1] + '.jpg'
        
        if self.verbose: 
            for link in range(len(self.topology)):
                cur_link = self.topology[link]
                x1, y1 = cur_link[0].transform.location.x, cur_link[0].transform.location.y    
                x2, y2 = cur_link[-1].transform.location.x, cur_link[-1].transform.location.y 
                plt.plot([-x1,-x2], [y1,y2], marker = 'o', color = 'black')
            plt.title("Final Processed Path")
            plt.show()
            plt.savefig(os.path.join(directory,town_image))
            return ""
        
        return os.path.join(directory, town_image)


    def run_app_with_interface(self): 
        self.call_GUI()
        self.start_point = carla.Location(self.start_point.location)
        self.destination_point = carla.Location(self.destination_point.location)
        

    def run_app_without_interface(self): 
        spawn_points = self.map.get_spawn_points()
        self.start_point = carla.Location(spawn_points[50].location)
        self.destination_point = carla.Location(spawn_points[100].location)


    def set_vehicle(self,model):
        blueprint = self.bpl_library.filter(model)[0]

        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        
        blueprint.set_attribute('role_name','hero')
        sp = self.map.get_waypoint(self.start_point).transform
        sp.location.z += 2.5
        self.vehicle = self.world.spawn_actor(blueprint, sp)
        self.actor_list.append(self.vehicle)
        logger.info("Vehicle instantiated at {}".format(sp))


    def set_wheels(self): 
        front_left_wheel = carla.WheelPhysicsControl(tire_friction = 2.0, damping_rate = 1.5, max_steer_angle = 70.0,long_stiff_value=1000)
        front_right_wheel = carla.WheelPhysicsControl(tire_friction = 2.0, damping_rate = 1.5, max_steer_angle = 70.0,long_stiff_value=1000)
        rear_left_wheel = carla.WheelPhysicsControl(tire_friction = 3.0, damping_rate = 1.5, max_steer_angle = 0.0,long_stiff_value=1000)
        rear_right_wheel = carla.WheelPhysicsControl(tire_friction = 3.0, damping_rate = 1.5, max_steer_angle = 0.0,long_stiff_value=1000)
        
        wheels = [front_left_wheel,front_right_wheel,rear_left_wheel,rear_right_wheel]
        return wheels


    def set_physics_control(self):
        pci = self.vehicle.get_physics_control()
        pci.torque_curve = [carla.Vector2D(x=0,y=325),carla.Vector2D(x=2000,y=325),
                                        carla.Vector2D(x=4000,y=325),carla.Vector2D(x=6000,y=255),
                                        carla.Vector2D(x=8000,y=177),carla.Vector2D(x=10000,y=120),
                                        carla.Vector2D(x=12000,y=77.5)]
        pci.max_rpm = 9000.0
        pci.moi = 1.0 
        pci.damping_rate_full_throttle = 0.0
        pci.use_gear_autobox = True 
        pci.gear_switch_time = 0.5
        pci.clutch_strength= 10
        pci.mass = 2565.0
        pci.drag_coefficient = 0.27 
        pci.steering_curve = [carla.Vector2D(x=0,y=1),carla.Vector2D(x=100,y=1), carla.Vector2D(x=300,y=1)]
        pci.use_sweep_wheel_collision = True 
        pci.wheels = self.set_wheels()
        return pci
    
    
    def set_time_travel(self): 
        velocities = {'speed_5' : 5,'speed_7' : 7 ,
                      'speed_10' : 10,'speed_20' : 20 , 
                      'speed_30' : 30,'speed_40' : 40 , 
                      'speed_50' : 50,'speed_60' : 60 ,
                      'speed_70' : 70,'speed_80' : 80 , 
                      'speed_90' : 90,'speed_100' : 100}
        
        return {(0,velocities['speed_5']): 1.02, (velocities['speed_5'],0): 0.22,
                (0,velocities['speed_7']): 1.20, (velocities['speed_7'],0): 0.37,
                (0,velocities['speed_10']): 1.41, (velocities['speed_10'],0): 0.49,
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
                (velocities['speed_90'],velocities['speed_100']): 0.68, (velocities['speed_100'],velocities['speed_90']): 0.45
        }
    
    
    def set_coefficients(self): 
        return {
            "A_coef":35.745,
            "B_coef":0.38704,
            "C_coef":0.018042,
            "d_coef":1.1,
            "Paux":520,
            "time_travel":self.set_time_travel()
        }


    def run_app_from_topology(self): 
        
        self.set_vehicle(model="etron")
        pci = self.set_physics_control()
        self.vehicle.apply_physics_control(pci)

        hero_entity = Entity(model="etron", company="Audi")
        hero_entity.set_PCI(pci,self.vehicle)
        hero_entity.set_VP([], self.vehicle)

        vc = self.set_coefficients()
        hero_entity.set_VC(vc)

        hero_entity.save_to_json_file("ego_vehicle.json")
        logger.info("Saving vehicle information to ego_vehicle.json...")
        self.args.json = "ego_vehicle.json"
        self.flag = True
        hero_entity = self.run_app_from_json()

        return hero_entity


    def run_app_from_json(self): 
 
        hero_entity = Entity(vehicle=None, model=None, company=None)
        hero_entity = hero_entity.from_json_file(self.args.json)
        
        if self.flag is None: 
            self.set_vehicle(hero_entity.model)

        hero_entity.set_PCI(hero_entity.physical_controls, self.vehicle)
        self.vehicle.apply_physics_control(hero_entity.get_physical_controls())

        hero_entity.set_VP(hero_entity.vehicle_parts, self.vehicle)
        self.vehicle.apply_control(hero_entity.get_components())

        hero_entity.set_VC({"vc":hero_entity.vehicle_coefficients,"acc_time":hero_entity.acc_time})
        
        logger.info("Retrieved Information from json file")
        
        return hero_entity
        

    def set_agent_spectator(self, hero_entity):       
        agent = Agent(vehicle=self.vehicle, 
                      entity=hero_entity, 
                      target_speed=30, 
                      opt_dict={'ignore_traffic_lights': False,
                                 'ignore_stop_signs':False, 
                                 'ignore_vehicles':False},
                      verbose=self.verbose, 
                      pfa_inst=self.args.pfa
                )
        
        agent.follow_speed_limits(True)
        agent.set_destination(self.destination_point, self.start_point)

        cam_bp = self.bpl_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{600}")
        cam_bp.set_attribute("image_size_y", f"{600}")
        cam_bp.set_attribute("fov", "110")
        cam_transform = carla.Transform((carla.Location(x=-4,z=2.5)))
        camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        # spectator = self.world.get_spectator()
        # spectator.set_transform(camera.get_transform())
        self.actor_list.append(camera)
        return agent


    def set_npc_agents(self): 
        spawn_points = self.map.get_spawn_points()
        transform = random.choice(spawn_points)
        transform.location += carla.Location(x=40,y=-3.2)
        transform.rotation.yaw = -180.0
        for _ in range(0,10):
            transform.location.x += 8.0 
            bp = random.choice(self.bpl_library.filter('vehicle'))
            npc = self.world.try_spawn_actor(bp,transform)
            if npc is not None:
                self.actor_list.append(npc)
                npc.set_autopilot(True)
        logger.info("Spawnning NPCs...")


    def run_app(self, agent):
       while True : 
            if agent.done(): 
               logger.info("Agent has reached the destination...")
               logger.info("Program will now exit...")
               break 
            
            control = agent.run_step()
            control.manual_gear_shift = False
            self.vehicle.apply_control(control)

        
           



            


        


