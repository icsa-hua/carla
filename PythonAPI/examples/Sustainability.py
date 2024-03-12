import glob 
import os 
import sys 
import random
import time
import argparse
import warnings

try: 
    import numpy as np
except ImportError: 
    raise RuntimeError('Cannot import numpy, make sure numpy is installed')
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
except IndexError:
    pass

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from agents.navigation.Basic_RP_agent import Agent
from agents.navigation.Vehicle_design import VehicleDesign,PhysicalControls,VehicleCoefficients, VehicleParts, VelocityTuple

IM_WIDTH = 640
IM_HEIGHT = 480
bg_photo = []
def process_image(image):
    cc = carla.ColorConverter.LogarithmicDepth
    image.save_to_disk('_out/%06d.png' % image.frame, cc)
    i = np.array(image.raw_data)

    #the way the images come from carla are flatten(all bits are in one dimension)
    i2 = i.reshape((IM_HEIGHT,IM_WIDTH,4)) #The four is the 3 rgb and a for alpha information
    i3 = i2[:,:, :3] #Get the first three elements so take the height, the width and the rgb values 
    #cv2.imshow("",i3)
    #cv2.waitKey(1)
    return i3/255.0

class WaypointSelector:
    def __init__(self, master, map,background_image):
        self.master = master
        self.master.title("-Location Setter-")
        self.map = map.get_spawn_points()

        self.start_waypoint = None
        self.end_waypoint = None
        self.mode = None
        background_image = background_image.replace("\\","/")
        max_x = max(waypoint.location.x for waypoint in self.map)
        max_y = max(waypoint.location.y for waypoint in self.map)
        self.window_width = max_x + 250
        self.window_height = max_y + 250

        self.canvas = tk.Canvas(master, width=self.window_width, height=self.window_height)
        self.canvas.pack()

        self._bg_image = Image.open(background_image)
        # self._bg_photo = ImageTk.PhotoImage(self._bg_image)

        # self.canvas.create_image(0, 0, anchor="nw", image=self._bg_photo)
        self.draw_map()
        
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.start_button = tk.Button(master, text="Select Start Waypoint", command=self.select_start)
        self.start_button.pack()
        
        self.end_button = tk.Button(master, text="Select End Waypoint", command=self.select_end)
        self.end_button.pack()
        
        self.done_button = tk.Button(master, text="Done", command=self.done)
        self.done_button.pack()
        
    def draw_map(self):
        # Draw map using waypoints
        # Group waypoints by location (e.g., every 5 units)
        grouped_waypoints = {}
        for waypoint in self.map:
            x_group = round(waypoint.location.x / 5) * 5
            y_group = round(waypoint.location.y / 5) * 5
            grouped_waypoints.setdefault((x_group, y_group), []).append(waypoint)
        for group, waypoints in grouped_waypoints.items():
            x, y = group
            color = "blue" if len(waypoints) > 1 else "grey"
            for waypoint in waypoints:
                self.canvas.create_oval(waypoint.location.x-2, waypoint.location.y-5, waypoint.location.x+5, waypoint.location.y+2, fill=color)
    
    def on_click(self, event):
        if self.mode == "start":
            self.start_waypoint = self.get_closest_waypoint(event.x, event.y)
            print("Start Waypoint:", self.start_waypoint)
        elif self.mode == "end":
            self.end_waypoint = self.get_closest_waypoint(event.x, event.y)
            print("End Waypoint:", self.end_waypoint)
    
    def select_start(self):
        self.mode = "start"
        print("Selecting Start Waypoint...")

    def select_end(self):
        self.mode = "end"
        print("Selecting End Waypoint...")        
    
    def done(self):
        if self.start_waypoint is None or self.end_waypoint is None:
            print("Please select both start and end waypoints.")
        else:
            self.master.destroy()

    def get_closest_waypoint(self, x, y):
        closest_waypoint = None
        min_distance = float('inf')
        for waypoint in self.map:
            distance = ((waypoint.location.x - x) ** 2 + (waypoint.location.y - y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_waypoint = waypoint
        return closest_waypoint


def main():
    actor_list = []
    try: 
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
                '--json', 
                default=None,
                help='Provide absolute path of the json file')
        argparser.add_argument(
                '--verbose', 
                default = False, 
                help='Select if you want to show the plots for the route establishment'
        )
        argparser.add_argument(
                '--sync', 
                default = False, 
                help='Select if you want to alernate the environment settings'
        )

        if len(sys.argv) < 1:
                argparser.print_help()
                return
        args = argparser.parse_args()

        print("Attempting to connect to host...\n")
        
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        print("Connected to localhost\n")
        sim_world = client.get_world() 
        sim_map = sim_world.get_map() 
        traffic_manager = client.get_trafficmanager()
        if args.sync: 
            settings = sim_world.get_settings()
            delta = 0.5 
            settings.fixed_delta_seconds = delta 
            settings.synchronous_mode = True
            sim_world.apply_settings(settings) 
            traffic_manager.set_synchronous_mode(True)

        spawn_points = sim_map.get_spawn_points()

        dirname = os.getcwd()
        directory = os.path.join(dirname,"Town_images/")
        if not os.path.exists(directory):
            os.makedirs(directory)

        top = sim_map.get_topology()
        for link in range(len(top)): 
            cur_link = top[link]
            x1, y1 = cur_link[0].transform.location.x, cur_link[0].transform.location.y    
            x2, y2 = cur_link[-1].transform.location.x, cur_link[-1].transform.location.y 
            plt.plot([-x1,-x2], [y1,y2], marker = 'o')
        plt.title("Final Processed Path")
        # plt.show()
        save_town = os.path.join(directory, "Town.png")
        plt.savefig(save_town)


        # root = tk.Tk()
        # app = WaypointSelector(root,sim_map,save_town)
        # root.mainloop()
        # start_point=app.start_waypoint
        # end_point = app.end_waypoint
        # start_point = carla.Location(start_point.location)
        # end_point = carla.Location(end_point.location)
        
        spawn_points = sim_map.get_spawn_points()
        start_point = carla.Location(spawn_points[50].location)
        end_point = carla.Location(spawn_points[100].location)

        print(f"Processing JSON file for path {args.json} ")
        vehicle_parser = VehicleDesign() 
        vehicle_design = vehicle_parser.from_json_file(args.json)
        
        print("Creating vehicle...")
        blueprint_library = sim_world.get_blueprint_library()
        bp = blueprint_library.filter(vehicle_design._model)[0]
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color',color)
        bp.set_attribute('role_name','hero')

        spawn_vehicle_wp = sim_map.get_waypoint(start_point).transform
        # spawn_vehicle_wp.location.z += 2.5 
        spawn_vehicle_wp.location.z += 2.5 
        ego_vehicle = sim_world.spawn_actor(bp,spawn_vehicle_wp)
        actor_list.append(ego_vehicle)
        print('Created %s' % ego_vehicle.type_id)

        physics_control = ego_vehicle.get_physics_control()
        torque_curve = vehicle_design.from_serializable(vehicle_design._phy_controls["torque_curve"])
        physics_control.torque_curve = [torque for torque in torque_curve]
        physics_control.max_rpm = vehicle_design._phy_controls["max_rpm"]
        physics_control.moi = vehicle_design._phy_controls["moi"]
        physics_control.damping_rate_full_throttle = vehicle_design._phy_controls["damping_rate_full_throttle"]
        physics_control.use_gear_autobox = vehicle_design._phy_controls["use_gear_autobox"]
        physics_control.gear_switch_time = vehicle_design._phy_controls["gear_switch_time"]
        physics_control.clutch_strength = vehicle_design._phy_controls["clutch_strength"]
        physics_control.mass = vehicle_design._phy_controls["mass"]
        physics_control.drag_coefficient = vehicle_design._phy_controls["drag_coefficient"]
        steering_curve = vehicle_design.from_serializable(vehicle_design._phy_controls["steering_curve"])
        physics_control.steering_curve = steering_curve
        physics_control.use_sweep_wheel_collision = vehicle_design._phy_controls["use_sweep_wheel_collision"]

        wheels = [ carla.WheelPhysicsControl(tire_friction=wheel["tire_friction"],damping_rate=wheel["damping_rate"],
                                                       max_steer_angle=wheel["max_steer_angle"],long_stiff_value=wheel["long_stiff_value"],
                                                       radius=wheel["radius"], max_brake_torque=wheel["max_brake_torque"],
                                                       lat_stiff_max_load=wheel["lat_stiff_max_load"],lat_stiff_value=wheel["lat_stiff_value"])
                                                       for wheel in vehicle_design._phy_controls["wheels"]]
        physics_control.wheels = wheels 
        ego_vehicle.apply_physics_control(physics_control)

        vc_data= vehicle_design._vehicle_coeffs 
        time_travel = vehicle_design._acc_time
        vehicle_coeffs = VehicleCoefficients(A_coef=vc_data["A_coef"],B_coef=vc_data["B_coef"],
                                             C_coef=vc_data["C_coef"],d_coef=vc_data["d_coef"],
                                             g_coef=vc_data["g_coef"],ndis=vc_data["ndis"],
                                             nchg=vc_data["nchg"],naux_move=vc_data["naux_move"],
                                             naux_idle=vc_data["naux_idle"],P_aux=vc_data["Paux"],
                                             nd=vc_data["nd"],n_aux=vc_data["n_aux"])
        
        acc_time = {} 
        for time in time_travel: 
            var = VelocityTuple(time)
            result = var.convert_to()
            acc_time[result] = time_travel[time]
            
        vehicle_coeffs._acc_time = acc_time
        oct_dictionary = {'ignore_traffic_lights':False,'ignore_stop_signs': False,'ignore_vehicles': False}
        agent = Agent(vehicle=ego_vehicle, vehicle_coeffs=vehicle_coeffs,target_speed=30,opt_dict=oct_dictionary,verbose=args.verbose)
        agent.follow_speed_limits(True)
        agent.set_destination(end_point, start_point)

        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        cam_bp.set_attribute("fov", "110")
        cam_transform = carla.Transform((carla.Location(x=-4,z=2.5)))#)
        camera = sim_world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
        spectator = sim_world.get_spectator()
        spectator.set_transform(camera.get_transform())
        actor_list.append(camera)

        spawn_cam_point = carla.Transform(carla.Location(x=2.5, z=0.7))#Relative position depending on the vehicle.
        sensor = sim_world.spawn_actor(cam_bp,spawn_cam_point,attach_to=ego_vehicle)
        actor_list.append(sensor)
        # sensor.listen(lambda data: process_image(data))
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
        time = 0 
        while True:
            time = time + 1         
            spectator.set_transform(camera.get_transform())
            if agent.done(): 
                print("The target has been reached, stopping the simulation")
                break
            control = agent.run_step()
            control.manual_gear_shift = False 
            ego_vehicle.apply_control(control)
        print(f"Simulation Ended after {time} steps")
    finally: 
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print("All clients are cleaned")    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except RuntimeError as e:
        print(e)


























































