import tkinter as tk
from tkinter import *
from PIL import ImageTk 
import logging
import warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 


class WaypointInterface:

    def __init__(self, master, map, bg_img) -> None:
        self.master = master 
        self.map = map.get_spawn_points() 
        self.bg_image = bg_img 
        self.start_wp = None
        self.end_wp = None 
        self.set_waypoints()
        self.set_image()
        self.tr_wps = self.transform_waypoints()
        self.create_window()
        self.draw_map()
        self.insert_buttons()


    def create_window(self):
        self.master.title("Waypoint Selector")
        self.canvas = tk.Canvas(self.master, width=self.window_width, height=self.window_height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor='nw', image=self.bg)


    def set_waypoints(self): 
        self.max_x = max(waypoint.location.x for waypoint in self.map)
        self.min_x = min(waypoint.location.x for waypoint in self.map)
        self.max_y = max(waypoint.location.y for waypoint in self.map)
        self.min_y = min(waypoint.location.y for waypoint in self.map)
        

    def set_image(self):
        image_width, image_height = self.bg_image.size
        self.window_width = image_width
        self.window_height = image_height
        self._scale_x = image_width /  (self.max_x - self.min_x) - 0.03 
        self._scale_y = image_height / (self.max_y - self.min_y)  - 0.03        
        self.bg = ImageTk.PhotoImage(self.bg_image, master=self.master)
        label = Label(self.master, image=self.bg)
        label.image=self.bg


    def transform_waypoints(self): 
        transformed_waypoints = []
        for waypoint in self.map:
            transformed_x = (waypoint.location.x - self.min_x ) * self.scale_x 
            transformed_y = (waypoint.location.y - self.min_y ) * self.scale_y
            transformed_waypoints.append((transformed_x, transformed_y))
        return transformed_waypoints
    

    def draw_map(self): 
        for wp in self.tr_wps: 
            self.canvas.create_oval(
                wp[0] - 2,
                wp[1] - 5,
                wp[0] + 2,
                wp[1] + 5,
                fill='blue'
            )


    def insert_buttons(self): 
        self.canvas.bind("<Button-1>", self.on_click)
        self.start_button = tk.Button(self.master, text="Select Starting Position", command=self.select_start_wp)
        self.start_button.pack() 
        self.end_button = tk.Button(self.master, text="Select Destination Position", command=self.select_end_wp)
        self.end_button.pack() 
        self.done_button = tk.Button(self.master, text="Done", command=self.done)
        self.done_button.pack() 


    def select_start_wp(self): 
        self.mode = 'start'
        logging.debug('Selecting start waypoint...')


    def select_end_wp(self): 
        self.mode = 'end' 
        logging.debug('Selecting destination waypoint...')


    def get_closest_wp(self, x, y): 
        best_wp = None 
        min_dist = float('inf')
        for wp in self.map: 
            dist = ((wp.location.x - x) ** 2 + (wp.location.y - y) ** 2) ** 0.5
            if dist < min_dist: 
                min_dist = dist
                best_wp = wp
        return best_wp


    def on_click(self, event): 

        if self.mode == 'start': 
            self.start_wp = self.get_closest_wp(event.x,event.y)
            if self.start_wp not in self.map: 
                self.start_wp = self.map[0]
                logging.debug('Start waypoint is not in the map. Resetting to first waypoint.')

        if self.mode == 'end': 
            self.end_wp = self.get_closest_wp(event.x,event.y) 
            if self.end_wp not in self.map:
                self.end_wp = self.map[-1]
                logging.debug('Destination waypoint is not in the map. Resetting to last waypoint.')


    def done(self):
        if self.start_wp is None or self.end_wp is None:
            warnings.warn("Please select both start and end waypoints.")
        else:
            self.master.quit()
            self.master.destroy()
