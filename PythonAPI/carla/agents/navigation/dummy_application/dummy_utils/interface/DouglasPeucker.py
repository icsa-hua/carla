from abc import ABC, abstractmethod

class DPA(ABC): 

    @abstractmethod
    def __init__(self, points:list, epsilon:float, curvature:float)->None:
        self.points = points
        self.epsilon = epsilon
        self.curvature = curvature

    @abstractmethod
    def simplify_segment(self, points:list, epsilon:float)->list: 
        """
        Simplifies a list of waypoints using the Douglas-Peucker algorithm.
        
        Args:
            points (list): A list of waypoints represented as (x, y, z) tuples.
            epsilon (float): The maximum distance a point can be from the simplified line.
        """
        pass 

    @abstractmethod
    def key_points_of_curvature(self)->list:
        """
        Identifies the key points of curvature in a list of waypoints.
        """
        pass

    @abstractmethod
    def calculate_curvature(self, waypoints)->list:
        """
        Calculates the curvature of a 3D path defined by three waypoints.
        """
        pass

    @abstractmethod
    def point_line_distance_3d(self, point, line_start, line_end)->float:
        """
        Calculates the 3D distance between a point and a line segment defined by two points.
        Args:
            point : The 3D coordinates of the point.
            line_start : The 3D coordinates of the start of the line segment.
            line_end : The 3D coordinates of the end of the line segment.
        """
        pass

    @abstractmethod
    def run(self)->list: 
        """
        Execute the Douglas-Peucker algorithm on a list of waypoints.
        """
        pass