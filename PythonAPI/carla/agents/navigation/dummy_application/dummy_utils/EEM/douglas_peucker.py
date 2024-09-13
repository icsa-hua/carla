from agents.navigation.dummy_application.dummy_utils.interface.DouglasPeucker import DPA
import numpy as np 
import pdb


class DouglasPeuckerModel(DPA):

    def __init__(self, points:list, epsilon:float, curvature:float, epsilon_curvature:float, epsilon_straight:float):
        super().__init__(points, epsilon, curvature)
        self.epsilon_curvature = epsilon_curvature
        self.epsilon_straight = epsilon_straight

    def simplify_segment(self,points, epsilon):

                
        start_point = points[0].transform.location.x, points[0].transform.location.y, points[0].transform.location.z
        end_point = points[-1].transform.location.x, points[-1].transform.location.y, points[-1].transform.location.z

        if self.calculate_distance(points[0].transform.location,points[-1].transform.location):
            return [points[0], [points[-1]]]
        
        dmax = 0.0
        index = 0
        for ii in range(1, len(points)-1): 
            current_point = points[ii].transform.location.x, points[ii].transform.location.y, points[ii].transform.location.z
            d = self.point_line_distance_3d(current_point, start_point, end_point)
            if d > dmax:
                index = ii
                dmax = d
        
        if dmax > epsilon: 
            rec_result1 = self.run(points[:index+1])
            rec_result2 = self.run(points[index:])
            return rec_result1[:-1] + rec_result2
        else: 
            return [points[0], points[-1]]


    def key_points_of_curvature(self)->list:
        """
        Identifies the key points of curvature in a list of waypoints.
        
        This method classifies each segment between consecutive waypoints as either 'straight' or 'curved' based on the calculated curvature. It then returns a list of the key waypoints, which includes the first and last waypoints, as well as any intermediate waypoints that are classified as 'curved'.
        
        Args:
            self (DouglasPeuckerModel): The instance of the DouglasPeuckerModel class.
        
        Returns:
            list: A list of the key waypoints, including the first, last, and any intermediate 'curved' waypoints.
        """
                
        segment_classification = ['straight'] * (len(self.points)-1)
        key_points = [self.points[0]]
        for i in range (1, len(self.points)-1): 
            curvature = self.calculate_curvature(self.points[i-1:i+2])
            if curvature > self.curvature:
                segment_classification[i-1] = 'curved'
                segment_classification[i] = 'curved'
                key_points.append(self.points[i])
        key_points.append(self.points[-1])
        return key_points


    def calculate_curvature(self, waypoints)->list:
        """
        Calculates the curvature of a 3D path defined by three waypoints.
        
        Args:
            waypoints (list): A list of three waypoints, represented as `Transform` objects.
        
        Returns:
            float: The curvature of the path defined by the three waypoints.
        """
                
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


    def point_line_distance_3d(self, point, line_start, line_end):
        """
        Calculates the 3D distance between a point and a line segment defined by two points.
        
        Args:
            point (np.ndarray): The 3D coordinates of the point.
            line_start (np.ndarray): The 3D coordinates of the start of the line segment.
            line_end (np.ndarray): The 3D coordinates of the end of the line segment.
        
        Returns:
            float: The 3D distance between the point and the line segment.
        """
                
        AP = np.array(point) - np.array(line_start)
        AB = np.array(line_end) - np.array(line_start)
        cross_product = np.cross(AP,AB)
        distance = np.linalg.norm(cross_product)/np.linalg.norm(AB)
        return distance


    def calculate_distance(self, p1, p2):
        return np.linalg.norm(np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]))
    

    def run(self)->list:
        
        segment_classification = self.key_points_of_curvature()
        results = [] 
        start_index = 0
        
        for i in range(1, len(segment_classification)): 
            if segment_classification[i] != segment_classification[start_index] or i == len(segment_classification)-1: 
                segment_points = self.points[start_index:i+1]
                epsilon = self.epsilon_curvature if segment_classification[start_index] == 'curved' else self.epsilon_straight
                simplified_segment = self.simplify_segment(segment_points, epsilon)
                results += simplified_segment[:-1]
                start_index = i 

        results.append(self.points[-1])
        return results

       

    