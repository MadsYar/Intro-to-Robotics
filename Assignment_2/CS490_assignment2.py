#
# file   CS490_assignment2.py 
# brief  Purdue University Fall 2022 CS490 robotics Assignment2 - 
#        Motion model and landmark detection
# date   2022-08-18
#

#from helper_functions import * 

# --- Add these imports near the top of your file if not present ---
import math
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *

# Helper functions
gt_landmark = [[1, 2], [4, 1], [2, 0], [5, -1], [1, -2], [3, -2]]

#visualization function
def draw_robot_trajectory(robot_node_list, gt_location = None):
    window = Tk()

    window.geometry('700x700')

    canvas = Canvas(window, width = 600, height = 600)

    canvas.pack()

    if gt_location:
        for i in range(len(gt_location)-1):
            x0, y0 = gt_location[i]
            x1, y1 = gt_location[i+1]

            x0 = int(x0 * 100)
            y0 = 600 - int(y0 * 100) - 300
            x1 = int(x1 * 100)
            y1 = 600 - int(y1 * 100) - 300

            canvas.create_line(x0, y0, x1, y1, fill = 'blue', width = 3)

    for i in range(len(robot_node_list)-1):
        x0, y0 = robot_node_list[i].x_, robot_node_list[i].y_
        x1, y1 = robot_node_list[i+1].x_, robot_node_list[i+1].y_

        x0 = int(x0 * 100)
        y0 = 600 - int(y0 * 100) - 300
        x1 = int(x1 * 100)
        y1 = 600 - int(y1 * 100) - 300

        canvas.create_line(x0, y0, x1, y1, fill = 'red', width = 3)



    window.mainloop()

# Convert a node's LiDAR scan to global coordinates
def _global(node):
    a_min, _, d_alpha, dists = node.lidar_
    angles = [a_min + i * d_alpha for i in range(len(dists))]

    global_x = [node.x_ + dists[i] * math.cos(node.theta_ + angles[i]) for i in range(len(dists))]
    global_y = [node.y_ + dists[i] * math.sin(node.theta_ + angles[i]) for i in range(len(dists))]
    return angles, dists, global_x, global_y

# Numerical gradient of the distances
def _gradient(dists):
    n = len(dists)

    if n < 3:
        return [0.0] * n
    
    gradient = [0.0] * n
    
    for i in range(1, n - 1):
        gradient[i] = (dists[i + 1] - dists[i - 1]) / 2.0

    return gradient

def question2(robot_node_list, steps=(50, 200, 350)):
    for t in steps:

        node = robot_node_list[t]
        
        angles, dists, global_x, global_y = _global(node)
        gradient = _gradient(dists)

        # Gradient plot
        plt.figure()
        plt.plot(angles, gradient)
        plt.title(f"Gradient plot at t={t}")
        plt.xlabel("rad")
        plt.ylabel("d'(li)")
        plt.grid(alpha=0.3)

        # LiDAR plot
        plt.figure()
        plt.scatter(global_x, global_y, s=6, label="LiDAR")
        plt.scatter([node.x_], [node.y_], c="black", marker="o", label="Robot")
        plt.title(f"LiDAR scan plot at t={t}")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.grid(alpha=0.3)

        # LiDAR + landmark plot
        plt.figure()
        plt.scatter(global_x, global_y, s=6, label="LiDAR")
        plt.scatter([node.x_], [node.y_], c="black", marker="o", label="Robot")
        
        # Detected landmarks
        if node.landmark_:
            landmark_x, landmark_y = zip(*node.landmark_)
            plt.scatter(landmark_x, landmark_y, marker='x', s=60, label="Landmarks")
        
        plt.legend()
        plt.title(f"LiDAR + landmarks plot at t={t}")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(alpha=0.3)

    plt.show()


def question3(robot_node_list, steps=(50, 200, 350)):
    cmap = plt.get_cmap("tab10")

    for t in steps:
        # Get data
        node  = robot_node_list[t]
        ground_truth   = np.asarray(gt_landmark)
        detected   = np.asarray(node.landmark_ or [])
        pairs = list(getattr(node, "pairs_", []) or [])
        
        plt.figure(figsize=(5, 5))
        plt.title(f"Matching pairs at timestep {t}")
        plt.scatter(ground_truth[:,0], ground_truth[:,1], marker="x", s=90, c="0.70", label="Unmatched ground truth")

        # Unmatched detections
        matched_det_idx = {i for (i, _) in pairs}
        if detected.size:
            um = [i for i in range(len(detected)) if i not in matched_det_idx]
            if um:
                plt.scatter(detected[um,0], detected[um,1], s=70, c="0.70", label="Unmatched detection")

        # Matched pairs
        for k, (i, j) in enumerate(pairs):
            c = cmap(k % 10)
            xi, yi = detected[i]
            xj, yj = ground_truth[j]
            plt.scatter([xi], [yi], s=80, color=c, label="Matched detection" if k == 0 else None)
            plt.scatter([xj], [yj], s=100, marker="x", color=c, label="Matched ground truth" if k == 0 else None)

        # Plotting
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, ls=":")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="best")
        plt.show()



#robot node class
class robot_node:
    def __init__(self, x, y, theta):
        self.x_ = x
        self.y_ = y
        self.theta_ = theta
        self.lidar_ = None
        self.landmark_ = None
        self.pairs_ = None
        self.wall_pairs_ = None

    def add_lidar_scan(self, lidar):
        self.lidar_ = lidar

    def add_detected_landmark(self, landmark):
        self.landmark_ = landmark

    def add_landmark_pairs(self, pairs):
        self.pairs_ = pairs

#Question1
#****************************************************************************************
def location_reader(filename = None):
    base = os.path.dirname(__file__)
    path = os.path.join(base, filename)

    with open(path, 'r') as file:
        return [tuple(map(float, line.split())) for line in file]

def robot_motion_reader(filename = None):
    base = os.path.dirname(__file__)
    path = os.path.join(base, filename)

    with open(path, 'r') as file:
        return [tuple(map(float, line.split())) for line in file]

def motion_model_calculation(robot_motion):
    timestep = 0.5
    x, y, theta = 1.0, 0.0, 0.0
    robot_node_list = [robot_node(x, y, theta)]
    
    for v, w in robot_motion:
        if abs(w) < 1e-9:  
            # Straight motion
            x = x + (v * timestep * math.cos(theta))
            y = y + (v * timestep * math.sin(theta))
        else: 
            # Circular motion
            x = x + ((v / w) * (math.sin(theta + w * timestep) - math.sin(theta)))
            y = y + ((v / w) * (-math.cos(theta + w * timestep) + math.cos(theta)))

        # Orientation
        theta = theta + (w * timestep)

        robot_node_list.append(robot_node(x, y, theta))

    return robot_node_list

#****************************************************************************************

#Question 2
#****************************************************************************************
def lidar_scan_reader(robot_node_list, filename = None):
    base = os.path.dirname(__file__)
    path = os.path.join(base, filename)

    with open(path, 'r') as file:

        for i, line in enumerate(file):
            node = robot_node_list[i]
            values = list(map(float, line.split()))

            a_min = values[0]
            a_max = values[1]
            d_alpha = values[2]

            distances = [3.5 if d == 10.0 else d for d in values[3:]]

            node.add_lidar_scan((a_min, a_max, d_alpha, distances))

    return robot_node_list


def landmark_detection(robot_node_list):
    for node in robot_node_list:
        
        # Skip if no LiDAR data
        if node.lidar_ is None or len(node.lidar_[3]) < 3: 
            node.add_detected_landmark([])
            continue

        a_min, _, d_alpha, distances = node.lidar_
        landmarks, inside, enter_idx = [], False, None
        n = len(distances)
        
        # Compute gradients for distance
        grads = [0.0] * n
        for i in range(1, n - 1):
            grads[i] = (distances[i + 1] - distances[i - 1]) / 2.0

        for i in range(1, n - 1):
            if not inside and grads[i] <= -0.24:
                inside = True
                enter_idx = i

            elif inside and grads[i] >= 0.24:
                detected = range(enter_idx + 1, i)
                
                # Converting from local to global coordinates
                if detected:
                    center_idx   = sum(detected) / len(detected)
                    center_angle = a_min + center_idx * d_alpha
                    center_range = (sum(distances[j] for j in detected) / len(detected)) + 0.15

                    global_angle = node.theta_ + center_angle
                    landmark_x = node.x_ + center_range * math.cos(global_angle)
                    landmark_y = node.y_ + center_range * math.sin(global_angle)
                    
                    landmarks.append((landmark_x, landmark_y))
                
                inside = False 

        node.add_detected_landmark(landmarks)

    return robot_node_list

#****************************************************************************************

#Question 3
#****************************************************************************************
def pair_landmarks(robot_node_list):
    for node in robot_node_list:
        pairs = []
        
        # Skip if no landmarks detected
        if not node.landmark_:
            node.add_landmark_pairs(pairs)
            continue
        
        # For detected landmarks, find the ground truth
        for i, (lx, ly) in enumerate(node.landmark_):
            best_j = -1
            best_d2 = float('inf')
            
            for j, (gx, gy) in enumerate(gt_landmark):
                d2 = (gx - lx)**2 + (gy - ly)**2
                
                if d2 < best_d2:
                    best_d2 = d2
                    best_j = j
            
            # Only accept matches within 1.0 units
            if math.sqrt(best_d2) <= 1.0:
                pairs.append((i, best_j))
        
        node.add_landmark_pairs(pairs)
    
    return robot_node_list
#****************************************************************************************


if __name__ == '__main__':

    #you can add visualization functions but do not change the existsing code 
    #please check what you need to implement for each function in the handout

    #Question1
    #************************************************************************************
    gt_location = location_reader('location.txt')
    robot_motion = robot_motion_reader('robot_motion.txt')
    robot_node_list = motion_model_calculation(robot_motion)

    #draw_robot_trajectory(robot_node_list, gt_location)
    #************************************************************************************

    #Question2
    #************************************************************************************
    lidar_data = lidar_scan_reader(robot_node_list, 'lidar_scan.txt')
    landmark_detection(robot_node_list)
    #question2(robot_node_list, steps=[50, 200, 350])
    #************************************************************************************

    #Question3
    #************************************************************************************
    pair_landmarks(robot_node_list)
    #question3(robot_node_list, steps=(50, 200, 350))
    #************************************************************************************



