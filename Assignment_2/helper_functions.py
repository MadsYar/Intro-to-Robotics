#
# file   helper_functions.py 
# brief  Purdue University Fall 2022 CS490 robotics Assignment2 - 
#        Motion model and landmark detection helper functions
# date   2022-08-18
#

import math
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *

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
        plt.set_aspect("equal", adjustable="box")
        plt.grid(True, ls=":")
        plt.set_xlabel("x")
        plt.set_ylabel("y")
        plt.legend(loc="best")
        plt.show()
