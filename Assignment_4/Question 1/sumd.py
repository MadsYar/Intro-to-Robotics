import numpy as np

# Load the data
rrt = np.loadtxt("rrt_clean.txt")
rrt_star = np.loadtxt("rrtstar_clean.txt")

# Compute column-wise averages
rrt_mean = np.mean(rrt, axis=0)
rrt_star_mean = np.mean(rrt_star, axis=0)

print("RRT column averages:", rrt_mean)
print("RRT* column averages:", rrt_star_mean)
