# CS 45800: Introduction to Robotics

This repository contains coursework for CS 45800: Intro to Robotics at Purdue University. They go through concepts in robotics including computer vision, localization, path planning and control systems.

---

## Course Overview

The assignments progressively build up core robotics concepts:

1. **Perception** - How robots see and interpret the world (GDA segmentation)
2. **Motion** - How robots move and process sensor data (motion models, LiDAR)
3. **Localization** - Where the robot is in the world (pose estimation)
4. **Planning** - Finding paths to reach goals (RRT/RRT*)
5. **Control** - Executing planned motions accurately (PD controllers)

---

## Assignment 1: Image Segmentation with Gaussian Discriminant Analysis

This assignment focuses on detecting objects (specifically barrels) in images using machine learning.

**What it does:**
- Uses Gaussian Discriminant Analysis (GDA) to classify pixels as either part of a barrel or background
- Implements two versions: one where both classes share the same variance, and one where each class has its own variance
- Allows users to manually label training regions in images (selecting barrel and non-barrel areas)
- Trains a model on labeled data and then segments new test images

**Key concepts:**
- Color-based segmentation using RGB features
- Statistical classification with Gaussian distributions
- Computing means, covariances and probabilities for classification

---

## Assignment 2: Robot Motion and Landmark Detection

This assignment simulates a mobile robot moving through an environment and detecting landmarks using LiDAR.

**What it does:**
- Models robot motion based on velocity (forward speed) and angular velocity (turning rate)
- Processes LiDAR scans to detect cylindrical landmarks (like poles or barrels)
- Identifies landmarks by finding sharp changes (gradients) in distance measurements
- Matches detected landmarks with known ground truth landmarks

**Key concepts:**
- Odometry and motion models (how robots track their movement)
- LiDAR data processing and interpretation
- Gradient-based feature detection
- Coordinate transformations from robot-local to global coordinates

---

## Assignment 3: Robot Localization

This assignment integrates motion models with sensor observations to accurately determine where a robot is located.

**What it does:**
- Combines dead reckoning (tracking motion) with landmark observations to reduce position errors
- Implements feature-based localization using detected landmarks
- Implements featureless localization using Iterative Closest Point (ICP) algorithm
- Corrects robot pose estimates by aligning LiDAR measurements with known boundaries

**Key concepts:**
- Sensor fusion (combining multiple sources of information)
- Iterative Closest Point (ICP) for alignment
- Transformation estimation (rotation, translation, and scale)
- Error correction and pose refinement

---

## Assignment 4: Path Planning with RRT and RRT*

This assignment implements algorithms for planning collision-free paths through obstacle-filled environments.

**What it does:**
- Implements Rapidly-exploring Random Tree (RRT) algorithm
- Implements RRT* with path optimization through rewiring
- Plans paths for point robots and rectangular robots
- Handles different obstacle configurations
- Compares path quality and computation time between RRT and RRT*

**Key concepts:**
- Sampling-based path planning
- Tree data structures for exploring space
- Collision detection for different robot geometries
- Path optimization and cost reduction
- Trade-offs between computation time and solution quality

---

## Assignment 5: Classical Control for Robotic Arms

This assignment implements PD (Proportional-Derivative) controllers to make a 2-joint robotic arm follow a specific trajectory.

**What it does:**
- Derives forward kinematics (joint angles → end-effector position)
- Computes the Jacobian matrix (relates joint velocities to end-effector velocities)
- Implements a closed-loop PD controller for trajectory tracking
- Simulates the robot arm using PyBullet physics engine
- Tunes controller parameters to minimize tracking error

**Key concepts:**
- Forward and inverse kinematics
- Jacobian matrices and differential kinematics
- PD control theory (proportional and derivative gains)
- Trajectory tracking
- Control tuning and error analysis
