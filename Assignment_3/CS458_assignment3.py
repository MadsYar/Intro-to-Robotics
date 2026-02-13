#
# file   CS490_assignment3.py 
# brief  Purdue University Fall 2023 CS490 robotics Assignment3 - 
#        Integrating Motion model and Observation model for localization
# date   2023-10-1
#

from helper_functions import * 

#Question 1 - feature-based localization
#****************************************************************************************

def estimate_transform(left_list, right_list, fix_scale=False):
    if (not left_list) or (not right_list) or (len(left_list) != len(right_list)):
        return (1.0, 1.0, 0.0, 0.0, 0.0)

    n = len(left_list)

    if n == 1:
        (lx, ly), (rx, ry) = left_list[0], right_list[0]
        return (1.0, 1.0, 0.0, rx - lx, ry - ly)

    # Centroids
    cxL, cyL = compute_center(left_list)
    cxR, cyR = compute_center(right_list)

    # Accumulate centered cross-terms
    Sxx = Sxy = Syx = Syy = 0.0
    var_left = 0.0

    for (lx, ly), (rx, ry) in zip(left_list, right_list):
        xl = lx - cxL; yl = ly - cyL
        xr = rx - cxR; yr = ry - cyR
        Sxx += xl * xr
        Sxy += xl * yr
        Syx += yl * xr
        Syy += yl * yr
        var_left += xl * xl + yl * yl

    # Rotation
    theta = math.atan2(Sxy - Syx, Sxx + Syy)
    c = math.cos(theta)
    s = math.sin(theta)

    # Scale
    if fix_scale or var_left < 1e-10:
        la = 1.0
    else:
        la = (c * (Sxx + Syy) + s * (Syx - Sxy)) / var_left
        if la <= 0.0:
            la = 1.0

    # Translation
    tx = cxR - la * (c * cxL - s * cyL)
    ty = cyR - la * (s * cxL + c * cyL)

    return (la, c, s, tx, ty)


def transform_estimate_and_correction(robot_node_list, robot_motion):

    # Tresholds
    MIN_PAIRS = 2
    REPROJ_THRESH = 0.28
    MAX_ROT_STEP = 0.70
    MAX_CONT_STEP = 0.75
    MAX_CONT_YAW  = 0.70

    def wrap(a):
        return ((a + math.pi) % (2.0 * math.pi)) - math.pi

    # Save original poses
    mm_poses = [(n.x_, n.y_, n.theta_) for n in robot_node_list]

    # Outputs
    refined  = [(n.x_, n.y_, n.theta_) for n in robot_node_list]
    accepted = [False] * len(robot_node_list)

    # Last accepted
    last_ok = None               
    E_last  = None    

    for t, node in enumerate(robot_node_list):
        rx, ry, rtheta = node.x_, node.y_, node.theta_
        pairs = getattr(node, "pairs_", []) or []
        lands = getattr(node, "landmark_", []) or []

        # Build local and world points
        local_pts, world_pts = [], []
        if pairs and lands:
            c_inv, s_inv = math.cos(-rtheta), math.sin(-rtheta)
            for li, gi in pairs:
                if 0 <= li < len(lands) and 0 <= gi < len(gt_landmark):
                    gx, gy = lands[li]
                    dx, dy = gx - rx, gy - ry
                    lx = c_inv * dx - s_inv * dy
                    ly = s_inv * dx + c_inv * dy
                    local_pts.append((lx, ly))
                    world_pts.append(gt_landmark[gi])

        ok = False

        # Transform
        if len(local_pts) >= MIN_PAIRS:
            la, c, s, tx, ty = estimate_transform(local_pts, world_pts, fix_scale=True)
            th_hat = math.atan2(s, c)

            if abs(wrap(th_hat - rtheta)) <= MAX_ROT_STEP:
                errors = []
                
                for p, q in zip(local_pts, world_pts):
                    px = la * (c * p[0] - s * p[1]) + tx
                    py = la * (s * p[0] + c * p[1]) + ty
                    errors.append(math.hypot(px - q[0], py - q[1]))
                
                errors.sort()
                med_err = errors[len(errors)//2] if errors else 0.0
                
                if med_err <= REPROJ_THRESH:
                    x_hat, y_hat, th_hat = tx, ty, th_hat

                    if last_ok is None:
                        ok = True
                    else:
                        step = math.hypot(x_hat - last_ok[0], y_hat - last_ok[1])
                        dth  = abs(wrap(th_hat - last_ok[2]))
                        ok = (step <= MAX_CONT_STEP) and (dth <= MAX_CONT_YAW)

        if ok:
            # Accept
            refined[t]  = (x_hat, y_hat, th_hat)
            accepted[t] = True
            last_ok     = refined[t]

            #  Compute correction E
            x_mm, y_mm, th_mm = mm_poses[t]
            dth = wrap(th_hat - th_mm)
            ce, se = math.cos(dth), math.sin(dth)
            tx_e = x_hat - (ce * x_mm - se * y_mm)
            ty_e = y_hat - (se * x_mm + ce * y_mm)
            E_last = (ce, se, tx_e, ty_e)

        else:
            # Apply last correction E
            if E_last is not None:
                ce, se, tx_e, ty_e = E_last
                x_mm, y_mm, th_mm = mm_poses[t]
                x_p = ce * x_mm - se * y_mm + tx_e
                y_p = se * x_mm + ce * y_mm + ty_e
                th_p = wrap(th_mm + math.atan2(se, ce))
                refined[t] = (x_p, y_p, th_p)
                last_ok = refined[t]
            
            elif last_ok is not None:
                refined[t] = last_ok

    # Write back
    for i, node in enumerate(robot_node_list):
        node.x_, node.y_, node.theta_ = refined[i]


#****************************************************************************************


#Question 2 - featureless localization
#****************************************************************************************

def get_subsample_rays(robot_node):
    subsample = 3  
    rx, ry, rth = robot_node.x_, robot_node.y_, robot_node.theta_
    scan = robot_node.lidar_
    
    if not scan or len(scan) < 4:
        return []

    a_min, a_max, a_step = scan[0], scan[1], scan[2]
    dists = scan[3:]

    pts = []
    
    for i in range(0, len(dists), subsample):
        a = a_min + i * a_step
        th = rth + a
        r = dists[i]
        
        if 0.15 < r < 3.3:
            ex = rx + math.cos(th) * r
            ey = ry + math.sin(th) * r
            pts.append((ex, ey))
    
    return pts


def get_corresponding_points_on_boundary(points):
    if not points:
        return [], []

    xmin, xmax = 0.0, 6.0
    ymin, ymax = -3.0, 3.0
    thresh = 0.4

    laser_pts, wall_pts = [], []
    
    for (x, y) in points:
        d_left = abs(x - xmin)
        d_right = abs(x - xmax)
        d_top = abs(y - ymax)
        d_bottom = abs(y - ymin)
        min_dist = min(d_left, d_right, d_top, d_bottom)
        
        if min_dist > thresh:
            continue
            
        if min_dist == d_left:
            q = (xmin, max(ymin, min(ymax, y)))
        elif min_dist == d_right:
            q = (xmax, max(ymin, min(ymax, y)))
        elif min_dist == d_top:
            q = (max(xmin, min(xmax, x)), ymax)
        else:
            q = (max(xmin, min(xmax, x)), ymin)

        laser_pts.append((x, y))
        wall_pts.append(q)

    return laser_pts, wall_pts


def get_icp_transform(points, iterations):
    laser_pts, wall_pts = points
    
    if len(laser_pts) < 8:
        return (1.0, 1.0, 0.0, 0.0, 0.0)

    T_total = (1.0, 1.0, 0.0, 0.0, 0.0)
    src = laser_pts[:]

    for it in range(iterations):
        T_k = estimate_transform(src, wall_pts, fix_scale=True)
        T_total = concatenate_transform(T_k, T_total)
        src = [apply_transform(T_k, p) for p in src]
        
        if it % 3 == 0:
            src, wall_pts = get_corresponding_points_on_boundary(src)
            
            if len(src) < 8:
                break

    return T_total


def featureless_transform_estimate_and_correction(robot_node_list, robot_motion):

    def wrap(a):
        return ((a + math.pi) % (2.0 * math.pi)) - math.pi

    n = len(robot_node_list)
    
    # Store motion model poses
    mm_poses = [(node.x_, node.y_, node.theta_) for node in robot_node_list]
    
    # Collect laser-wall correspondences
    all_laser_pts = []
    all_wall_pts = []
    
    for t, node in enumerate(robot_node_list):
        world_pts = get_subsample_rays(node)
        
        if len(world_pts) < 10:
            continue
        
        laser_pts, wall_pts = get_corresponding_points_on_boundary(world_pts)
        
        if len(laser_pts) >= 10:
            all_laser_pts.extend(laser_pts)
            all_wall_pts.extend(wall_pts)
    
    if len(all_laser_pts) < 50:
        return
    
    T_global = get_icp_transform((all_laser_pts, all_wall_pts), iterations=30)
    
    global_corrected = []
    
    for t in range(n):
        x, y, th = mm_poses[t]
        x_new, y_new, th_new = correct_pose((x, y, th), T_global)
        global_corrected.append((x_new, y_new, th_new))
    
    for t in range(n):
        node = robot_node_list[t]
        
        node.x_, node.y_, node.theta_ = global_corrected[t]
        
        world_pts = get_subsample_rays(node)
        
        if len(world_pts) < 10:
            continue
        
        laser_pts, wall_pts = get_corresponding_points_on_boundary(world_pts)
        
        if len(laser_pts) < 10:
            continue
        
        T_local = get_icp_transform((laser_pts, wall_pts), iterations=15)
        
        x_curr, y_curr, th_curr = node.x_, node.y_, node.theta_
        x_new, y_new, th_new = correct_pose((x_curr, y_curr, th_curr), T_local)
        
       
        errors = [math.hypot(apply_transform(T_local, p)[0] - q[0],
                          apply_transform(T_local, p)[1] - q[1])
               for p, q in zip(laser_pts, wall_pts)]
        
        if errors:
            errors.sort()
            median_err = errors[len(errors)//2]
            
            angle_change = abs(wrap(th_new - th_curr))
            pos_change = math.hypot(x_new - x_curr, y_new - y_curr)
            
            if median_err < 0.15 and angle_change < 0.3 and pos_change < 0.3:
                node.x_ = x_new
                node.y_ = y_new
                node.theta_ = th_new
    
    poses = [(node.x_, node.y_, node.theta_) for node in robot_node_list]
    smoothed = []
    
    for t in range(n):
        window = 3
        start = max(0, t - window)
        end = min(n, t + window + 1)
        
        positions = [(poses[i][0], poses[i][1]) for i in range(start, end)]
        x_avg, y_avg = compute_center(positions)
        
        cos_sum = sum(math.cos(poses[i][2]) for i in range(start, end))
        sin_sum = sum(math.sin(poses[i][2]) for i in range(start, end))
        th_avg = math.atan2(sin_sum, cos_sum)
        
        smoothed.append((x_avg, y_avg, th_avg))
    
    for t, node in enumerate(robot_node_list):
        node.x_, node.y_, node.theta_ = smoothed[t]

#****************************************************************************************



if __name__ == '__main__':

    #please don't change existing code in this main function
    #check what you need to implement for each each function in the handout

    #correct implementation of all functions of assignment2 are provided
    #************************************************************************************
    gt_location = location_reader('location.txt')
    robot_motion = robot_motion_reader('robot_motion.txt')
    robot_node_list = motion_model_calculation(robot_motion)

    lidar_data = lidar_scan_reader(robot_node_list, 'lidar_scan.txt')

    robot_node_list_feature_based = copy.deepcopy(robot_node_list)

    robot_node_list_featureless = copy.deepcopy(robot_node_list)

    landmark_detection(robot_node_list_feature_based)

    pair_landmarks(robot_node_list_feature_based)
    #************************************************************************************


    #Question 1
    transform_estimate_and_correction(robot_node_list_feature_based, robot_motion)

    #Question 2
    featureless_transform_estimate_and_correction(robot_node_list_featureless, robot_motion)


    # draw_robot_trajectory(robot_node_list_feature_based, gt_location)

    # draw_robot_trajectory(robot_node_list_featureless, gt_location)