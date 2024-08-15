import numpy as np
from PA_CommonRoad.planner.Frenet.utils.frenet_functions import FrenetTrajectory

def cartesian_to_frenet_b(spline, x, y, psi, v, delta, dt, wheelbase):
    """
    Convert Cartesian trajectory to Frenet frame.
    
    :param spline: CubicSpline2D object representing the reference trajectory
    :param x: Array of x coordinates of the trajectory
    :param y: Array of y coordinates of the trajectory
    :param psi: Array of heading angles of the trajectory
    :param v: Array of velocities along the trajectory
    :return: Frenet frame coordinates (s, d, theta, v_s, v_d)
    """
    s_frenet = []
    d_frenet = []

  

    for i in range(len(x)):
        # Step 1: Calculate the closest point on the reference spline
        s, _ = spline.get_min_arc_length([x[i], y[i]])
        
        # Step 2: Calculate the Frenet coordinates (s, d)
        sx, sy = spline.calc_position(s)
        dx = x[i] - sx
        dy = y[i] - sy
        d = np.hypot(dx, dy)  # Euclidean distance as the lateral deviation

        # Calculate the sign of d based on the cross product
        yaw_ref = spline.calc_yaw(s)
        normal = np.array([-np.sin(yaw_ref), np.cos(yaw_ref)])
        sign_d = np.sign(np.dot([dx, dy], normal))
        d *= sign_d  # Apply the sign to the lateral deviation

        # Store the Frenet frame coordinates
        s_frenet.append(s)
        d_frenet.append(d)



    # COnvert to NP arrays
    s_frenet = np.array(s_frenet)
    d_frenet = np.array(d_frenet)



    # Find 1 2 and 3 derivatives of s and d with timestep dt 
    ds = np.gradient(s_frenet, dt)
    dds = np.gradient(ds, dt)
    ddds = np.gradient(dds, dt)

    dd = np.gradient(d_frenet, dt)
    ddd = np.gradient(dd, dt)
    dddd = np.gradient(ddd, dt)

    # Find curvature with steering angle delta
    curvature = np.tan(delta) / wheelbase


    t = np.arange(0, len(x)*dt, dt)


    return FrenetTrajectory( t=t, d=d_frenet, d_d=dd, d_dd=ddd, d_ddd=dddd, s=s_frenet, s_d=ds, s_dd=dds, s_ddd=ddds, x=x, y=y, yaw=psi, v=v, curv=curvature)



def cartesian_to_frenet_c(spline, x, y, psi, v, delta, dt, wheelbase):
    """
    Convert Cartesian trajectory to Frenet frame with derivatives and curvature.

    :param spline: CubicSpline2D object representing the reference trajectory
    :param x: Array of x coordinates of the trajectory
    :param y: Array of y coordinates of the trajectory
    :param psi: Array of heading angles of the trajectory
    :param v: Array of velocities along the trajectory
    :param delta: Array of steering angles along the trajectory
    :param dt: Time step between each trajectory point
    :param wheelbase: The wheelbase of the vehicle
    :return: FrenetTrajectory object containing the Frenet frame coordinates and derivatives
    """
    # Initialize lists to hold Frenet frame coordinates
    s_frenet = np.zeros_like(x)
    d_frenet = np.zeros_like(x)

    # Step 1: Calculate the closest points on the reference spline
    for i in range(len(x)):
        s_frenet[i], _ = spline.get_min_arc_length([x[i], y[i]])

    # Step 2: Calculate the position and yaw at the closest points on the reference spline
    sx, sy = spline.calc_position(s_frenet)
    yaw_ref = np.array([spline.calc_yaw(s) for s in s_frenet])

    # Calculate dx, dy, and lateral deviation (d)
    dx = x - sx
    dy = y - sy
    d_frenet = np.hypot(dx, dy)

    # Calculate the sign of d based on the cross product
    normal_x = -np.sin(yaw_ref)
    normal_y = np.cos(yaw_ref)
    signs = np.sign(dx * normal_x + dy * normal_y)
    d_frenet *= signs

    # Convert to NumPy arrays
    s_frenet = np.array(s_frenet)
    d_frenet = np.array(d_frenet)

    # Step 3: Compute the derivatives of s and d with respect to time
    ds = np.gradient(s_frenet, dt)
    dds = np.gradient(ds, dt)
    ddds = np.gradient(dds, dt)

    dd = np.gradient(d_frenet, dt)
    ddd = np.gradient(dd, dt)
    dddd = np.gradient(ddd, dt)

    # Step 4: Calculate curvature based on steering angle delta
    curvature = np.tan(delta) / wheelbase

    print(curvature.shape)
    # Time array
    t = np.arange(0, len(x) * dt, dt)

    # Step 5: Create and return the FrenetTrajectory object
    return FrenetTrajectory(
        t=t,
        d=d_frenet,
        d_d=dd,
        d_dd=ddd,
        d_ddd=dddd,
        s=s_frenet,
        s_d=ds,
        s_dd=dds,
        s_ddd=ddds,
        x=x,
        y=y,
        yaw=psi,
        v=v,
        curv=curvature
    )




def compute_reference_info(x_ref, y_ref):
    """
    Pre-compute the arc length, yaw, and other relevant information for the reference trajectory.

    :param x_ref: Array of x coordinates of the reference trajectory
    :param y_ref: Array of y coordinates of the reference trajectory
    :return: s_ref, yaw_ref, normal_ref
    """
    # Calculate differences between waypoints
    dx = np.diff(x_ref)
    dy = np.diff(y_ref)

    # Cumulative arc length along the trajectory
    s_ref = np.hstack(([0], np.cumsum(np.hypot(dx, dy))))

    # Yaw angles along the trajectory
    yaw_ref = np.arctan2(dy, dx)
    yaw_ref = np.hstack((yaw_ref, yaw_ref[-1]))  # Make the array same length as x_ref, y_ref

    # Normal vectors (perpendicular to the yaw angles)
    normal_ref_x = -np.sin(yaw_ref)
    normal_ref_y = np.cos(yaw_ref)

    return s_ref, yaw_ref, normal_ref_x, normal_ref_y

def cartesian_to_frenet_w(x_ref, y_ref, s_ref, yaw_ref, normal_ref_x, normal_ref_y, x, y, psi, v, delta, dt, wheelbase):
    """
    Convert Cartesian trajectory to Frenet frame using reference points.
    
    :param x_ref: Array of x coordinates of the reference trajectory
    :param y_ref: Array of y coordinates of the reference trajectory
    :param s_ref: Precomputed arc lengths of the reference trajectory
    :param yaw_ref: Precomputed yaw angles of the reference trajectory
    :param normal_ref_x: Precomputed x components of normal vectors to the reference trajectory
    :param normal_ref_y: Precomputed y components of normal vectors to the reference trajectory
    :param x: Array of x coordinates of the trajectory
    :param y: Array of y coordinates of the trajectory
    :param psi: Array of heading angles of the trajectory
    :param v: Array of velocities along the trajectory
    :param delta: Array of steering angles along the trajectory
    :param dt: Time step between each trajectory point
    :param wheelbase: The wheelbase of the vehicle
    :return: FrenetTrajectory object containing the Frenet frame coordinates and derivatives
    """
    # Initialize lists to hold Frenet frame coordinates
    s_frenet = np.zeros_like(x)
    d_frenet = np.zeros_like(x)

    # Step 1: Find the closest points on the reference trajectory
    dists = np.hypot(x[:, np.newaxis] - x_ref, y[:, np.newaxis] - y_ref)
    min_indices = np.argmin(dists, axis=1)

    # Corresponding arc length (s) and yaw at the closest point
    s_frenet = s_ref[min_indices]
    yaw_frenet = yaw_ref[min_indices]
    normal_x = normal_ref_x[min_indices]
    normal_y = normal_ref_y[min_indices]

    # Step 2: Compute lateral deviation (d) using normal vectors
    dx = x - x_ref[min_indices]
    dy = y - y_ref[min_indices]
    d_frenet = dx * normal_x + dy * normal_y

    # Step 3: Compute the derivatives of s and d with respect to time
    ds = np.gradient(s_frenet, dt)
    dds = np.gradient(ds, dt)
    ddds = np.gradient(dds, dt)


    dd = np.gradient(d_frenet, dt)
    ddd = np.gradient(dd, dt)
    dddd = np.gradient(ddd, dt)

    # Step 4: Calculate curvature based on steering angle delta
    curvature = np.tan(delta) / wheelbase

    # Time array
    t = np.arange(0, len(x) * dt, dt)

    # Step 5: Create and return the FrenetTrajectory object
    return FrenetTrajectory(
        t=t,
        d=d_frenet,
        d_d=dd,
        d_dd=ddd,
        d_ddd=dddd,
        s=s_frenet,
        s_d=ds,
        s_dd=dds,
        s_ddd=ddds,
        x=x,
        y=y,
        yaw=psi,
        v=v,
        curv=curvature
    )


def cartesian_to_frenet(x_ref, y_ref, s_ref, yaw_ref, normal_ref_x, normal_ref_y, x, y, psi, v, delta, dt, wheelbase):
    """
    Convert Cartesian trajectory to Frenet frame using reference points with vectorized interpolation.
    
    :param x_ref: Array of x coordinates of the reference trajectory
    :param y_ref: Array of y coordinates of the reference trajectory
    :param s_ref: Precomputed arc lengths of the reference trajectory
    :param yaw_ref: Precomputed yaw angles of the reference trajectory
    :param normal_ref_x: Precomputed x components of normal vectors to the reference trajectory
    :param normal_ref_y: Precomputed y components of normal vectors to the reference trajectory
    :param x: Array of x coordinates of the trajectory
    :param y: Array of y coordinates of the trajectory
    :param psi: Array of heading angles of the trajectory
    :param v: Array of velocities along the trajectory
    :param delta: Array of steering angles along the trajectory
    :param dt: Time step between each trajectory point
    :param wheelbase: The wheelbase of the vehicle
    :return: FrenetTrajectory object containing the Frenet frame coordinates and derivatives
    """
    # Step 1: Compute the Euclidean distances from trajectory points to each reference point
    dists = np.hypot(x[:, np.newaxis] - x_ref, y[:, np.newaxis] - y_ref)

    # Step 2: Find the indices of the two closest reference points for each trajectory point
    min_indices = np.argpartition(dists, 1, axis=1)[:, :2]
    min_indices.sort(axis=1)  # Ensure indices are in order for interpolation

    # Extract the relevant reference points and arc lengths
    s1 = s_ref[min_indices[:, 0]]
    s2 = s_ref[min_indices[:, 1]]
    x1 = x_ref[min_indices[:, 0]]
    y1 = y_ref[min_indices[:, 0]]
    x2 = x_ref[min_indices[:, 1]]
    y2 = y_ref[min_indices[:, 1]]
    n1_x = normal_ref_x[min_indices[:, 0]]
    n1_y = normal_ref_y[min_indices[:, 0]]
    n2_x = normal_ref_x[min_indices[:, 1]]
    n2_y = normal_ref_y[min_indices[:, 1]]

    # Step 3: Compute interpolation weights based on distances
    dist1 = np.hypot(x - x1, y - y1)
    dist2 = np.hypot(x - x2, y - y2)
    weight1 = dist2 / (dist1 + dist2)
    weight2 = dist1 / (dist1 + dist2)

    # Step 4: Interpolate s and d
    s_frenet = weight1 * s1 + weight2 * s2
    d_frenet_1 = (x - x1) * n1_x + (y - y1) * n1_y
    d_frenet_2 = (x - x2) * n2_x + (y - y2) * n2_y
    d_frenet = weight1 * d_frenet_1 + weight2 * d_frenet_2

    # Step 5: Compute the derivatives of s and d with respect to time
    ds = np.gradient(s_frenet, dt)
    dds = np.gradient(ds, dt)
    ddds = np.gradient(dds, dt)

    dd = np.gradient(d_frenet, dt)
    ddd = np.gradient(dd, dt)
    dddd = np.gradient(ddd, dt)

    # Step 6: Calculate curvature based on steering angle delta
    curvature = np.tan(delta) / wheelbase

    # Time array
    t = np.arange(0, len(x) * dt, dt)

    # Step 7: Create and return the FrenetTrajectory object
    return FrenetTrajectory(
        t=t,
        d=d_frenet,
        d_d=dd,
        d_dd=ddd,
        d_ddd=dddd,
        s=s_frenet,
        s_d=ds,
        s_dd=dds,
        s_ddd=ddds,
        x=x,
        y=y,
        yaw=psi,
        v=v,
        curv=curvature
    )



def C2F(s_frenet, d_frenet, states,  dt, wheelbase):

    ft_list = []
    # Time array
    t = np.arange(0, s_frenet.shape[1]* dt, dt)

    for i in range(states.shape[0]):

        x = states[i,:, 0]
        y = states[i,:, 1]
        psi = states[i,:, 2]
        v = states[i, :,3]
        delta = states[i,:, 4]

        # Step 1: Calculate the closest point on the reference spline
        s = s_frenet[i,:]
        d = d_frenet[i,:]
        

        # Find 1 2 and 3 derivatives of s and d with timestep dt
        ds = np.gradient(s, dt)
        dds = np.gradient(ds, dt)
        ddds = np.gradient(dds, dt)

        dd = np.gradient(d, dt)
        ddd = np.gradient(dd, dt)
        dddd = np.gradient(ddd, dt)

        # Find curvature with steering angle delta
        curvature = np.tan(delta) / wheelbase

        ft = FrenetTrajectory( t=t, 
                              d=d, 
                              d_d=dd, 
                              d_dd=ddd, 
                              d_ddd=dddd, 
                              s=s, s_d=ds, 
                              s_dd=dds, 
                              s_ddd=ddds, 
                              x=x, 
                              y=y, 
                              yaw=psi, 
                              v=v, 
                              curv=curvature)
        ft.traj_index = i
        ft_list.append(ft)
        
    return ft_list




        
    
