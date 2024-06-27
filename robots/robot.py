import numpy as np
import matplotlib.pyplot as plt
import do_mpc
from casadi import *

from shapely.geometry import Polygon, Point, LineString

"""
Created on June 21st, 2024
@author: Taekyung Kim

@description: 
This code implements a BaseRobot class for 2D robot simulation with unicycle dynamics.
It includes functionalities for robot movement, FoV visualization, obstacle detection, and safety area calculation (maximum braking distance).
The class supports both kinematic (Unicycle2D) and dynamic (DynamicUnicycle2D) unicycle models.
It incorporates Control Barrier Function (CBF) constraints for obstacle avoidance, which can be used as within a CBF-QP formulation.
The main function demonstrates the robot's movement towards a goal while avoiding an obstacle, visualizing the process in real-time.

@required-scripts: robots/unicycle2D.py, robots/dynamic_unicycle2D.py
"""


class BaseRobot:
    
    def __init__(self, X0, dt, ax, type='DynamicUnicycle2D', robot_id=0, data_generation=False, goal=None, obs=None):
        '''
        X0: initial state
        dt: simulation time step
        ax: plot axis handle
        '''
        
        self.type = type
        self.robot_id = robot_id
        self.data_generation = data_generation

        colors = plt.get_cmap('Pastel1').colors # color palette
        color = colors[robot_id % len(colors) + 1]

        self.test_type = 'cbf_qp' # or 'gatekeeper'
        if type == 'DynamicUnicycle2D':
            try:
                from dynamic_unicycle2D import DynamicUnicycle2D
            except ImportError:
                from robots.dynamic_unicycle2D import DynamicUnicycle2D
            self.robot = DynamicUnicycle2D(dt)
        else:
            raise ValueError("Invalid robot type")
        
        self.X = X0.reshape(-1,1)
        self.dt = dt

        self.robot_radius = 0.25 # including padding
        self.max_decel = 0.5 # [m/s^2]
        self.max_ang_decel = 0.5  # [rad/s^2]

        # MPC 
        self.goal = goal
        self.obs = obs
        self.horizon = 30
        self.gamma1 = 0.2
        self.gamma2 = 0.2
        self.Q = np.diag([50, 50, 0.01, 30]) # State cost matrix
        self.R = np.array([0.5, 0.5]) # Controls cost matrix
        self.model = self.create_model()
        self.mpc = self.create_mpc(self.model)
        self.simulator = self.define_simulator()
        self.estimator = do_mpc.estimator.StateFeedback(self.model)
        self.set_init_state()
      
        # FOV parameters
        self.fov_angle = np.deg2rad(70)  # [rad]
        self.cam_range = 3.0  # [m]

        self.U = np.array([0,0]).reshape(-1,1)
        
        # Plot handles
        self.vis_orient_len = 0.3
        # Robot's body represented as a scatter plot
        self.body = ax.scatter([], [], s=200, facecolors=color, edgecolors=color) #facecolors='none'
        # Robot's orientation axis represented as a line
        self.axis,  = ax.plot([self.X[0,0],self.X[0,0]+self.vis_orient_len*np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+self.vis_orient_len*np.sin(self.X[2,0])], color='r')
        # initialize the sensing_footprints with the initial robot location with radius 1 
        init_robot_position = Point(self.X[0, 0], self.X[1, 0]).buffer(1)
        self.positions = []  # List to store the positions for plotting


        if self.data_generation == False:
            # Store the unsafe points and scatter plot
            self.unsafe_points = []
            self.unsafe_points_handle = ax.scatter([],[],s=40,facecolors='r',edgecolors='r')
            # Initialize FOV line handle with placeholder data
            self.fov, = ax.plot([], [], 'k--')  # Unpack the tuple returned by plot
            # Initialize FOV fill handle with placeholder data
            self.fov_fill = ax.fill([], [], 'k', alpha=0.1)[0]  # Access the first element
            self.sensing_footprints_fill = ax.fill([], [], color=color, alpha=0.4)[0]  # Access the first element
            self.safety_area_fill = ax.fill([], [], 'r', alpha=0.3)[0]  

            self.detected_obs = None
            self.detected_points = []
            self.detected_obs_patch = ax.add_patch(plt.Circle((0, 0), 0, edgecolor='black',facecolor='orange', fill=True))
            self.detected_points_scatter = ax.scatter([],[],s=10,facecolors='r',edgecolors='r') #facecolors='none'
            self.safety_area = Polygon() # preserve the union of all the safety areas
            self.sensing_footprints = Polygon() # preserve the union of all the FOV triangles
            self.sensing_footprints = self.sensing_footprints.union(init_robot_position)
    
    
    def f(self):
        return self.robot.f(self.X)
    
    def g(self):
        return self.robot.g(self.X)
    
    def stop(self):
        return self.robot.stop()
    
    def agent_barrier(self, obs):
        return self.robot.agent_barrier(self.X, obs, self.robot_radius)

    def step(self, U):
        # wrap step function
        self.U = U.reshape(-1,1)
        self.X = self.robot.step(self.X, self.U)
        return self.X
    
    def render_plot(self):
        x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])
        self.body.set_offsets([x[0], x[1]])
        
        try:
            if len(self.unsafe_points) > 0 and self.test_type == 'gatekeeper':
                self.unsafe_points_handle.set_offsets(np.array(self.unsafe_points))
        except: pass
        
        self.axis.set_ydata([self.X[1,0],self.X[1,0]+self.vis_orient_len*np.sin(self.X[2,0])])
        self.axis.set_xdata( [self.X[0,0],self.X[0,0]+self.vis_orient_len*np.cos(self.X[2,0])] )

        if self.data_generation == False:
            # Calculate FOV points
            fov_left, fov_right = self.calculate_fov_points()

            # Define the points of the FOV triangle (including robot's robot_position)
            fov_x_points = [self.X[0, 0], fov_left[0], fov_right[0], self.X[0, 0]]  # Close the loop
            fov_y_points = [self.X[1, 0], fov_left[1], fov_right[1], self.X[1, 0]]

            # Update FOV line handle
            self.fov.set_data(fov_x_points, fov_y_points)  # Update with new data

            # Update FOV fill handle
            self.fov_fill.set_xy(np.array([fov_x_points, fov_y_points]).T)  # Update the vertices of the polygon

            if not self.sensing_footprints.is_empty:
                sensing_footprints_x, sensing_footprints_y = self.sensing_footprints.exterior.xy
                self.sensing_footprints_fill.set_xy(np.array([sensing_footprints_x, sensing_footprints_y]).T)  # Update the vertices of the polygon
                #ax.fill(sensing_footprints_x, sensing_footprints_y, alpha=0.1, fc='r', ec='none')
            if not self.safety_area.is_empty:
                if self.safety_area.geom_type == 'Polygon':
                    safety_x, safety_y = self.safety_area.exterior.xy
                elif self.safety_area.geom_type == 'MultiPolygon':
                    safety_x = [x for poly in self.safety_area.geoms for x in poly.exterior.xy[0]]
                    safety_y = [y for poly in self.safety_area.geoms for y in poly.exterior.xy[1]]
                self.safety_area_fill.set_xy(np.array([safety_x, safety_y]).T)
            if self.detected_obs is not None:
                self.detected_obs_patch.center = self.detected_obs[0], self.detected_obs[1]
                self.detected_obs_patch.set_radius(self.detected_obs[2])
            if len(self.detected_points) > 0:
                self.detected_points_scatter.set_offsets(np.array(self.detected_points))
    
    def update_sensing_footprints(self):
        fov_left, fov_right = self.calculate_fov_points()
        robot_position = (self.X[0, 0], self.X[1, 0])
        new_area = Polygon([robot_position, fov_left, fov_right])
    
        self.sensing_footprints = self.sensing_footprints.union(new_area)
        #self.sensing_footprints = self.sensing_footprints.simplify(0.1)

    def update_safety_area(self):
        theta = self.X[2, 0]  # Current heading angle in radians
        if self.type == 'Unicycle2D':
            v = self.U[0, 0]  # Linear velocity
        elif self.type == 'DynamicUnicycle2D':
            v = self.X[3, 0]
        omega = self.U[1, 0]  # Angular velocity
        
        if omega != 0:
            # Stopping times
            t_stop_linear = v / self.max_decel
            
            # Calculate the trajectory
            trajectory_points = [Point(self.X[0, 0], self.X[1, 0])]
            t = 0  # Start time
            while t <= t_stop_linear:
                v_current = max(v - self.max_decel * t, 0)
                if v_current == 0:
                    break  # Stop computing trajectory once v reaches 0
                omega_current = omega - np.sign(omega) * self.max_ang_decel * t
                if np.sign(omega_current) != np.sign(omega):  # If sign of omega changes, it has passed through 0
                    omega_current = 0  
                theta += omega_current * self.dt
                x = trajectory_points[-1].x + v_current * np.cos(theta) * self.dt
                y = trajectory_points[-1].y + v_current * np.sin(theta) * self.dt
                trajectory_points.append(Point(x, y))
                t += self.dt

            # Convert trajectory points to a LineString and buffer by robot radius
            if len(trajectory_points) >= 2:
                trajectory_line = LineString([(p.x, p.y) for p in trajectory_points])
                self.safety_area = trajectory_line.buffer(self.robot_radius)
            else:
                self.safety_area = Point(self.X[0, 0], self.X[1, 0]).buffer(self.robot_radius)
        else:    
            braking_distance = v**2 / (2 * self.max_decel)  # Braking distance
            # Straight motion
            front_center = (self.X[0, 0] + braking_distance * np.cos(theta),
                            self.X[1, 0] + braking_distance * np.sin(theta))
            self.safety_area = LineString([Point(self.X[0, 0], self.X[1, 0]), Point(front_center)]).buffer(self.robot_radius)
    
    def is_beyond_sensing_footprints(self):
        flag = not self.sensing_footprints.contains(self.safety_area)
        if flag:
            self.unsafe_points.append((self.X[0, 0], self.X[1, 0]))
        return flag
    
    def find_extreme_points(self, detected_points):
        # Convert points and robot position to numpy arrays for vectorized operations
        points = np.array(detected_points)
        robot_pos = self.X[0:2].reshape(-1)
        robot_yaw = self.X[2, 0]
        vectors_to_points = points - robot_pos
        robot_heading_vector = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
        angles = np.arctan2(vectors_to_points[:, 1], vectors_to_points[:, 0]) - np.arctan2(robot_heading_vector[1], robot_heading_vector[0])

        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        leftmost_index = np.argmin(angles)
        rightmost_index = np.argmax(angles)
        
        # Extract the most left and most right points
        leftmost_point = points[leftmost_index]
        rightmost_point = points[rightmost_index]
        
        return leftmost_point, rightmost_point

    def detect_unknown_obs(self, unknown_obs, obs_margin=0.05):
        if unknown_obs is None:
            return []
        #detected_obs = []
        self.detected_points = []

        # sort unknown_obs by distance to the robot, closest first
        sorted_unknown_obs = sorted(unknown_obs, key=lambda obs: np.linalg.norm(np.array(obs[0:2]) - self.X[0:2].reshape(-1)))
        for obs in sorted_unknown_obs:
            obs_circle = Point(obs[0], obs[1]).buffer(obs[2]-obs_margin)
            intersected_area = self.sensing_footprints.intersection(obs_circle)

            # Check each point on the intersected area's exterior

            points = []
            if intersected_area.geom_type == 'Polygon':
                for point in intersected_area.exterior.coords:
                    points.append(point)
            elif intersected_area.geom_type == 'MultiPolygon':
                for poly in intersected_area.geoms:
                    for point in poly.exterior.coords:
                        points.append(point)

            for point in points:
                point_obj = Point(point)
                # Line from robot's position to the current point
                line_to_point = LineString([Point(self.X[0, 0], self.X[1, 0]), point_obj])

                # Check if the line intersects with the obstacle (excluding the endpoints)
                # only consider the front side of the obstacle
                if not line_to_point.crosses(obs_circle):
                    self.detected_points.append(point)
                
            if len(self.detected_points) > 0:
                break

        if len(self.detected_points) == 0:
            self.detected_obs = None
            return []
        leftmost_most, rightmost_point = self.find_extreme_points(self.detected_points)

        # Calculate the center and radius of the circle
        center = (leftmost_most + rightmost_point) / 2
        radius = np.linalg.norm(rightmost_point - leftmost_most) / 2

        self.detected_obs = [center[0], center[1], radius]
        return self.detected_obs
    
    def calculate_fov_points(self):
        """
        Calculate the left and right boundary points of the robot's FOV.
        """
        # Calculate left and right boundary angles
        angle_left = self.X[2,0] - self.fov_angle / 2
        angle_right = self.X[2,0] + self.fov_angle / 2

        # Calculate points at the boundary of the FOV
        fov_left = (self.X[0,0] + self.cam_range * np.cos(angle_left), self.X[1,0] + self.cam_range * np.sin(angle_left))
        fov_right = (self.X[0,0] + self.cam_range * np.cos(angle_right), self.X[1,0] + self.cam_range * np.sin(angle_right))

        return fov_left, fov_right
      

    def create_model(self):
        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)

        # States (_x[0] = x pos, _x[1] = y pos, _x[2] = theta _x[3] = velocity)
        n_states = 4
        _x = model.set_variable(var_type='_x', var_name='x', shape=(n_states, 1)) 
        
        # Inputs (u1 = linear acceleration, u2 = angular velocity)
        n_controls = 2
        _u = model.set_variable(var_type='_u', var_name='u', shape=(n_controls, 1))

        _p1 = model.set_variable(var_type='_tvp', var_name='gamma1')
        _p2 = model.set_variable(var_type='_tvp', var_name='gamma2')

        # State Space equations for Dynamic unicycle model
        x_next = _x[0] + _x[3] * cos(_x[2]) * self.dt
        y_next = _x[1] + _x[3] * sin(_x[2]) * self.dt
        theta_next = _x[2] + _u[1] * self.dt
        v_next = _x[3] + _u[0] * self.dt

        # Update model RHS
        model.set_rhs('x', vertcat(x_next, y_next, theta_next, v_next))

        # stage and terminal cost of the control problem
        model, cost_expr = self.get_cost_expression(model)
        model.set_expression(expr_name='cost', expr=cost_expr)
        
        model.setup()
        return model

    def create_mpc(self, model):
        mpc = do_mpc.controller.MPC(model)
        mpc.settings.supress_ipopt_output()

        setup_mpc = {'n_robust': 0,  # Robust horizon
                     'n_horizon': self.horizon,
                     't_step': self.dt,
                     'state_discretization': 'discrete',
                     'store_full_solution': True,
                     }
        mpc.set_param(**setup_mpc)

        # Configure objective function
        mterm = self.model.aux['cost'] # Terminal cost
        lterm = self.model.aux['cost'] # Stage cost
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u=self.R) # Input penalty (R diagonal matrix in objective fun)

        # Bounds
        max_u = np.array([self.max_decel, self.max_ang_decel])
        mpc.bounds['lower', '_u', 'u'] = -max_u
        mpc.bounds['upper', '_u', 'u'] = max_u

        # Set gamma1 and gamma2 as TVP
        mpc = self.set_tvp_for_mpc(mpc)
        
        # Add CBF constraints
        if self.obs is not None:
            mpc = self.get_cbf_constraints(mpc) 

        mpc.setup()
        mpc.settings.supress_ipopt_output()
        return mpc

    def get_cost_expression(self, model):
        """Defines the objective function wrt the state cost"""
        X = SX.zeros(4, 1)
        X[0] = model.x['x'][0] - self.goal[0]  # x position error
        X[1] = model.x['x'][1] - self.goal[1]  # y position error
        X[2] = model.x['x'][2] - self.goal[2]  # theta orientation error
        X[3] = model.x['x'][3] - self.goal[3]  # v (velocity) error

        cost_expression = transpose(X) @ self.Q @ X
        
        return model, cost_expression
    
    def get_cbf_constraints(self, mpc):
        # Get state vector x_{t+k+1}
        x_k = self.model.x['x']  # Current state [0] xpos, [1] ypos, [2] orien, [3] velocity
        u_k = self.model.u['u']  # Current control input [0] acc, [1] omega

        #p_template = self.simulator.get_p_template()
        gamma1 = self.model.tvp['gamma1']
        gamma2 = self.model.tvp['gamma2']
        
        # Dynamics equations for the next states
        x_next = x_k[0] + x_k[3] * cos(x_k[2]) * self.dt
        y_next = x_k[1] + x_k[3] * sin(x_k[2]) * self.dt
        theta_next = x_k[2] + u_k[1] * self.dt
        v_next = x_k[3] + u_k[0] * self.dt

        x_next2 = x_next + v_next * cos(theta_next) * self.dt
        y_next2 = y_next + v_next * sin(theta_next) * self.dt
        theta_next2 = theta_next + u_k[1] * self.dt
        v_next2 = v_next + u_k[0] * self.dt
        
        # Next states
        x_k1 = vertcat(x_next, y_next, theta_next, v_next)
        x_k2 = vertcat(x_next2, y_next2, theta_next2, v_next2)
        
        cbf_constraints = []
        
        h_k2 = self.h(x_k2, self.obs)
        h_k1 = self.h(x_k1, self.obs)
        h_k = self.h(self.model.x['x'], self.obs)
        h_ddot = h_k2 - 2 * h_k1 + h_k
        h_dot = h_k1 - h_k
        cbf_2nd_order = h_ddot + (gamma1 + gamma2) * h_dot + (gamma1 * gamma2) * h_k
        cbf_constraints.append(-cbf_2nd_order)
            
        i = 0
        for cbc in cbf_constraints:
            mpc.set_nl_cons('cbf_constraint'+str(i), cbc, ub=0)
            i += 1
        
        return mpc        
    
    def h(self, x, obstacle):
        """Computes the Control Barrier Function"""
        x_obs, y_obs, r_obs = obstacle
        h = (x[0] - x_obs)**2 + (x[1] - y_obs)**2 - (self.robot_radius + r_obs)**2
        return h
    
    def set_tvp_for_mpc(self, mpc):
        tvp_struct_mpc = mpc.get_tvp_template()
        def tvp_fun_mpc(t_now):
            tvp_struct_mpc['_tvp', :, "gamma1"] = self.gamma1
            tvp_struct_mpc['_tvp', :, "gamma2"] = self.gamma2
            return tvp_struct_mpc
        mpc.set_tvp_fun(tvp_fun_mpc)
        return mpc
      
    def define_simulator(self):
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=self.dt)
        tvp_template = simulator.get_tvp_template()

        def tvp_fun(t_now):
            return tvp_template
        simulator.set_tvp_fun(tvp_fun)

        simulator.setup()

        return simulator

    def set_init_state(self):
        """Sets the initial state in all components."""
        self.mpc.x0 = self.X
        self.simulator.x0 = self.X
        self.estimator.x0 = self.X
        self.mpc.set_initial_guess()
      
    # def run_simulation(self):
    #     for k in range(self.sim_time):
    #         u0 = self.mpc.make_step(x0)
    #         y_next = self.simulator.make_step(u0)
    #         x0 = self.estimator.make_step(y_next)
    #         self.X = x0.reshape(-1, 1)
    #         self.render_plot()

    #         fig.canvas.draw()
    #         plt.pause(0.01)
      
    def nominal_input(self, goal, d_min=0.05):
        return self.robot.nominal_input(self.X, goal, d_min)

    
        
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import cvxpy as cp

    type = 'DynamicUnicycle2D'

    plt.ion()
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)

    dt = 0.02
    tf = 20
    num_steps = int(tf / dt)

    obs = np.array([0.5, 0.3, 0.5]).reshape(-1, 1)
    goal = np.array([2, 0.5, 0, 0])
    ax.scatter(goal[0], goal[1], c='g')
    circ = plt.Circle((obs[0, 0], obs[1, 0]), obs[2, 0], linewidth=1, edgecolor='k', facecolor='k')
    ax.add_patch(circ)
    
    robot = BaseRobot(np.array([-1, -1, np.pi / 4, 0.0]).reshape(-1, 1), dt, ax, type=type, data_generation=True, goal=goal, obs=obs)

    mpc = robot.mpc
    simulator = do_mpc.simulator.Simulator(robot.model)
    simulator.set_param(t_step=dt)
    tvp_template = simulator.get_tvp_template()

    def tvp_fun(t_now):
        return tvp_template
    simulator.set_tvp_fun(tvp_fun)
    
    simulator.setup()

    estimator = do_mpc.estimator.StateFeedback(robot.model)

    x0 = robot.X.flatten()
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0
    mpc.set_initial_guess()

    for i in range(num_steps):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)
        
        robot.X = x0.reshape(-1, 1)
        robot.render_plot()

        fig.canvas.draw()
        plt.pause(0.01)
        