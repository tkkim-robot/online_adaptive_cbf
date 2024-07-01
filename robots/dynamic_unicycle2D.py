import numpy as np
import casadi
from casadi import *

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


class DynamicUnicycle2D:
    
    def __init__(self, dt):
        '''
            X: [x, y, theta, v]
            U: [a, omega]
            cbf: h(x) = ||x-x_obs||^2 - d_min^2
            relative degree: 2
        '''
        self.type = 'DynamicUnicycle'   
        self.dt = dt     

    def f(self, X):
        return np.array([X[3,0]*np.cos(X[2,0]),
                         X[3,0]*np.sin(X[2,0]),
                         0,
                         0]).reshape(-1,1)
    
    def g(self, X):
        return np.array([ [0, 0],[0, 0], [0, 1], [1, 0] ])

    def f_casadi(self, X):
        return casadi.vertcat(
            X[3,0] * casadi.cos(X[2,0]),
            X[3,0] * casadi.sin(X[2,0]),
            0,
            0
        )

    def g_casadi(self, X):
        return casadi.DM([
            [0, 0], 
            [0, 0], 
            [0, 1], 
            [1, 0]
        ])

    def step(self, X, U): #Just holonomic X,T acceleration
        X = X + ( self.f(X) + self.g(X) @ U )*self.dt
        # X[2,0] = angle_normalize(X[2,0])
        return X
    
    def stop(self):
        return np.array([0,0]).reshape(-1,1)

    def agent_barrier_casadi(self, x_k, u_k, gamma1, gamma2, dt, robot_radius, obs):
        """Computes the Discrete Time High Order CBF"""
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)
        x_k2 = self.step(x_k1, u_k)

        def h(x, robot_radius, obstacle):
            """Computes the Control Barrier Function"""
            x_obs = obstacle[0]
            y_obs = obstacle[1]
            r_obs = obstacle[2]
            h = (x[0, 0] - x_obs)**2 + (x[1, 0] - y_obs)**2 - (robot_radius + r_obs)**2
            return h

        h_k2 = h(x_k2, robot_radius, obs)
        h_k1 = h(x_k1, robot_radius, obs)
        h_k = h(x_k, robot_radius, obs)
        h_ddot = h_k2 - 2 * h_k1 + h_k
        h_dot = h_k1 - h_k
        hocbf_2nd_order = h_ddot + (gamma1 + gamma2) * h_dot + (gamma1 * gamma2) * h_k

        return hocbf_2nd_order

        
        
    