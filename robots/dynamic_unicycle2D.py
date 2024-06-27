import numpy as np

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


class DynamicUnicycle2D:
    
    def __init__(self, dt):
        '''
            X: [x, y, theta, v]
            U: [a, omega]
            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2
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

        
    def step(self, X, U): #Just holonomic X,T acceleration
        X = X + ( self.f(X) + self.g(X) @ U )*self.dt
        X[2,0] = angle_normalize(X[2,0])
        return X

    
    def stop(self):
        return np.array([0,0]).reshape(-1,1)
    

    def agent_barrier2(self, X, obs, robot_radius):
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius # obs radius + robot radius

        beta = 1.01

        # Current state
        h_k = np.linalg.norm(X[0:2] - obsX[0:2])**2 - beta * d_min**2

        # Next state (discrete time)
        X_next = self.step(X, self.nominal_input(X, obsX))
        h_k1 = np.linalg.norm(X_next[0:2] - obsX[0:2])**2 - beta * d_min**2

        # Second next state
        X_next2 = self.step(X_next, self.nominal_input(X_next, obsX))
        h_k2 = np.linalg.norm(X_next2[0:2] - obsX[0:2])**2 - beta * d_min**2

        h_dot = h_k1 - h_k
        h_ddot = h_k2 - 2 * h_k1 + h_k

        return h_k, h_dot, h_ddot





    def df_dx(self, X):
        return np.array([  
                    [0, 0, -X[3,0]*np.sin(X[2,0]), np.cos(X[2,0])],
                    [0, 0,  X[3,0]*np.cos(X[2,0]), np.sin(X[2,0])],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                    ])
        
    def agent_barrier(self, X, obs, robot_radius):
        obsX = obs[0:2]
        d_min = obs[2][0] + robot_radius # obs radius + robot radius

        beta = 1.01
    
        h = np.linalg.norm( X[0:2] - obsX[0:2] )**2 - beta*d_min**2   
        h_dot = 2 * (X[0:2] - obsX[0:2]).T @ ( self.f(X)[0:2]) # Lgh is zero => relative degree is 2

        df_dx = self.df_dx(X)
        dh_dot_dx = np.append( ( 2 * self.f(X)[0:2] ).T, np.array([[0,0]]), axis = 1 ) + 2 * ( X[0:2] - obsX[0:2] ).T @ df_dx[0:2,:]
        return h, h_dot, dh_dot_dx

    def nominal_input(self, X, G, d_min = 0.05, k_omega = 2.0, k_a = 1.0, k_v = 1.0):
        G = np.copy(G.reshape(-1,1)) # goal state
        max_v = 1.0

        distance = max(np.linalg.norm( X[0:2,0]-G[0:2,0] ) - d_min, 0.0) # don't need a min dist since it has accel
        theta_d = np.arctan2( G[1,0]-X[1,0], G[0,0]-X[0,0] )
        error_theta = angle_normalize( theta_d - X[2,0] )

        omega = k_omega * error_theta
        if abs(error_theta) > np.deg2rad(90):
            v = 0.0
        else:
            v = min(k_v * distance * np.cos(error_theta), max_v)
        #print("distance: ", distance, "v: ", v, "error_theta: ", error_theta)
        
        accel = k_a * ( v - X[3,0] )
        #print(f"CBF nominal acc: {accel}, omega:{omega}")
        return np.array([accel, omega]).reshape(-1,1)
        
        
        
    