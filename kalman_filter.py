# %%
import math
import numpy as np
import matplotlib.pyplot as plt

# %%

class KalmanFilter:
    def __init__(self, _state, _ip):
        '''
        The matrices of Kalman filter:
        System:
        next_state = A*prev_state + B*input
        output = C*next+state + D*input
        '''
        self.state = _state
        self.ip = _ip
        self.output = None
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.SigmaState=None
        self.SigmaOutput = None
        self.P = None
    
    def set_kalman_matrices(self, A, B, C, D, P, SigmaState, SigmaOutput):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.SigmaState= SigmaState
        self.SigmaOutput = SigmaOutput
        self.P = P
    
    def predict(self, state):
        self.state = next_state(self.state, self.ip) + np.random.normal([1,1,0], [0.4, 0.4, 0.001], 3)
        self.P = self.A*self.P*np.transpose(A) + self.SigmaState
        self.output = C.dot(self.state)
        return self.state

    def estimate(self, measurement):
        Kalman_gain = self.P*np.transpose(C) * np.linalg.pinv((C*self.P*np.transpose(C) + SigmaOutput))
        gain_factor = Kalman_gain.dot((measurement - self.output))
        self.state = self.state + gain_factor
        self.P = (np.diag([1,1,1]) - Kalman_gain*self.C)*self.P
        self.output = C.dot(self.state)

        return self.state, self.output 

# %%
""" Plot """
import matplotlib.pyplot as plt
import numpy as np

def plot_results(odometry, gps, perfect_world, kal_out):
    plt.figure(1)
    plt.plot(np.array(odometry)[:,0], np.array(odometry)[:,1], label='odometry')
    plt.plot(np.array(gps)[:,0], np.array(gps)[:,1], label='GPS')
    plt.plot(np.array(kal_out)[:,0], np.array(kal_out)[:,1], label='Filtered signal')
    plt.plot(np.array(perfect_world)[:,0], np.array(perfect_world)[:,1], label='Perfect World')
    plt.legend()
    plt.show()

# %%
""" State Evolution """

'''
Defining the physics of the system. In our case, a 2D robot model
states = [x, y, theta], inputs = [v, omega]
x_next = x + v * cos(theta)
y_next = y + v * sin(theta)
theta_next = theta + omega
'''

def next_state(prev_state, ip):
    x_next = prev_state[0] + ip[0] * math.acos(prev_state[2])
    y_next = prev_state[1] + ip[0] * math.acos(prev_state[2])
    theta_next = prev_state[2] + ip[1]

    return np.transpose(np.array([x_next, y_next, theta_next]))

# %% 
""" Simulate Sensor data """

def mock_odo_gps_data(state, ip):
    odo = []
    gps = []
    perfect_world = []

    for _ in range(50):
        state = next_state(state, ip) # state update
        perfect_world.append(state)
        odo.append(state + np.random.normal([0,0,0], [0.1, 0.1, 0.001], 3)) # odo estimate is bad
        gps.append(state + np.random.normal([0,0,0], [0.05, 0.05, 0], 3)) # gps estimate is much better

    return odo, gps, perfect_world

def generate_cirular_traj_ips():
    print("Test")

# %%
if __name__ == "__main__":

    state = np.transpose(np.array([0,0,0.002]))
    ip = np.array([1,0.01])
    output = None
    
    # Covariance matrices
    sigmax = 1
    sigmay = 1
    sigmatheta = 0.1

    # Tuning parameters
    P = np.diag([sigmax**2, sigmay**2, sigmatheta**2])
    SigmaState = np.diag([0.01, 0.01, 0.007]) #uncertainity in state
    SigmaOutput = np.diag([0.1,0.1,0]) #uncertainity in measurement - How much you believe in ur measurement

    odometry, gps, perfect_world = mock_odo_gps_data(state, ip)
    kal_out = []
    odo_out = []

    kal_filter = KalmanFilter(state, ip)

    for idx, gps_data in enumerate(gps):
        # from the physics of the system we derive A, B, C, D
        A = np.array(np.identity(3))
        B = np.array([[math.acos(state[2]), 0], [math.asin(state[2]), 0], [0,1]])
        C = np.array(np.identity(3))
        C[2][2] = 0
        D = np.zeros((2,2))

        kal_filter.set_kalman_matrices(A,B,C,D,P,SigmaState,SigmaOutput)
        odo_out.append(kal_filter.predict(state))
        state, output = kal_filter.estimate(gps_data)
        kal_out.append(output)

    plot_results(odo_out, gps, perfect_world, kal_out)

# %%
