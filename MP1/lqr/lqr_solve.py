import numpy as np

def get_system(env):
    if env.unwrapped.spec.id == 'PendulumInvert-v1':
        system = PendulumBalance(dt=0.05)
    if env.unwrapped.spec.id == 'PendulumBalance-v1':
        system = PendulumBalance(dt=0.05)
    if env.unwrapped.spec.id == 'DoubleIntegrator-v1':
        system = DoubleIntegrator(dt=0.05)
    return system

class LQRControl(object):
    def __init__(self, env, state=None):
        self.system = get_system(env)
        A, B, Q, R = self.system.get_system()
        self.lqr = LQRSolver(A, B, Q, R, 200)
        self.lqr.solve()
        self.step = 0
        self.action_space = env.action_space

    def act(self, state):
        u = self.action_space.sample()
        # TODO: Once you finish implementing LQRSolver, uncomment the following
        # line to use it to control the system 
        u = self.lqr.get_control(state, self.step)
        self.step += 1
        return u

class LQRSolver(object):
    def __init__(self, A, B, Q, R, T):
        self.A, self.B, self.Q, self.R = A, B, Q, R
        self.T = T

    def solve(self):
        P = np.zeros((self.T + 1, self.A.shape[0], self.A.shape[0]))
        K = np.zeros((self.T, self.B.shape[1], self.A.shape[0]))
        P[self.T] = self.Q  

        for t in range(self.T - 1, -1, -1):
            nextP = P[t + 1]
            K[t] = np.linalg.inv(self.R + self.B.T @ nextP @ self.B) @ (self.B.T @ nextP @ self.A)
            P[t] = self.Q + self.A.T @ nextP @ self.A - self.A.T @ nextP @ self.B @ K[t]

        self.K = K
        self.P = P
        return K, P

    def get_control(self, x, i):
        return -self.K[i] @ x

class DoubleIntegrator(object):
    def __init__(self, dt):
        self.dt = dt
        None
    
    def get_system(self):
        return np.array([
            [1, self.dt],
            [0, 1]
        ]), np.array([
            [0],
            [self.dt]
        ]), np.array([[1, 0], [0, 1]]), np.array([[1]])

class PendulumBalance(object):
    def __init__(self, dt):
        self.dt = dt
    
    def get_system(self):
        return np.array([
            [1, self.dt],
            [15 * self.dt, 1]
        ]), np.array([
            [0],
            [3 * self.dt]
        ]), np.array([[1, 0], [0, 1]]), np.array([[1]])
