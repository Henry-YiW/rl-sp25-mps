import numpy as np
from gymnasium.envs.registration import register
from .cartpole import CartPoleEnv
from .pendulum import PendulumEnv 
from .double_integrator import DoubleIntegratorEnv 

register(
    id='CartPole-v2',
    entry_point='envs:CartPoleEnv',
)

register(
    id='VisualCartPole-v2',
    entry_point='envs:CartPoleEnv',
    kwargs={'visual': True}
)

register(
    id='DoubleIntegrator-v1',
    entry_point='envs:DoubleIntegratorEnv',
    max_episode_steps=200,
    kwargs={'max_acc': np.inf, 'init_y': 5., 'init_v': 3.},
)

register(
    id='PendulumBalance-v1',
    entry_point='envs:PendulumEnv',
    max_episode_steps=200,
    kwargs={'init_theta': 0.2, 'init_thetadot': 0.0, 'max_torque': 1.0, 'noise': 0.} 
)
