import gymnasium as gym
import os
import numpy as np
from pathlib import Path
import pdb
from tqdm import tqdm
import envs
import logging
import time
import torch
import utils
from absl import app
from absl import flags
from policies import NNPolicy, CNNPolicy
from evaluation import val, test_model_in_env
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
from framestack import FrameStack
from train_bc import train_model

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes_val', 100, 'Number of episodes to evaluate.')
flags.DEFINE_integer('num_episodes_train', 250, 'Number of episodes to train.')
flags.DEFINE_integer('episode_len', 200, 'Length of each episode at test time.')
flags.DEFINE_string('env_name', 'CartPole-v2', 'Name of environment.')
flags.DEFINE_boolean('vis', False, 'To visualize or not.')
flags.DEFINE_boolean('vis_save', False, 'To save visualization or not')
flags.DEFINE_string('logdir', None, 'Directory to store loss plots, etc.')
flags.DEFINE_string('datadir', 'data/', 'Directory with expert data.')
flags.mark_flag_as_required('logdir')

def get_dims(env_name):
    if env_name == 'CartPole-v2':
        discrete = True
        return 4, 2, discrete 
    elif env_name == 'DoubleIntegrator-v1':
        discrete = False
        return 2, 1, discrete
    elif env_name == 'PendulumBalance-v1':
        discrete = False
        return 2, 1, discrete

def load_data():
    datadir = Path(FLAGS.datadir)
    
    # Load training data for training the policy.
    dt = utils.load_variables(datadir / f'{FLAGS.env_name}.pkl')
    dt['states'] = dt['states'][:FLAGS.num_episodes_train,:]
    dt['actions'] = dt['actions'][:FLAGS.num_episodes_train,:]
    return dt
    
def main(_):
    logdir = Path(FLAGS.logdir) / FLAGS.env_name
    logdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_num_threads(4)
    
    dt = load_data()
    # Setup your model.
    state_dim, action_dim, discrete = get_dims(FLAGS.env_name)
    model = NNPolicy(state_dim, [16, 32, 64], action_dim, discrete)

    #Train model
    from train_bc import train_model
    train_model(model, logdir,  dt['states'], dt['actions'], device, discrete)

    model = model.eval()

    # Setting up validation environments.
    val_envs = [gym.make(FLAGS.env_name) for _ in range(FLAGS.num_episodes_val)]
    [env.reset(seed=i+1000) for i, env in enumerate(val_envs)]
    val(model, device, val_envs, FLAGS.episode_len, False)
    [env.close() for env in val_envs]

    if FLAGS.vis or FLAGS.vis_save:
        env_vis = gym.make(FLAGS.env_name)
        state, g, gif, info = test_model_in_env(
            model, env_vis, FLAGS.episode_len, device, vis=FLAGS.vis, 
            vis_save=FLAGS.vis_save, visual=False)
        if FLAGS.vis_save:  
            gif[0].save(fp=f'{logdir}/vis-{env_vis.unwrapped.spec.id}.gif',
                        format='GIF', append_images=gif,
                        save_all=True, duration=50, loop=0)
        env_vis.close()

if __name__ == '__main__':
    app.run(main)
