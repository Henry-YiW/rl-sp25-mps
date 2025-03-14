import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm

num_steps_per_rollout = 5 
num_updates = 10000
reset_every = 200
val_every = 10000

replay_buffer_size = 1000000
q_target_update_every = 50
q_batch_size = 256
q_num_steps = 1
    
def log(writer, iteration, name, value, print_every=10, log_every=10):
    # A simple function to let you log progress to the console and tensorboard.
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

# Implement a replay buffer class that a) stores rollouts as they
# come along, overwriting older rollouts as needed, and b) allows random
# sampling of transition quadruples for training of the Q-networks.
class ReplayBuffer(object):
    def __init__(self, size, state_dim, action_dim):
        self.size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize buffer arrays
        self.states = torch.zeros((size, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((size, 1), dtype=torch.long)
        self.rewards = torch.zeros(size, dtype=torch.float32)
        self.next_states = torch.zeros((size, state_dim), dtype=torch.float32)
        
        self.insert_position = 0  # Points to the next slot to insert data
        self.current_size = 0  # Tracks the current size of the buffer

    def insert(self, rollouts):
        # Iterate through the rollouts and insert them into the buffer
        states_batch, actions_batch, rewards_batch, next_states_batch = rollouts
        for states, actions, rewards, next_states in zip(states_batch, actions_batch, rewards_batch, next_states_batch):
            for state, action, reward, next_state in zip(states, actions, rewards, next_states):
                self.states[self.insert_position] = state
                self.actions[self.insert_position] = action
                self.rewards[self.insert_position] = reward
                self.next_states[self.insert_position] = next_state
                
                # Update pointer and size
                self.insert_position = (self.insert_position + 1) % self.size
                self.current_size = min(self.current_size + 1, self.size)

    def sample_batch(self, batch_size):
        # Sample batch_size indices from the buffer
        indices = torch.randint(0, self.current_size, size=(batch_size,))
        # print('self.states', self.states.shape)
        # print('self.actions', self.actions.shape)
        # print('self.rewards', self.rewards.shape)
        # print('self.next_states', self.next_states.shape)
        # Extract the data using the indices
        states = torch.FloatTensor(self.states[indices])
        actions = torch.LongTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        
        return (states, actions, rewards, next_states)

# Starting off from states in envs, rolls out num_steps_per_rollout for each
# environment using the policy in `model`. Returns rollouts in the form of
# states, actions, rewards and new states. Also returns the state the
# environments end up in after num_steps_per_rollout time steps.
def collect_rollouts(models, envs, states, num_steps_per_rollout, epsilon, device):
    # print('states', states)
    states_tensor = torch.FloatTensor(states).to(device)
    # Roll out num_steps_per_rollout steps
    states_batch = []
    actions_batch = []
    rewards_batch = []
    next_states_batch = []
    for step in range(num_steps_per_rollout):
        states_batch.append(states_tensor)
        # Get action using epsilon-greedy policy
        with torch.no_grad():
            actions = models[0].act(states_tensor, epsilon=epsilon).cpu().numpy()
        # print('actions', actions)
        actions_batch.append(torch.LongTensor(actions))
        new_states = []
        rewards = []
        for env, action in zip(envs, actions):
            new_state, reward, terminated, info = env.step(action)
            # print('new_state', new_state)
            new_states.append(torch.FloatTensor(new_state))
            # print('reward', reward)
            rewards.append(torch.tensor(reward))
            # print('terminated', terminated)
            # print('info', info)
        next_states_batch.append(torch.stack(new_states))
        rewards_batch.append(torch.stack(rewards))
        states_tensor = next_states_batch[-1]
    
    return (states_batch, actions_batch, rewards_batch, next_states_batch), states_tensor
   
# Function to train the Q function. Samples q_num_steps batches of size
# q_batch_size from the replay buffer, runs them through the target network to
# obtain target values for the model to regress to. Takes optimization steps to
# do so. Returns the bellman_error for plotting.
def update_model(replay_buffer, models, targets, optim, gamma, action_dim,
                 q_batch_size, q_num_steps):
    total_bellman_error = 0.0
    device = next(models[0].parameters()).device
    
    # Check if we have enough samples in the buffer
    if replay_buffer.current_size < q_batch_size:
        return total_bellman_error
    
    # Perform q_num_steps updates
    for _ in range(q_num_steps):
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states = replay_buffer.sample_batch(q_batch_size)
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        # print(states.shape, actions.shape, rewards.shape, next_states.shape)
        # Compute Q-values for current states and actions
        current_q_values = models[0](states).gather(1, actions).squeeze(-1)
        # print('actions', actions)
        
        # Compute target Q-values
        with torch.no_grad():
            # Get the maximum Q-value across all actions for the next states
            next_q_values = targets[0](next_states).max(dim=1)[0]
            
            # Compute the target Q-value using the Bellman equation
            target_q_values = rewards + gamma * next_q_values
        
        # Compute the loss (MSE between current and target Q-values)
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the model
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        # Accumulate the bellman error for reporting
        total_bellman_error += loss.item()
    
    # Return the average bellman error
    return total_bellman_error / q_num_steps

def train_model_dqn(models, targets, state_dim, action_dim, envs, gamma, device, logdir, val_fn):
    train_writer = SummaryWriter(logdir / 'train')
    val_writer = SummaryWriter(logdir / 'val')
    
    # Set up optimizer for DQN
    optim = torch.optim.Adam(models[0].parameters(), lr=0.001)

    # Set up the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, state_dim, action_dim)

    # Resetting all environments to initialize the state.
    num_steps, total_samples = 0, 0
    states = [e.reset() for e in envs]
    
    for updates_i in range(num_updates):
        # Annealing epsilon for exploration
        epsilon_start = 1.0
        epsilon_end = 0.05
        # Linear annealing over 50% of total updates
        epsilon_decay = num_updates * 0.5
        epsilon = max(epsilon_end, epsilon_start - (updates_i / epsilon_decay) * (epsilon_start - epsilon_end))

        # Put model in training mode.
        [m.train() for m in models]

        if np.mod(updates_i, q_target_update_every) == 0:
            # Copy the model parameters to the target network
            for target, model in zip(targets, models):
                target.load_state_dict(model.state_dict())
        
        # Collect rollouts using the policy.
        rollouts, states = collect_rollouts(models, envs, states, num_steps_per_rollout, epsilon, device)
        # print('num_steps_per_rollout', num_steps_per_rollout)
        # print('rollouts', len(rollouts[0]), len(rollouts[1]), len(rollouts[2]), len(rollouts[3]))
        num_steps += num_steps_per_rollout
        total_samples += num_steps_per_rollout*len(envs)
        
        # Push rollouts into the replay buffer.
        replay_buffer.insert(rollouts)

        # print('replay_buffer.current_size', replay_buffer.current_size)

        # Use replay buffer to update the policy and take gradient steps.
        bellman_error = update_model(replay_buffer, models, targets, optim,
                                     gamma, action_dim, q_batch_size,
                                     q_num_steps)
        log(train_writer, updates_i, 'train-samples', total_samples, 100, 10)
        log(train_writer, updates_i, 'train-bellman-error', bellman_error, 100, 10)
        log(train_writer, updates_i, 'train-epsilon', epsilon, 100, 10)
        log(train_writer, updates_i, None, None, 100, 10)

        # We are solving a continuing MDP which never returns a done signal. We
        # are going to manully reset the environment every few time steps. To
        # track progress on the training envirnments you can maintain the
        # returns on the training environments, and log or print it out when
        # you reset the environments.
        if num_steps >= reset_every:
            states = [e.reset() for e in envs]
            num_steps = 0
        
        # Every once in a while run the policy on the environment in the
        # validation set. We will use this to plot the learning curve as a
        # function of the number of samples.
        cross_boundary = total_samples // val_every > \
            (total_samples - len(envs)*num_steps_per_rollout) // val_every
        if cross_boundary:
            models[0].eval()
            mean_reward = val_fn(models[0], device)
            log(val_writer, total_samples, 'val-mean_reward', mean_reward, 1, 1)
            log(val_writer, total_samples, None, None, 1, 1)
            models[0].train()
