from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F

num_steps_per_rollout = 5
num_updates = 10000
reset_every = 200
val_every = 10000

def log(writer, iteration, name, value, print_every=10, log_every=10):
    # A simple function to let you log progress to the console and tensorboard.
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

# Starting off from states in envs, rolls out num_steps_per_rollout for each
# environment using the policy in `model`. Returns rollouts in the form of
# states, actions, rewards and new states. Also returns the state the
# environments end up in after num_steps_per_rollout time steps.
def collect_rollouts(model, envs, states, num_steps, device):
    # print('states', states)
    states_tensor = torch.FloatTensor(states).to(device)
    states_batch = []
    actions_batch = []
    rewards_batch = []
    next_states_batch = []
    for step in range(num_steps):
        states_batch.append(states_tensor)
        with torch.no_grad():
            actions = model.act(states_tensor).cpu().numpy()
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
        
# Using the rollouts returned by collect_rollouts function, updates the actor
# and critic models. You will need to:
# 1a. Compute targets for the critic using the current critic in model.
# 1b. Compute loss for the critic, and optimize it.
# 2a. Compute returns, or estimate for returns, or advantages for updating the actor.
# 2b. Set up the appropriate loss function for actor, and optimize it.
# Function can return actor and critic loss, for plotting.


# Actor-Critic (without baseline)
def update_model_qvalue_ac(model, gamma, optim, rollouts, device, iteration, writer):
    states_batch, actions_batch, rewards_batch, next_states_batch = rollouts
    
    num_envs = states_batch[0].size(0)
    num_steps = len(states_batch)
    
    all_states = torch.cat(states_batch)
    all_actions = torch.cat(actions_batch)
    actor_logits, critic_values = model(all_states)
    critic_values = critic_values.squeeze(-1).view(num_steps, num_envs)
    
    with torch.no_grad():
        _, final_next_values = model(next_states_batch[-1])
        final_next_values = final_next_values.squeeze(-1)
    
    q_values = []
    
    for t in range(num_steps):
        with torch.no_grad():
            if t == num_steps - 1:
                next_values = final_next_values
            else:
                with torch.no_grad():
                    _, next_state_values = model(states_batch[t+1])
                    next_values = next_state_values.squeeze(-1)
                # next_values = critic_values[t+1]
        
        # Q-value estimate: r + γV(s')
        q_estimate = rewards_batch[t] + gamma * next_values.detach()
        q_values.append(q_estimate)
    
    q_values_tensor = torch.stack(q_values)
    
    action_dist = model.actor_to_distribution(actor_logits)
    
    log_probs = action_dist.log_prob(all_actions)
    log_probs = log_probs.view(num_steps, num_envs)
    
    actor_loss = -(log_probs * q_values_tensor.detach()).mean()
    
    critic_loss = F.mse_loss(critic_values, q_values_tensor.detach())
    
    total_loss = actor_loss + 0.5 * critic_loss
    
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    
    return actor_loss.item(), critic_loss.item()


# Advantage Actor-Critic (with state-dependent baseline) using TD error
def update_model_advantage_ac(model, gamma, optim, rollouts, device, iteration, writer):
    states_batch, actions_batch, rewards_batch, next_states_batch = rollouts
    
    num_envs = states_batch[0].size(0)
    num_steps = len(states_batch)
    
    all_states = torch.cat(states_batch)
    all_actions = torch.cat(actions_batch)
    all_logits, all_values = model(all_states)
    all_values = all_values.squeeze(-1).view(num_steps, num_envs)
    
    td_targets = []
    advantages = []
    
    with torch.no_grad():
        _, final_next_values = model(next_states_batch[-1])
        final_next_values = final_next_values.squeeze(-1)
    
    for t in range(num_steps):
        with torch.no_grad():
            if t == num_steps - 1:
                next_values = final_next_values
            else:
                with torch.no_grad():
                    _, next_state_values = model(states_batch[t+1])
                    next_values = next_state_values.squeeze(-1)
                # next_values = all_values[t+1]
        
        # TD target: r + γV(s')
        target = rewards_batch[t] + gamma * next_values.detach()
        td_targets.append(target)
        
        # Advantages: (r + γV(s')) - V(s)
        advantage = target - all_values[t]
        advantages.append(advantage)
    
    td_targets_tensor = torch.stack(td_targets)
    advantages_tensor = torch.stack(advantages)
    
    action_dist = model.actor_to_distribution(all_logits)
    log_probs = action_dist.log_prob(all_actions)
    log_probs = log_probs.view(num_steps, num_envs)
    
    actor_loss = -(log_probs * advantages_tensor.detach()).mean()
    
    critic_loss = F.mse_loss(all_values, td_targets_tensor.detach())
    
    total_loss = actor_loss + 0.5 * critic_loss
    
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    
    return actor_loss.item(), critic_loss.item()


def train_model_ac(model, envs, gamma, device, logdir, val_fn):
    model.to(device)
    train_writer = SummaryWriter(logdir / 'train')
    val_writer = SummaryWriter(logdir / 'val')
    

    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Resetting all environments to initialize the state.
    num_steps, total_samples = 0, 0
    states = [e.reset() for e in envs]
    
    for updates_i in range(num_updates):
        
        # Put model in training mode.
        model.train()
        
        # Collect rollouts using the policy.
        rollouts, states = collect_rollouts(model, envs, states, num_steps_per_rollout, device)
        num_steps += num_steps_per_rollout
        total_samples += num_steps_per_rollout*len(envs)

        # Use rollouts to update the policy and take gradient steps.
        actor_loss, critic_loss = update_model_qvalue_ac(model, gamma, optim, rollouts, 
                                               device, updates_i, train_writer)
        log(train_writer, updates_i, 'train-samples', total_samples, 100, 10)
        log(train_writer, updates_i, 'train-actor_loss', actor_loss, 100, 10)
        log(train_writer, updates_i, 'train-critic_loss', critic_loss, 100, 10)
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
            model.eval()
            mean_reward = val_fn(model, device)
            log(val_writer, total_samples, 'val-mean_reward', mean_reward, 1, 1)
            log(val_writer, total_samples, None, None, 1, 1)
            model.train()