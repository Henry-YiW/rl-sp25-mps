import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import random
from evaluation import val, test_model_in_env
from tqdm import tqdm
from framestack import FrameStack
import gymnasium as gym

def dagger_trainer(env,learner,query_expert,device):
    batch_size=64
    lr=1e-4
    discrete = True
    iterations = 10
    num_episodes_per_iter = 15
    max_steps = 200
    epochs=20
    max_dataset_size = 5000

    learner.to(device)

    obs_list = []
    actions_list = []

    criterion = nn.CrossEntropyLoss() if discrete else nn.MSELoss()

    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)

    for iteration in range(iterations):
        new_obs_list_length = 0
        episode = 0
        while new_obs_list_length < max_steps * num_episodes_per_iter:
            # get image observations
            obs = env.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                # transform to pytorch tensor
                obs = torch.from_numpy(obs).permute(2,0,1).to(device)
                obs_list.append(obs.clone().detach().cpu())
                # get state from environment
                state = torch.tensor(env.unwrapped.state).float().to(device)
                # query expert for action
                action = query_expert(state).item()
                actions_list.append(action)

                with torch.no_grad():
                    # print('obs.shape w', obs.shape)
                    action = action if iteration == 0 else learner.act(obs.unsqueeze(0)).item()
                
                # take step with expert action
                obs,reward,done,_,info = env.step(action)
                steps += 1
                new_obs_list_length += 1
            
            print(f"Episode {episode+1} of {num_episodes_per_iter}. New obs list length: {new_obs_list_length}")
            episode += 1

        learner.train()

        obs_list = obs_list[-max_dataset_size:]
        actions_list = actions_list[-max_dataset_size:]
        # obs_list_to_use = random.sample(obs_list, max_dataset_size) if len(obs_list) > max_dataset_size else obs_list
        # actions_list_to_use = random.sample(actions_list, max_dataset_size) if len(actions_list) > max_dataset_size else actions_list
        obs_tensor = torch.stack(obs_list, dim=0)
        actions_tensor = torch.tensor(actions_list, dtype=torch.int64)
        print('obs_tensor.shape before stack', obs_tensor.shape)
        print('actions_tensor.shape before stack', actions_tensor.shape)
        print('actions_tensor', actions_tensor)

        dataset = TensorDataset(obs_tensor, actions_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            training_steps = 0
            for batch_obs, batch_actions in dataloader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)
                if batch_actions.dim() > 1 or batch_actions.shape[0] > 1:
                    batch_actions = batch_actions.squeeze(-1)
                # print('batch_obs.shape', batch_obs.shape)
                # print('batch_actions.shape', batch_actions.shape)
                outputs = learner(batch_obs)
                outputs = outputs.squeeze(-1)
                # print('outputs.shape for loss', outputs.shape)
                # print('batch_actions.shape for loss', batch_actions.shape)
                loss = criterion(outputs, batch_actions)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(learner.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * batch_obs.shape[0]
                training_steps += 1
            avg_loss = total_loss / len(dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Training Steps: {training_steps}")

        learner.eval()
        num_episodes_per_iter = 10 
        print(f"Iteration {iteration+1} of {iterations}, Expert-Labeled Steps: {len(actions_list)}")
        # # Evaluation
        # val_envs = [FrameStack(gym.make('VisualCartPole-v2'),4) for _ in range(5)]
        # [env.reset(seed=i+1000) for i, env in enumerate(val_envs)]
        # val(learner, device, val_envs, 200, visual=True)
        # [env.close() for env in val_envs]
    
    return len(actions_list)