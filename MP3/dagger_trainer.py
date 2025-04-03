import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
def dagger_trainer(env,learner,query_expert,device):
    batch_size=64
    lr=1e-3
    discrete = True
    iterations = 10
    num_episodes_per_iter = 5
    max_steps = 200
    epochs=20

    learner.to(device)

    obs_list = []
    actions_list = []

    criterion = nn.CrossEntropyLoss() if discrete else nn.MSELoss()

    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)

    for iteration in range(iterations):
        for episode in range(num_episodes_per_iter):
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
                print('state shape', state.shape)
                # query expert for action
                action = query_expert(state).item()
                actions_list.append(action)
                print('expert action: ', action)

                with torch.no_grad():
                    action = action if iteration == 0 else learner.act(obs).item()
                
                # take step with expert action
                obs,reward,done,_,info = env.step(action)
                print('next state: ', env.unwrapped.state)
                print('image shape: ', obs.shape)
                steps += 1

            print(f"Episode {episode+1} of {num_episodes_per_iter}")

        learner.train()
        obs_tensor = torch.stack(obs_list, dim=0)
        actions_tensor = torch.tensor(actions_list, dtype=torch.long)

        dataset = TensorDataset(obs_tensor, actions_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch_obs, batch_actions in dataloader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)
                batch_actions = batch_actions.squeeze(-1)

                outputs = learner(batch_obs)
                outputs = outputs.squeeze(-1)

                loss = criterion(outputs, batch_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_obs.shape[0]

            avg_loss = total_loss / len(dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        learner.eval()
        print(f"Iteration {iteration+1} of {iterations}")
    
    return len(actions_list)