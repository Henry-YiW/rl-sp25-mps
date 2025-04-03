import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

def train_model(model, logdir, states,actions, device, discrete):
    batch_size=64
    epochs=50
    lr=1e-3
    model.to(device)
    model.train()
    
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long if discrete else torch.float32)
    states_tensor = states_tensor.reshape(-1, states_tensor.shape[-1])
    actions_tensor = actions_tensor.reshape(-1, actions_tensor.shape[-1])

    dataset = TensorDataset(states_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss() if discrete else nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_actions = batch_actions.squeeze(-1)

            outputs = model(batch_states)
            outputs = outputs.squeeze(-1)
            
            loss = criterion(outputs, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_states.shape[0]

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Optional: Save model checkpoint or log to TensorBoard using `logdir`

    print("Training completed.")