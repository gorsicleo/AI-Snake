import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Neural_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_layer_one = nn.Linear(input_size, hidden_size)
        self.linear_layer_two = nn.Linear(hidden_size, output_size)
        file_name = os.path.join('./', 'saved_state.pth')
        self.load_state_dict(torch.load(file_name))

    def forward(self, tensor):
        # activation function
        tensor = F.relu(self.linear_layer_one(tensor))
        tensor = self.linear_layer_two(tensor)
        return tensor 

    def save_state(self, file_name='saved_state.pth'):
        file_name = os.path.join('./', file_name)
        torch.save_state(self.state_dict(), file_name)
        
class QTrainer:
    def __init__(self, model, lr, discount_rate):
        self.lr = lr
        self.discount_rate = discount_rate
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, is_finished):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            is_finished = (is_finished, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(is_finished)):
            Q_new = reward[idx]
            if not is_finished[idx]:
                Q_new = reward[idx] + self.discount_rate * \
                    torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not is_finished
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
