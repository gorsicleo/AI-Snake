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
        
        print("Trying to load my brain")
        File_object = open("./model/records.txt","r")
        self.file_path='./model/71model.pth'
        if os.path.exists(self.file_path):
            self.load_state_dict(torch.load(self.file_path))
            print("I am smart now :)")
        else:
            print("Sorry i could not load my brain :(")


    def forward(self, x):
        # activation function -- activates tensor x 
        x = F.relu(self.linear_layer_one(x))
        x = self.linear_layer_two(x)
        return x 

    def save_state(self, number_of_games):
        print("Backuping my brain!")
        torch.save(self.state_dict(), "./model/"+str(number_of_games)+"model.pth")
        
        
class Reinforcment_Learner:
    def __init__(self, model, learning_rate, discount_rate):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()


    def learn_from_one_step(self, state, action, reward, next_state, is_finished):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            is_finished = (is_finished, )
        
        prediction_for_next_move = self.model(state)

        prediction_after_bellman = prediction_for_next_move.clone()
        for idx in range(len(is_finished)):
            Q_new = reward[idx]
            if not is_finished[idx]:
                Q_new = reward[idx] + self.discount_rate * torch.max(self.model(next_state[idx]))

            prediction_after_bellman[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(prediction_after_bellman, prediction_for_next_move)
        loss.backward()

        self.optimizer.step()
