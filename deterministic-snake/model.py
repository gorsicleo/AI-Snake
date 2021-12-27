from numpy import dtype
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time

class Neural_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_layer_one = nn.Linear(input_size, hidden_size)
        self.linear_layer_two = nn.Linear(hidden_size, output_size)
        
        print("Trying to load my brain")
        File_object = open("./model/records.txt","r")
        self.file_path='./model/-171model.pth'
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
        
        
class Deterministic_AI_Learner:
    def __init__(self, model, learning_rate, discount_rate):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()


    def learn_from_one_step(self, state, deterministic_action):
        state = torch.tensor(state, dtype=torch.float)
        deterministic_action = torch.tensor(deterministic_action, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            deterministic_action = torch.unsqueeze(deterministic_action,0)
        
        prediction_for_next_move = self.model(state)
    
        self.optimizer.zero_grad()
        #print("------------------model print------------------")
        #print(deterministic_action)
        #print(prediction_for_next_move)
        #print("------------------end model print------------------")
        loss = self.criterion(deterministic_action, prediction_for_next_move)
        loss.backward()

        self.optimizer.step()
