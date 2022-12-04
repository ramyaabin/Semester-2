import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


##Creating a class named Linear_Qnet for initializing the linear neural network.

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

##The function forward is used to take the input(11 state vector) and pass it through the Neural network and apply relu activation function and give the output back i.e the next move of 1 x 3 vector size. In short, this is the prediction function that would be called by the agent.        
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

##The save function is used to save the trained model for future use.    
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

#Initialising QTrainer class
#   ∗ setting the learning rate for the optimizer.
#   * Gamma value that is the discount rate used in Bellman equation.
#   * initialising the Adam optimizer for updation of weight and biases.
#   * criterion is the Mean squared loss function.        

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

#Train_step function 
#   * As you know that PyTorch work only on tensors, so we are converting all the input to tensors.
#   * As discussed above we had a short memory training then we would only pass one value
#    of state, action, reward, move so we need to convert them into a vector, so we had used
#    unsqueezed function .
#   * Get the state from the model and calculate the new Q value using the below formula:
#                   Q_new = reward + gamma * max(next_predicted Qvalue)
#   * calculate the mean squared error between the new Q value and previous Q value and 
#   backpropagate that loss for weight updation. 
                
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (1, x)
        
# if only one parameter to train , then convert to tuple of shape (1, x)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = reward + gamma * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward() # backward propagation of loss

        self.optimizer.step()