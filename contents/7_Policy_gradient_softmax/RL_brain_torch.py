import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
# from maze_env import Maze

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
# env = Maze()
# N_ACTIONS = env.n_actions
# N_STATES = env.n_features

class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        actions_prob = F.softmax(actions_value, dim=1)
        return actions_prob
      
net = Net(4, 3)
actions_value = net.forward(Variable(torch.randn(2, 4)))
print(actions_value)

class PolicyGradientTorch(object):
    def __init__(self, N_ACTIONS, N_STATES) -> None:
        super().__init__()
        self.N_ACTIONS, self.N_STATES = N_ACTIONS, N_STATES
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.cost_his = []
    
    def choose_action(self, x):
        
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.N_ACTIONS)
        return action
           
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        N_ACTIONS, N_STATES = self.N_ACTIONS, self.N_STATES
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            print("Replace Learning: ", self.learn_step_counter)
            self.target_net.load_state_dict(self.eval_net.state_dict()) ## TODO: WHY
        self.learn_step_counter += 1
        
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)
        self.cost_his.append(float(loss.data.numpy()))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()