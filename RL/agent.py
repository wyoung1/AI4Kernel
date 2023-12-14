from collections import defaultdict, deque
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

cpu_clock_list = [345600,499200,652800,806400,960000,1113600,
                  1267200,1420800,1574400,1728000,1881600,2035200]
gpu_clock_list = [114750000,216750000,318750000,420750000,
                  522750000,624750000,726750000,828750000,
                  930750000,1032750000,1134750000,1236750000,1300500000]

dir_thermal='/sys/devices/virtual/thermal'
dir_power='/sys/bus/i2c/drivers/ina3221x'
dir_power1='/sys/kernel/debug/bpmp/debug/regulator'

DEFAULT_PROTOCOL = 0
PORT = 8702
experiment_time=2000 #14100
clock_change_time=30
cpu_power_limit=1000
gpu_power_limit=1600
action_space=9
target_fps=30
target_temp=50
beta=2 #8

class Model(nn.Module):
    def __init__(self, state_size, action_size, learning_rate):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(state_size, 6)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(6, 6)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(6, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x

    def compute_loss(self, predicted, target):
        return self.criterion(predicted, target)

class DQNAgent:
	def __init__(self, state_size, action_size):
		self.load_model = False
		self.training = 0
		self.state_size = state_size
		self.action_size = action_size
		self.actions = list(range(9))
		self.q_table = defaultdict(lambda:[0.0 for i in range(action_space)])
		self.clk_action_list = []
		for i in range(3):
			for j in range(3):
				clk_action = (4 * i + 3, 4 * j + 3)
				self.clk_action_list.append(clk_action)
		
        # Hyperparameter
		self.learning_rate = 0.05    # 0.01
		self.discount_factor = 0.99
		self.epsilon = 1
		self.epsilon_decay = 0.95 # 0.99
		self.epsilon_min = 0 # 0.1
		self.epsilon_start, self.epsilon_end = 1.0, 0.0 # 1.0, 0.1
		self.batch_size = 64
		self.train_start = 200 #200
		self.q_max = 0
		self.avg_q_max = 0
		self.currentLoss = 0
		# Replay memory (=500)
		self.memory = deque(maxlen=500)
		
        # model initialization
		self.model = Model(self.state_size, self.action_size, self.learning_rate)
		self.target_model = Model(self.state_size, self.action_size, self.learning_rate)
		self.update_target_model()
		if self.load_model:
			self.model.load_state_dict(torch.load('model_weights.pth'))
			self.epsilon_start = 0.1
	
	def update_target_model(self):
		self.target_model.load_state_dict(self.model.state_dict())
	
	def get_action(self, state):
		self.model.eval()
		state=np.array([state])
		if np.random.rand() <= self.epsilon:
			with torch.no_grad():
				q_value=self.model(state)
			print('state={}, q_value={}, action=exploration, epsilon={}'.format(state[0], q_value[0], self.epsilon))
			return random.randrange(self.action_size) # exploration
		else:
			with torch.no_grad():
				q_value = self.model(state)
			print('state={}, q_value={}, action={}, epsilon={}'.format(state[0], q_value[0], np.argmax(q_value[0]), self.epsilon))
			return np.argmax(q_value[0]) # exploitation
	
	def append_sample(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))
	
	def train_model(self):
		self.training = 1

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
		else:
			self.epsilon = self.epsilon_min

		mini_batch = random.sample(self.memory, self.batch_size)

		states = torch.Tensor([sample[0] for sample in mini_batch])
		actions = torch.LongTensor([sample[1] for sample in mini_batch])
		rewards = torch.Tensor([sample[2] for sample in mini_batch])
		next_states = torch.Tensor([sample[3] for sample in mini_batch])
		dones = torch.BoolTensor([sample[4] for sample in mini_batch])

		self.model.train()
		self.target_model.eval()

		# 计算 Q 值
		q_values = self.model(states)
		next_q_values = self.target_model(next_states).detach()

		target = q_values.clone()

		for i in range(self.batch_size):
			if dones[i]:
				target[i, actions[i]] = rewards[i]
			else:
				target[i, actions[i]] = rewards[i] + self.discount_factor * torch.max(next_q_values[i])

		# 计算损失并进行反向传播
		loss = self.model.compute_loss(q_values, target)
		self.model.optimizer.zero_grad()
		loss.backward()
		self.model.optimizer.step()

		self.currentLoss = loss.item()
		print('loss = {}'.format(self.currentLoss))
		self.training = 0