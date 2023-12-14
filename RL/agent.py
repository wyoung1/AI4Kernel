from collections import defaultdict, deque
import random
import socket
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


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
		state=torch.Tensor(state)
		if np.random.rand() <= self.epsilon:
			q_value=self.model(state).detach().numpy()
			print('state={}, q_value={}, action=exploration, epsilon={}'.format(state[0], q_value[0], self.epsilon))
			return random.randrange(self.action_size) # exploration
		else:
			q_value = self.model(state).detach().numpy()
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

def get_reward(fps, power, target_fps, c_t, g_t, c_t_prev, g_t_prev, beta):
	v1 = 0.2 * np.tanh(target_temp - c_t)
	if c_t >= target_temp:
			v1 = -2
	if c_t_prev < target_temp:
		if c_t >= target_temp:
			v1 = -2

	if fps >= target_fps:
		u = 1
	else:
		u = fps / target_fps

	return u + v1 + beta / power

if __name__ == "__main__":
	agent = DQNAgent(7, 9)
	scores, episodes = [], []

	t = 0
	learn = 1
	ts = []
	fps_data = []
	power_data = []
	avg_q_max_data = []
	loss_data = []
	reward_tmp = []
	avg_reward = []
	cnt = 0
	c_c = 11
	g_c = 11
	c_t = 37
	g_t = 37
	c_t_prev = 37
	g_t_prev = 37
	state = (11, 12, 20, 27, 40, 40, 30)
	score = 0
	action = 0
	copy = 1
	clk = 11

	print("Waiting connection")
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_socket.bind(("", 8702))
	server_socket.listen(5)

	try:
		client_socket, address = server_socket.accept()
		fig = plt.figure(figsize = (12,14))
		ax1 = fig.add_subplot(4, 1, 1)
		ax2 = fig.add_subplot(4, 1, 2)
		ax3 = fig.add_subplot(4, 1, 3)
		ax4 = fig.add_subplot(4, 1, 4)
		
		while t < experiment_time:
			msg = client_socket.recv(512).decode()
			state_tmp = msg.split(',')
			if not msg:
				print('No receiveddata')
				break

			c_t_prev = c_t
			g_t_prev = g_t
			c_c = int(state_tmp[0])
			g_c = int(state_tmp[1])
			c_p = int(state_tmp[2])
			g_p = int(state_tmp[3])
			c_t = float(state_tmp[4])
			g_t = float(state_tmp[5])
			fps = float(state_tmp[6])
			ts.append(t)
			fps_data.append(fps)
			power_data.append((c_p + g_p) * 100)

			next_state = (c_c, g_c, c_p, g_p, c_t, g_t, fps)
			agent.q_max += np.amax(agent.model(
				torch.Tensor(next_state)).detach().numpy())
			agent.avg_q_max = agent.q_max / t
			avg_q_max_data.append(agent.avg_q_max)
			loss_data.append(agent.currentLoss)

			# Pop dummy value at the first sensing
			if (c_p + g_p <= 0):
				c_p = 20
				g_p = 13
			reward = get_reward(fps, c_p+g_p, target_fps, c_t, g_t, c_t_prev, g_t_prev, beta)
			reward_tmp.append(reward)
			if(len(reward_tmp) >= 300) :
				reward_tmp.pop(0)

			done = 1

			# replay memory
			agent.append_sample(state, action, reward, next_state, done)
			# double copy for highlighted sample
			for i in range(copy):
				if reward<0:
					agent.learning_rate = 1
					agent.append_sample(state, action, reward, next_state, done)
					agent.append_sample(state, action, reward, next_state, done)
				if reward>1:
					agent.learning_rate = 1
					agent.append_sample(state, action, reward, next_state, done)
					agent.append_sample(state, action, reward, next_state, done)

			print('[{}] state:{} action:{} next_state:{} reward:{} fps:{}, avg_q={}'.format(t, state,action,next_state,reward,fps,agent.avg_q_max))
			if len(agent.memory) >= agent.train_start:
				agent.train_model()

			score += reward			
			avg_reward.append(sum(reward_tmp) / len(reward_tmp))			
			print('learning_rate:{}'.format(agent.learning_rate))

			# get action
			state=next_state
			if c_t>=target_temp:
				c_c=int(4*random.randint(0,int(c_c/3)-1)+3)
				g_c=int(4*random.randint(0,int(g_c/3)-1)+3)
				action=3*int(c_c/4)+int(g_c/4)
			elif target_temp-c_t>=5:
				if fps<target_fps:
					if np.random.rand() <= 0.3:
						print('previous clock : {} {}'.format(c_c,g_c))
						c_c=int(4*random.randint(int(c_c/3)-1,2)+3)
						g_c=int(4*random.randint(int(g_c/3)-1,2)+3)
						print('explore higher clock@@@@@  {} {}'.format(c_c,g_c))
						action=3*int(c_c/4)+int(g_c/4)
					else:
						action=agent.get_action(state)
						c_c=agent.clk_action_list[action][0]
						g_c=agent.clk_action_list[action][1]
				else:
					action=agent.get_action(state)
					c_c=agent.clk_action_list[action][0]
					g_c=agent.clk_action_list[action][1]

			else:
				action=agent.get_action(state)
				c_c=agent.clk_action_list[action][0]
				g_c=agent.clk_action_list[action][1]	

			send_msg=str(c_c)+','+str(g_c)
			client_socket.send(send_msg.encode())

			### Real-time graph
			ax1.plot(ts, fps_data, linewidth=1, color='pink')
			ax1.set_ylabel('Frame rate (fps)')
			ax1.set_xlabel('Time (s) ')
			ax1.set_xticks([0, 500, 1000, 1500, 2000])
			ax1.set_yticks([15, 20, 25, 30, 35, 40])
			ax1.grid(True)
			
			ax2.plot(ts, power_data, linewidth=1, color='blue')
			ax2.set_ylabel('Power (mW)')
			ax2.set_yticks([0, 2000, 4000, 6000, 8000])
			ax2.set_xticks([0, 250, 500, 750, 1000])
			ax2.set_xlabel('Time (s) ')
			ax2.grid(True)

			ax3.plot(ts, avg_q_max_data, linewidth=1, color='orange')
			ax3.set_ylabel('AVg. max Q-value')
			ax3.set_xticks([0, 500, 1000, 1500, 2000])
			ax3.set_xlabel('Time (s) ')
			ax3.grid(True)
			
			ax4.plot(ts, loss_data, linewidth=1, color='black')
			ax4.set_ylabel('Average loss')
			ax4.set_xticks([0, 500, 1000, 1500, 2000])
			ax4.set_xlabel('Time (s) ')
			ax4.grid(True)		
			
			plt.pause(0.1)
			
			if done:
				agent.update_target_model()
			t = t + 1
			if t%60 == 0:
				agent.learning_rate=0.2
				print('[Reset learning_rate]')
			if t%500 == 0:
				torch.save(agent.model.state_dict(), "model_weights.pth")
				print("[Save model]")
			if t==experiment_time:
				break

	finally:
		ts = range(0, len(avg_q_max_data))
		f = open('maxq.csv', 'w', encoding='utf-8', newline='')
		wr = csv.writer(f)
		wr.writerow(ts)
		wr.writerow(avg_q_max_data)
		f.close

		f = open('reward.csv', 'w', encoding='utf-8', newline='')
		wr = csv.writer(f)
		wr.writerow(ts)
		wr.writerow(avg_reward)
		f.close

		f = open('loss.csv', 'w', encoding='utf-8', newline='')
		wr = csv.writer(f)
		wr.writerow(ts)
		wr.writerow(loss_data)
		f.close
		server_socket.close()

	plt.show()