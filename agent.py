'''
Copyright 2018 Hongzheng Chen
E-mail: chenhzh37@mail2.sysu.edu.cn

This is the implementation of Deep-reinforcement-learning-based scheduler for High-Level Synthesis.

This file contains the Agent class.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from policy import Policy

class Agent(object):
	def __init__(self, state_size, use_network="", device="cuda",lr=5e-4):
		super(Agent, self).__init__()
		self.device = device
		if use_network == "":
			net = Policy(state_size[0]).to(self.device)
			print("Build a new network!")
		else:
			try:
				net = torch.load("./Networks/" + use_network).to(self.device)
				net.classifier = nn.Sequential(*list(net.classifier.children())[:-1]) # delete the softmax layer
				print("Loaded %s." % use_network)
			except:
				net = Policy(state_size[0]).to(self.device)
				print("No such network named %s. Rebuild a new network!" % use_network)
		self.policy = net
		# self.policy = net.eval() # avoid dropout
		self.optimizer = optim.Adam(self.policy.parameters(),lr=lr)

	def get_sl_action(self, state):
		output = self.policy(state) # bs(1)*50
		# randomly select
		action = torch.topk(output,1)
		action = action[1] # op
		criterion = nn.NLLLoss()
		nllloss = criterion(output,torch.Tensor([action]).type(torch.LongTensor).to(self.device).resize_((1,)))
		return nllloss, action

	def get_action(self, state, legal_move):
		output = self.policy.forward_without_softmax(state) # bs(1)*50
		legal_move_dict = legal_move[1]
		legal_move = torch.tensor(legal_move[0]).long().to(self.device)
		legal_prob = torch.index_select(output,1,legal_move)
		# randomly select
		if len(legal_prob.shape) == 2 and legal_prob.shape[1] != 1:
			m = Categorical(F.softmax(legal_prob,dim=1))
			index = m.sample().item()
		else:
			index = 0
		action = legal_move_dict[index]
		criterion = nn.NLLLoss()
		nllloss = criterion(F.log_softmax(legal_prob,dim=1),torch.Tensor([index]).type(torch.LongTensor).to(self.device).resize_((1,)))
		del output
		return nllloss, action # log_prob, action

	def get_deterministic_action(self, state, legal_move):
		output = self.policy(state) # bs(1)*50
		legal_move_dict = legal_move[1]
		legal_move = torch.tensor(legal_move[0]).long().to(self.device)
		legal_prob = torch.index_select(output,1,legal_move)
		action = torch.topk(legal_prob,1)
		action = action[1] # op
		if len(legal_prob.shape) == 2 and legal_prob.shape[1] != 1:
			action = legal_move_dict[action.item()]
		else:
			action = legal_move_dict[0]
		log_prob = output[0][action] # requires_grad
		return log_prob, action

	def update_weight(self, all_log_probs, all_rewards, baseline=False):
		gamma = 0.99
		eps = np.finfo(np.float32).eps.item()
		tot_loss = []
		res_rewards, avg_reward = [], []
		# baseline `1/N\sum_{i=1}^N r(\tau)`
		for log_prob, temp_rewards in zip(all_log_probs,all_rewards):
			# a full trace \tau
			R = 0
			rewards = []
			for r in temp_rewards[::-1]:
				R = r + gamma * R
				rewards.insert(0, R)
			avg_reward.append(rewards[0]) # r(\tau)
			res_rewards.append(rewards)
		if baseline:
			avg_reward = np.array(avg_reward).mean()
		else:
			avg_reward = 0
		for log_prob, rewards in zip(all_log_probs,res_rewards):
			rewards = torch.tensor(rewards).to(self.device)
			rewards = rewards - avg_reward # minus baseline
			loss = torch.Tensor([0]).float().to(self.device)
			for step, (nllloss, reward) in enumerate(zip(log_prob,rewards)):
				# if prob is very small (say 0.01) then -log(prob) is extremely large
				# reward needs to be small to make loss small
				loss += nllloss * reward # minus!
			tot_loss.append(loss)
			# tot_loss.append(torch.dot(torch.tensor(log_prob).to(self.device),rewards))
		# backpropagate
		self.optimizer.zero_grad()
		# loss = torch.stack(tot_loss, dim=0).sum() / len(tot_loss)
		tot_loss = torch.cat(tot_loss).mean() # sum()
		tot_loss.backward()
		self.optimizer.step()
		res = tot_loss.item()
		del tot_loss
		return res