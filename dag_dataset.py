'''
Copyright 2018 Hongzheng Chen
E-mail: chenhzh37@mail2.sysu.edu.cn

This is the implementation of Deep-reinforcement-learning-based scheduler for High-Level Synthesis.

This file contains the DagDataset class.
'''

import numpy as np
from torch.utils.data import Dataset

class DagDataset(Dataset):
	def __init__(self,state_action_pair):
		super(DagDataset, self).__init__()
		self.state_action_pair = np.array(state_action_pair)

	def __len__(self):
		return len(self.state_action_pair)

	def __getitem__(self, idx):
		eps = np.finfo(np.float32).eps.item()
		state = np.array(self.state_action_pair[idx][0]).astype(np.float64)
		state = (state - state.mean(axis = 0)) / (state.std(axis = 0) + eps)
		return (state,np.array(self.state_action_pair[idx][1]))