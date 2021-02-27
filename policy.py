'''
Copyright 2018 Hongzheng Chen
E-mail: chenhzh37@mail2.sysu.edu.cn

This is the implementation of Deep-reinforcement-learning-based scheduler for High-Level Synthesis.

This file contains the architecture of the policy network.
The policy network is modify from the VGG network.
@article{vgg,
	author    = {Karen Simonyan and Andrew Zisserman},
	title     = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
	journal   = {CoRR},
	year      = {2014},
	url       = {http://arxiv.org/abs/1409.1556},
}
'''

import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
	# modify from VGG11
	def __init__(self, output_size, batch_norm=False):
		super(Policy, self).__init__()
		# minibatch*in_channels*iH*iW
		# bs*output_size
		self.features = nn.Sequential(
			# 3*50*50
			nn.Conv2d(3,64,kernel_size=3,padding=1), # default stride = 1
			nn.ReLU(True),
			# 64*50*50
			nn.Conv2d(64,64,kernel_size=3,padding=1),
			nn.ReLU(True),
			# 64*50*50
			nn.MaxPool2d(kernel_size=2,stride=2), # default stride = kernel size
			# 64*25*25
			nn.Conv2d(64,128,kernel_size=3,padding=1),
			nn.ReLU(True),
			# 128*25*25
			nn.Conv2d(128,128,kernel_size=3,padding=1),
			nn.ReLU(True),
			# 128*25*25
			nn.MaxPool2d(kernel_size=2,stride=2),
			# 128*12*12
			nn.Conv2d(128,256,kernel_size=3,padding=1),
			nn.ReLU(True),
			# 256*12*12
			nn.Conv2d(256,256,kernel_size=3,padding=1),
			nn.ReLU(True),
			# 256*12*12
			nn.MaxPool2d(kernel_size=2,stride=2)
			# 256*6*6
		)
		self.classifier = nn.Sequential(
			nn.Linear(256 * 6 * 6, 2048),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(2048, 2048),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(2048, output_size),
		)
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		x = F.log_softmax(x,dim=1)
		return x
	
	def forward_without_softmax(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x