'''
Copyright 2018 Hongzheng Chen
E-mail: chenhzh37@mail2.sysu.edu.cn

This is the implementation of Deep-reinforcement-learning-based scheduler for High-Level Synthesis.

This file contains the supervised learning (SL) part of the training pipeline.
'''

import time, sys, os, argparse
import random
import numpy as np
import visdom
import matplotlib.pyplot as plt
from logger import LogHandler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from graph import Graph
from preprocess import preprocess
from policy import Policy
from dag_dataset import DagDataset

parser = argparse.ArgumentParser(description="Deep-RL-Based HLS Scheduler (Supervised Learning)")
parser.add_argument("--use_cuda", type=int, default=1, help="Use cuda? (default: True, the 1st GPU)")
parser.add_argument("--input_graphs", type=int, default=3500, help="Number of input graphs? (default: 3500)")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size? (default: 128)")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate? (default: 5e-4)")
parser.add_argument("--epoch", type=int, default=10000, help="Number of epoch? (default: 10000)")
parser.add_argument("--use_network", type=str, default="", help="Use previous network? Input the name of the network. (default: None)")
args = parser.parse_args()

logger_num, logger = LogHandler("sl").getLogger()
logger.info("Deep-RL-Based HLS Scheduler (Supervised Learning)")
print("Logger num: %d" % logger_num)
device = torch.device(("cuda:%d" % (args.use_cuda-1)) if args.use_cuda != 0 else "cpu")
file_name = "_sl_" + time.strftime("%Y%m%d_") + str(logger_num)

STATE_SIZE = (50,50)

if args.use_network == "":
	net = Policy(STATE_SIZE[0]).to(device)
	print("Build a new network!")
else:
	try:
		net = torch.load("./Networks/" + args.use_network).to(device)
		print("Loaded %s." % args.use_network)
		logger.info("Pretrained network: %s (%s)" % (args.use_network,"gpu" if args.use_cuda else "cpu"))
	except:
		print("No such network named %s. Rebuild a new network!" % args.use_network)
		net = Policy(STATE_SIZE[0]).to(device)
network_file = "./Networks/policy" + file_name + ".pkl"
logger.info("New network: %s (%s)" % (network_file,"gpu" if args.use_cuda else "cpu"))
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=args.learning_rate)
logger.info(net.features)
logger.info(net.classifier)
logger.info("NLLLoss (Negative Log likelihood loss) + Adam")
logger.info("Batch size: %d, Learning rate: %f" % (args.batch_size,args.learning_rate))

best_accuracy = 0
viz = visdom.Visdom()
cur_batch_win, epoch_loss_win = None, None

def train(epoch):
	global cur_batch_win
	net.train()
	total_correct = 0
	loss_list, batch_list = [], []
	for i, (state, action) in enumerate(data_train_loader):
		state = torch.Tensor(state.float()).to(device)
		action = torch.Tensor(action.float()).type(torch.LongTensor).to(device)
		optimizer.zero_grad()
		output = net(state)
		# bs*50 <- bs labels
		loss = criterion(output,action)
		loss_list.append(loss.item())
		batch_list.append(i+1)
		predict = output.data.max(1)[1]
		total_correct += predict.eq(action.data.view_as(predict)).sum()
		if i % 10 == 0:
			logger.info("Train - Epoch %d, Batch: %d, Loss: %f" % (epoch,i,loss.item()))
		if viz.check_connection():
			cur_batch_win = viz.line(X=torch.FloatTensor(batch_list), Y=torch.FloatTensor(loss_list),
									 win=cur_batch_win, name='current_batch_loss',
									 update=(None if cur_batch_win is None else 'replace'),
									 opts={'title': 'Epoch Loss Trace','xlabel': 'Batch Number','ylabel': 'Loss','width': 1200,'height': 600})
		loss.backward()
		optimizer.step()
	avg_loss = np.array(loss_list).sum() / len(data_train_loader)
	accuracy = float(total_correct) / len(data_train)
	logger.info("Train Epoch %d: Avg. Loss: %f, Accuracy: %f" % (epoch,avg_loss,accuracy))
	print("Train Epoch %d: Avg. Loss: %f, Accuracy: %f" % (epoch,avg_loss,accuracy))
	return avg_loss

def test(epoch):
	global best_accuracy
	net.eval()
	total_correct = 0
	avg_loss = 0.0
	for i, (state, action) in enumerate(data_test_loader):
		state = torch.Tensor(state.float()).to(device)
		action = torch.Tensor(action.float()).type(torch.LongTensor).to(device)
		output = net(state)
		avg_loss += criterion(output, action).item() # sum()
		predict = output.data.max(1)[1]
		total_correct += predict.eq(action.data.view_as(predict)).sum()
	avg_loss /= (len(data_test_loader))
	accuracy = float(total_correct) / len(data_test)
	logger.info("Test Epoch %d: Avg. Loss: %f, Accuracy: %f" % (epoch,avg_loss,accuracy))
	print("Test Epoch %d: Avg. Loss: %f, Accuracy: %f" % (epoch,avg_loss,accuracy))
	if best_accuracy < accuracy:
		best_accuracy = accuracy
		torch.save(net,network_file[:-4]+"_best.pkl")
	return avg_loss

def visualization(epoch,train_loss,test_loss):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot([i for i in range(1,epoch+1)],np.array(train_loss),label="train")
	ax.plot([i for i in range(1,epoch+1)],np.array(test_loss),label="test")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.legend()
	fig.savefig("./Loss/fig" + file_name + ".jpg")
	plt.cla()
	plt.close()
	np.save("./Loss/train_loss" + file_name + ".npy",np.array(train_loss))
	np.save("./Loss/test_loss" + file_name + ".npy",np.array(test_loss))

state_action_pair = preprocess(args.input_graphs)
random.shuffle(state_action_pair) # important to break out the corelation
cut = int(0.96*len(state_action_pair))
data_train = DagDataset(state_action_pair[:cut])
data_test = DagDataset(state_action_pair[cut:])
data_train_loader = DataLoader(data_train,shuffle=True,batch_size=args.batch_size,num_workers=12)
data_test_loader = DataLoader(data_test,shuffle=True,batch_size=args.batch_size,num_workers=12)
print("# of train data: %d" % len(data_train))
print("# of test data: %d" % len(data_test))
logger.info("# of input graphs: %d" % args.input_graphs)
logger.info("# of train data: %d" % len(data_train))
logger.info("# of test data: %d" % len(data_test))
startTime = time.time()
logger.info("Begin training...")
train_loss = []
test_loss = []
for epoch in range(1,args.epoch+1):
	train_loss.append(train(epoch))
	test_loss.append(test(epoch))
	visualization(epoch,train_loss,test_loss)
	torch.save(net,network_file)
	usedTime = time.time() - startTime
	print("Finish %d / %d. Total time used: %f min. Rest of time: %f min."
		% (epoch,args.epoch,usedTime/60,usedTime/60*args.epoch/epoch-usedTime/60))
logger.info("Finish training.")