'''
Copyright 2018 Hongzheng Chen
E-mail: chenhzh37@mail2.sysu.edu.cn

This is the implementation of Deep-reinforcement-learning-based scheduler for High-Level Synthesis.

This file contains the reinforcement learning (RL) part of the training pipeline.
'''

import time, sys, os, argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from logger import LogHandler
from graph import Graph
from policy import Policy
from agent import Agent
from dag_dataset import DagDataset

parser = argparse.ArgumentParser(description="Deep-RL-Based HLS Scheduler (Reinforcement learning)")
parser.add_argument("--mode", type=str, default="TCS", help="Scheduling mode: TCS or RCS (default TCS)")
parser.add_argument("--lc", type=float, default=1, help="Latency factor used for TCS (default: 1)")
parser.add_argument("--mul_delay", type=int, default=2, help="MUL delay (default: 2)")
parser.add_argument("--episodes", type=int, default=1000, help="Max iteration episodes (default: 1000)")
parser.add_argument("--input_graphs", type=int, default=3000, help="Number of input graphs? (default: 3000)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size? (default: 32)")
parser.add_argument("--timesteps", type=int, default=2500, help="Max timestep in one simulation (default: 2500)")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate? (default: 1e-3)")
parser.add_argument("--use_cuda", type=int, default=1, help="Use cuda? (default: True, the 1st GPU)")
parser.add_argument("--use_network", type=str, default="", help="Use previous network? Input the name of the network. (default: None)")
parser.add_argument("--test", type=int, default=-1, help="Test file num? (default: -1)")
parser.add_argument("--stride", type=int, default=3, help="Stride of the kernel? (default: 3)")
args = parser.parse_args()

best_reward = 0

STATE_SIZE = (50,50)
device = torch.device(("cuda:%d" % (args.use_cuda-1)) if args.use_cuda != 0 else "cpu")
agent = Agent(STATE_SIZE,use_network=args.use_network,device=device,lr=args.learning_rate)

if args.test == -1:
	logger_num, logger = LogHandler("rl").getLogger()
	logger.info("Deep-RL-Based HLS Scheduler (Reinforcement Learning)")
	print("Logger num: %d" % logger_num)
	file_name = "_rl_" + time.strftime("%Y%m%d_") + str(logger_num)
	logger.info(agent.policy.features)
	logger.info(agent.policy.classifier)
	logger.info("NLLLoss + Adam")
	logger.info("Batch size: %d, Learning rate: %f" % (args.batch_size,args.learning_rate))
	logger.info(Graph("TCS").reward)

def train(episode): # Monte Carol REINFORCE
	global best_reward
	res_loss, res_reward = [], []
	for i_graph in range(args.input_graphs//args.batch_size):
		all_log_probs, all_rewards = [], []
		# simulate batch_size graphs
		for minibatch in range(args.batch_size):
			log_probs, rewards = [], []
			graph = Graph(args.mode) # "TCS"
			graph.read(open("./DAG/dag_%d.dot" % (i_graph*args.batch_size+minibatch+1),"r"))
			graph.initialize()
			graph.initial_schedule()
			# one full trace \tau
			for timestep in range(args.timesteps):
				state = torch.Tensor(graph.get_partial_state(STATE_SIZE)).float().to(device)
				state = state.resize_((1,state.size()[0],state.size()[1],state.size()[2]))
				legalMove = graph.getLegalMove()
				if len(legalMove[0]) == 0:
					break
				log_prob, action = agent.get_action(state,legalMove)
				fes, reward = graph.schedule_node(action, graph.vertex if action >= graph.vertex else graph.adjlist[action].cstep + 1)
				log_probs.append(log_prob)
				rewards.append(reward)
				if fes == False:
					break
			all_log_probs.append(log_probs)
			all_rewards.append(np.array(rewards).astype(np.float))
		# update policy
		loss = agent.update_weight(all_log_probs,all_rewards,baseline=False) # be careful that the rewards are not aligned
		avg_reward = np.array([x.sum() for x in all_rewards]).mean()
		res_loss.append(loss)
		res_reward.append(avg_reward)
		if i_graph % 10 == 0:
			print("Train - Episode %d, Batch: %d, Loss: %f, Reward: %f" % (episode,i_graph,loss,avg_reward))
			logger.info("Train - Episode %d, Batch: %d, Loss: %f, Reward: %f" % (episode,i_graph,loss,avg_reward))
		if best_reward < avg_reward:
			best_reward = avg_reward
			torch.save(agent.policy,"./Networks/policy" + file_name + "_best.pkl")
		del all_log_probs[:]
		del all_rewards[:]
	return (np.array(res_loss).mean(), np.array(res_reward).mean())

def test(file_num):
	print("Begin testing...")
	nrt, nrta, step = [], [], []
	graph = Graph(args.mode,args.mul_delay) # "TCS"
	graph.setLatencyFactor(args.lc)
	graph.read(open("./DAG/dag_%d.dot" % file_num,"r"))
	graph.initialize()
	graph.initial_schedule()
	print("ASAP # of resources: MUL: %d, ALU: %d" % (graph.currNr["MUL"],graph.currNr["ALU"]))
	step.append(0)
	nrt.append(graph.currNr["MUL"])
	nrta.append(graph.currNr["ALU"])
	flag_in = False
	timestep = 0
	cnt_loop = 0
	stride = args.stride
	pos_num = [0]
	while pos_num[-1] + STATE_SIZE[0] <= graph.vertex:
		pos_num.append(pos_num[-1] + stride)
	print(pos_num)
	while timestep < args.timesteps:
		for i in pos_num:
			state = torch.Tensor(graph.get_partial_state(STATE_SIZE,pos=(i,0))).float().to(device)
			state = state.resize_((1,state.size()[0],state.size()[1],state.size()[2]))
			legalMove = graph.getLegalMove(pos=(i,0))
			if cnt_loop >= len(pos_num):
				print("Early stop! No legal actions!")
				flag_in = True
				break
			if len(legalMove[0]) == 0:
				cnt_loop += 1
				continue
			cnt_loop = 0
			# log_prob, action = agent.get_sl_action(state)
			log_prob, action = agent.get_deterministic_action(state, legalMove)
			action += i
			fes, reward = graph.schedule_node(action, graph.vertex if action >= graph.vertex else graph.adjlist[action].cstep + 1)
			if fes == False:
				if action >= graph.vertex:
					print("Timestep %d: op %d (exceed), not available!" % (timestep+1,action))
				else:
					print("Timestep %d: op %d move to %d, early stop!" % (timestep+1,action,graph.adjlist[action].cstep + 1))
				flag_in = True
				break
			else:
				print("Timestep %d: op %d move to %d, reward: %f" % (timestep+1,action,graph.adjlist[action].cstep,reward))
				step.append(timestep+1)
				nrt.append(graph.currNr["MUL"])
				nrta.append(graph.currNr["ALU"])
			timestep += 1
		if flag_in:
			break
	print("Finish testing.")
	print(graph.test_final())
	print(graph.get_state())
	graph.output()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	l1 = ax.plot(step,nrt,label="MUL")
	l2 = ax.plot(step,nrta,label="ALU")
	ax.set_xlabel("Step")
	ax.set_ylabel("# of ops")
	# ax.set_title("%s" % input())
	ax.legend(loc=1)
	fig.savefig("./fig_test_%d.pdf" % file_num,format="pdf")
	plt.show()
	return (nrt[0],nrta[0],graph.bestNr["MUL"],graph.bestNr["ALU"])

def visualization(results):
	res_r = np.array([x[0] for x in results])
	res_l = np.array([x[1] for x in results])
	np.save("./Loss/" + "reward" + file_name + ".npy",res_r)
	np.save("./Loss/" + "loss" + file_name + ".npy",res_l)
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	lns1 = ax1.plot(range(len(res_r)),res_r,label="Reward",color="b")
	ax2 = ax1.twinx()  # this is the most important function
	lns2 = ax2.plot(range(len(res_l)),res_l,label="Loss",color="r")
	lns = lns1 + lns2
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc=0)
	fig.savefig("./Loss/" + "fig" + file_name + ".jpg")

if args.test != -1:
	agent.policy.eval()
	res = []
	# for i in range(10001,10021):
	# 	res.append(test(i))
	res.append(test(i))
	for x in res:
		print("%d %d %d %d %d %d" % (x[0],x[1],x[0]+x[1],x[2],x[3],x[2]+x[3]))
	sys.exit()

logger.info("Begin training...")
startTime = time.time()
results = []
for episode in range(1,args.episodes+1):
	results.append(train(episode))
	visualization(results)
	logger.info("Train Episode %d: Avg. Loss: %f, Avg. Reward: %f" % (episode,results[-1][0],results[-1][1]))
	print("Train Episode %d: Avg. Loss: %f, Avg. Reward: %f" % (episode,results[-1][0],results[-1][1]))
	torch.save(agent.policy,"./Networks/policy" + file_name +".pkl")
	usedTime = time.time() - startTime
	print("Finish %d / %d. Total time used: %f min. Rest of time: %f min."
		% (episode,args.episodes,usedTime/60,usedTime/60*args.episodes/episode-usedTime/60))
logger.info("Finish training.")