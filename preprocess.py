'''
Copyright 2018 Hongzheng Chen
E-mail: chenhzh37@mail2.sysu.edu.cn

This is the implementation of Deep-reinforcement-learning-based scheduler for High-Level Synthesis.

This file contains the preprocess part.
'''

from graph import Graph
from ilp_solver import ILPSolver

def preprocess(tot_files):
	state_action_pair = []
	# logger.info("Begin generating data...")
	for file_num in range(1,tot_files+1):
		generateData(file_num,state_action_pair)
		if file_num % 10 == 0:
			print("Generated %d / %d." % (file_num,tot_files))
			# logger.info("Generated %d / %d." % (file_num,tot_files))
	# logger.info("Finish generating data.")
	return state_action_pair

def generateData(file_num,state_action_pair,state_size=(50,50)):
	graph = Graph("TCS")
	graph.read(open("./DAG/dag_%d.dot" % file_num,"r"))
	graph.initialize()
	graph.initial_schedule()
	if graph.getConstrainedL() > 50:
		print("File %d exceeds 50 latency." % file_num)
		return
	try:
		sol = open("./Sol/dag_%d.sol" % file_num,"r")
		ops = dict()
		for line in sol:
			op, cstep = map(int,line.split(", "))
			ops[op] = cstep
	except:
		ilp = ILPSolver(file_num,graph.getMulDelay(),graph.getLf())
		ops = ilp.getOptSchedule()
	for node in graph.revOrder:
		if ops[node.num] == node.cstep:
			continue
		for t in range(node.cstep+1,ops[node.num]+1):
			state_action_pair.append((graph.get_partial_state(state_size),node.num))
			graph.schedule_node(node.num,t)
			# logger.debug("Node %d schedules on cstep %d" % (node.num,t))