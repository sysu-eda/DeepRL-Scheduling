'''
Copyright 2018 Hongzheng Chen
E-mail: chenhzh37@mail2.sysu.edu.cn

This is the implementation of Deep-reinforcement-learning-based scheduler for High-Level Synthesis.

This file contains the definition and implementation of Graph class.
'''

import re, sys
import numpy as np
from node import Node

class Graph(object):
	def __init__(self, mode, mul=2):
		self.mode = mode
		self.mul_delay = mul
		self._LC = 1
		self.vertex = 0
		self.edge = 0
		self.adjlist = []
		self.depth = 0
		self.order = []
		self.revOrder = []
		self.totLatency = 0
		self.numScheduledOp = 0
		# state[0]: Current schedule
		# state[1]: Current possible move
		# state[2]: All possible move
		self.state = []
		# reward and punishment
		self.reward = dict()
		self.reward["penalty"] = 0
		self.reward["small"] = 0
		self.reward["nothing"] = 0

	def setLatencyFactor(self,lc):
		self._LC = lc

	def setConstrainedL(self,conL):
		self.CONSTRAINED_L = conL

	def getConstrainedL(self):
		return self.CONSTRAINED_L+1

	def getMulDelay(self):
		return self.mul_delay

	def getLf(self):
		return self._LC

	def setMAXRESOURCE(self,r):
		self.maxNr = {"MUL":r[0], "ALU":r[1]}
		print("Constrained resources: MUL: %d ALU: %d" % (self.maxNr["MUL"],self.maxNr["ALU"]))

	def initialize(self):
		self.dfs() # obtain CONSTRAINED_L
		self.currNr = {"MUL":0, "ALU":0}
		self.bestNr = {"MUL":0x3f3f3f, "ALU":0x3f3f3f}
		self.nrt = {"MUL":np.array([0]*(self.CONSTRAINED_L+1)), "ALU":np.array([0]*(self.CONSTRAINED_L+1))}

	def read(self,infile):
		# print("Begin parsing...")
		for line in infile:
			if not ("label" in line or "name" in line):
				if "property" in line:
					res = re.split("=|,|\\].*",line)
					self.mul_delay = int(res[1])
					self.setLatencyFactor(float(res[3]))
				else:
					continue
			elif "label" in line:
				res = re.split(" *\\[ *label *= *| *\\];| +",line)
				op, op_type = res[1], res[2]
				self.add_vertex(op,op_type)
			else:
				res = re.split(" *\\[ *name *= *| *\\];| *-> *| +",line)
				src, des = res[1], res[2]
				self.add_edge(src,des)
		# print("Finish parsing!")

	def mapR(self,type_,mode=0):
		if (type_ == "mul" or type_ == "MUL" or type_ == "div" or type_ == "DIV"):
			return ("MUL" if mode == 0 else 0)
		else:
			return ("ALU" if mode == 0 else 1)

	def add_vertex(self,name_,type_):
		delay = 1
		if self.mapR(type_) == "MUL":
			delay = self.mul_delay
		v = Node(self.vertex,name_,type_,delay)
		self.vertex += 1
		self.adjlist.append(v)

	def add_edge(self,src,des):
		for i in range(len(self.adjlist)):
			if self.adjlist[i].name == src:
				for j in range(len(self.adjlist)):
					if self.adjlist[j].name == des:
						self.adjlist[i].succ.append(j)
						self.adjlist[j].pred.append(i)
						self.edge += 1
						break

	def dfsASAP(self,num):
		if self.mark[num]:
			return
		if len(self.adjlist[num].pred) == 0:
			self.adjlist[num].setASAP(-1,0)
		else:
			for j in self.adjlist[num].pred:
				self.dfsASAP(j)
				self.adjlist[num].setASAP(j,self.adjlist[j].getASAP() + self.adjlist[j].delay)
		self.depth = max(self.adjlist[num].getASAP() + self.adjlist[num].delay - 1, self.depth)
		if self.mode == "TCS":
			self.setConstrainedL(int((self.depth)*self._LC))
		else:
			self.setConstrainedL(self.CONSTRAINED_L)
		self.mark[num] = True
		self.order.append(self.adjlist[num])

	def dfsALAP(self,num):
		if self.mark[num]:
			return
		if len(self.adjlist[num].succ) == 0:
			# CONSTRAINED_L is used here, dfsASAP must be done first
			self.adjlist[num].setALAP(-1, self.CONSTRAINED_L - self.adjlist[num].delay + 1)
		else:
			for j in self.adjlist[num].succ:
				self.dfsALAP(j)
				self.adjlist[num].setALAP(j,self.adjlist[j].getALAP() - self.adjlist[num].delay)
		self.mark[num] = True
		self.revOrder.append(self.adjlist[num])

	def dfs(self):
		# print("Begin DFS...")
		self.mark = np.zeros(self.vertex,dtype=bool)
		for i in range(len(self.adjlist)):
			if len(self.adjlist[i].succ) == 0:
				self.dfsASAP(i)
		self.mark = np.zeros(self.vertex,dtype=bool)
		for i in range(len(self.adjlist)):
			if len(self.adjlist[i].pred) == 0:
				self.dfsALAP(i)
		# print("Finish DFS.")
		# print("Constrained Latency is %d" % (self.CONSTRAINED_L+1))

	def initial_schedule(self):
		# clear previous state
		self.totLatency = 0
		self.numScheduledOp = 0
		self.currNr = {"MUL":0, "ALU":0}
		self.bestNr = {"MUL":0x3f3f3f, "ALU":0x3f3f3f}
		self.nrt = {"MUL":np.array([0]*(self.CONSTRAINED_L+1)), "ALU":np.array([0]*(self.CONSTRAINED_L+1))}
		for i in range(len(self.adjlist)):
			self.adjlist[i].initial()
		# reschedule
		self.state = np.zeros((3,self.vertex,self.CONSTRAINED_L+1))
		for i in range(self.vertex):
			self.state[1:3,i,self.adjlist[i].getASAP():self.adjlist[i].getALAP() + self.adjlist[i].delay] = 1
		for i in range(self.vertex):
			self.schedule_node(i,self.adjlist[i].getASAP(),0)

	def schedule_node(self,op,step,mode=1):
		if not self.test_val(op,step):
			return False, self.reward["penalty"]
		reward = 0
		tempR = self.mapR(self.adjlist[op].type)
		tempNum = self.mapR(self.adjlist[op].type,1)
		# remove old state
		oldOpNr = 0
		for d in range(self.adjlist[op].delay):
			oldOpNr += self.nrt[tempR][self.adjlist[op].cstep + d]
		if mode == 1:
			self.numScheduledOp += 1
			for d in range(self.adjlist[op].delay):
				# since the op initially placed here, so it should be at least WA
				self.state[0,op,self.adjlist[op].cstep + d] = 0
				self.nrt[tempR][self.adjlist[op].cstep + d] -= 1
		# current operation
		self.adjlist[op].schedule(step)
		delay = self.adjlist[op].delay
		for d in range(delay):
			self.nrt[tempR][step + d] += 1
		self.state[0,op,step:step+delay] = 1
		self.state[1,op,step:step+delay] = 0
		self.state[1,op,self.adjlist[op].getASAP():step] = 1
		self.state[1,op,step+delay:self.adjlist[op].getALAP()+delay] = 1
		# other influenced operations
		for vpred in self.adjlist[op].pred:
			tempALAP = self.adjlist[vpred].getALAP()
			d = self.adjlist[vpred].delay
			self.adjlist[vpred].setALAP(op,step - d)
			currALAP = self.adjlist[vpred].getALAP()
			self.state[1,vpred,min(tempALAP,currALAP)+d:max(tempALAP,currALAP)+d] = 0 if currALAP < tempALAP else 1
			if currALAP > tempALAP:
				reward += self.reward["small"]
		for vsucc in self.adjlist[op].succ:
			tempASAP = self.adjlist[vsucc].getASAP()
			self.adjlist[vsucc].setASAP(op,step + self.adjlist[op].delay)
			currASAP = self.adjlist[vsucc].getASAP()
			self.state[1,vsucc,min(tempASAP,currASAP):max(tempASAP,currASAP)] = 0 if currASAP > tempASAP else 1
			if currASAP < tempASAP:
				reward += self.reward["small"]
		self.totLatency = max(self.totLatency, step + self.adjlist[op].delay) # step start from 0
		oldNr = self.currNr[tempR]
		self.currNr[tempR] = self.nrt[tempR].max()
		if mode != 0:
			if self.currNr["MUL"] != 0 and self.currNr["ALU"] != 0 and self.currNr["MUL"] + self.currNr["ALU"] <= self.bestNr["MUL"] + self.bestNr["ALU"]:
				self.bestNr["MUL"], self.bestNr["ALU"] = self.currNr["MUL"], self.currNr["ALU"]
		newOpNr = 0
		for d in range(self.adjlist[op].delay):
			newOpNr += self.nrt[tempR][self.adjlist[op].cstep + d]
		# early stop
		cnt = 0
		legal_move = self.getAllLegalMove()[0]
		for legal_op in legal_move:
			legal_op = self.adjlist[legal_op]
			typeR = self.mapR(legal_op.type)
			if (self.nrt[typeR][legal_op.cstep+1:legal_op.cstep+1+legal_op.delay] + 1
				> self.currNr[typeR]).any():
				cnt += 1
		if cnt >= len(legal_move):
			return False, self.reward["nothing"]
		# final reward
		if self.mode == "RCS":
			reward += 10 / self.totLatency
		else:
			reward += oldNr - self.currNr[tempR]
			# reward += (oldOpNr - newOpNr)/5
		return True, reward

	# mode 0: without recursion
	# mode 1: recursion
	def test_val(self,op,step,mode=0):
		if op < 0 or op >= self.vertex:
			return False
		tempR = self.mapR(self.adjlist[op].type)
		# Constraints
		if self.mode == "RCS":
			if self.nrt[tempR][step] + 1 > self.maxNr[tempR]:
				return False
		else:
			if step + self.adjlist[op].delay - 1 > self.CONSTRAINED_L:
				return False
		if mode == 1:
			return True
		if self.adjlist[op].getASAP() > step or self.adjlist[op].getALAP() < step:
			return False
		for vsucc in self.adjlist[op].succ:
			vsucc = self.adjlist[vsucc]
			if vsucc.cstep > -1 and step + self.adjlist[op].delay - 1 >= vsucc.cstep:
				return False
		for vpred in self.adjlist[op].pred:
			vpred = self.adjlist[vpred]
			if vpred.cstep > -1 and vpred.cstep + vpred.delay > step:
				return False
		return True

	def schedule_node_recursion(self,op,step): # only support top-down
		if not self.test_val(op,step,1):
			return False, self.reward["penalty"]
		delay = self.adjlist[op].delay
		if not self.state[2,op,step:step+delay].all():
			return False, self.reward["penalty"]
		elif self.state[1,op,step:step+delay].all(): # the final operation that needn't move
			return self.schedule_node(op,step)
		if step < self.adjlist[op].cstep:
			return True, 0
		tot_reward = 0
		for vsucc in self.adjlist[op].succ: # move the operations backward
			if self.adjlist[vsucc].cstep < step + delay:
				fes, reward = self.schedule_node_recursion(vsucc,step+delay)
				if fes == False:
					return fes, reward
				else:
					tot_reward += reward
		fes, reward = self.schedule_node(op,step)
		if fes == False:
			return fes, reward
		else:
			tot_reward += reward
			return fes, tot_reward

	def test_final(self):
		flag = True
		for v in self.adjlist:
			for vsucc in v.succ:
				vsucc = self.adjlist[vsucc]
				if v.cstep + v.delay - 1 >= vsucc.cstep:
					flag = False
					print("Schedule conflicts with Node %d(%s) and Node %d(%s)." % (v.num,v.name,vsucc.num,vsucc.name))
					return flag
		return flag

	def get_state(self):
		return self.state

	def get_partial_state(self,size,pos=(0,0)):
		res = np.zeros((3,size[0],size[1]))
		x = min(self.state.shape[1]-pos[0],size[0])
		y = min(self.state.shape[2]-pos[1],size[1])
		res[:,0:x,0:y] = np.copy(self.state)[:,pos[0]:x+pos[0],pos[1]:y+pos[1]]
		return res

	def getNrt(self):
		return self.nrt

	def getAllLegalMove(self):
		res = []
		res_dict = dict()
		cnt = 0
		for (op,row) in enumerate(self.get_state()[1,:,:]):
			if (row[self.adjlist[op].cstep:] == 1).any(): # backward!
				res.append(op)
				res_dict[cnt] = op
				cnt += 1
		return (res,res_dict)

	def getLegalMove(self,pos=(0,0)):
		res = []
		res_dict = dict()
		cnt = 0
		for (op,row) in enumerate(self.get_state()[1,:,:]):
			if pos[0] <= op < pos[0] + 50: # 50!
				if (row[max(pos[1],self.adjlist[op].cstep):] == 1).any(): # backward!
					res.append(op-pos[0])
					res_dict[cnt] = op - pos[0]
					cnt += 1
		return (res,res_dict)

	def output_adjlist(self):
		print("Adjacent List:")
		for v in self.adjlist:
			print("Node %d(%s):" % (v.num,v.name),end=" ")
			for op in v.succ:
				print(op+1,end=" ")
			print()

	def output_axap(self):
		print("AXAP:")
		for v in self.adjlist:
			print("Node %d(%s): [%d, %d]" % (v.num,v.name,v.getASAP(),v.getALAP()))

	def output(self):
		print("# of operations: %d" % self.vertex)
		print("Latency factor: %f, CONSTRAINED_L: %d, Mul_delay: %d" % (self._LC,self.CONSTRAINED_L+1,self.mul_delay))
		print("Best # of resources: MUL: %d, ALU: %d" % (self.bestNr["MUL"], self.bestNr["ALU"]))
		print("Current # of resources: MUL: %d, ALU: %d" % (self.currNr["MUL"], self.currNr["ALU"]))
		print("Latency: %d" % self.totLatency)
		print("Schedule: ")
		for v in self.adjlist:
			print("Node %d(%s): %d" % (v.num,v.name,v.cstep))