'''
Copyright 2018 Hongzheng Chen
E-mail: chenhzh37@mail2.sysu.edu.cn

This is the implementation of Deep-reinforcement-learning-based scheduler for High-Level Synthesis.

This file contains the ILP solver for HLS scheduling.
'''

import pulp
from graph import Graph

class ILPSolver(object):
	def __init__(self, file_num, mul_delay=2, lf=1.0):
		self.schedule = dict()
		g = Graph("TCS",mul_delay)
		g.setLatencyFactor(lf)
		with open("./DAG/dag_%d.dot" % file_num) as infile:
			g.read(infile)
		g.initialize()
		# print("Begin generating ILP formulas for time-constrained scheduling problem...")
		prob = pulp.LpProblem("Time-Constrained Scheduling Problem",pulp.LpMinimize)
		M1 = pulp.LpVariable("MUL",lowBound=1,upBound=None,cat=pulp.LpInteger)
		M2 = pulp.LpVariable("ALU",lowBound=1,upBound=None,cat=pulp.LpInteger)
		prob += M1 + M2, "Minimize the number of FUs"
		# Time frame constraints
		x = pulp.LpVariable.dicts("x",(range(len(g.adjlist)),range(g.getConstrainedL())),lowBound=0,upBound=1,cat=pulp.LpInteger)
		for (i,node) in enumerate(g.adjlist):
			prob += pulp.lpSum([x[i][t] for t in range(node.getASAP(),node.getALAP()+1)]) == 1, ""
		# print("Time frame constraints generated.")
		# Resource constraints
		rowR = []
		for i in range(g.getConstrainedL()):
			rowR.append({"ALU":[],"MUL":[]}) # be careful of share memory
		for (i,node) in enumerate(g.adjlist):
			for t in range(node.getASAP(),node.getALAP()+node.delay):
				rowR[t][g.mapR(node.type)].append(i)
		for t in range(g.getConstrainedL()):
			for typeR in ["ALU","MUL"]:
				if len(rowR[t][typeR]) < 2:
					continue
				else:
					prob += pulp.lpSum([x[i][td] for i in rowR[t][typeR]
						for td in range(max(t-g.adjlist[i].delay+1,0),t+1)]) - (M1 if typeR == "MUL" else M2)<= 0, ""
		# print("Resource constraints generated.")
		# Precedence constraints
		for (i,node) in enumerate(g.adjlist):
			for vsucc in node.succ:
				prob += (pulp.lpSum([(t+1)*x[i][t] for t in range(node.getASAP(),node.getALAP()+1)])
					- pulp.lpSum([(t+1)*x[vsucc][t] for t in range(g.adjlist[vsucc].getASAP(),g.adjlist[vsucc].getALAP()+1)])
					<= (-1)*node.delay), ""
		# print("Precedence constraints generated.")
		# print("Finish ILP generation.")
		prob.writeLP("./ILP_formulation/dag_%d.lp" % (file_num))
		prob.solve()
		# print("MUL = %d" % prob.variablesDict()["MUL"].varValue)
		# print("ALU = %d" % prob.variablesDict()["ALU"].varValue)
		out_file = open("./Sol/dag_%d.sol" % file_num,"w")
		for v in sorted(prob.variables(),key=lambda x: int(x.name.split("_")[1]) if len(x.name.split("_")) != 1 else 0):
			if v.name[0] == "x" and v.varValue == 1:
				op = v.name.split("_")[1]
				cstep = v.name.split("_")[-1]
				self.schedule[int(op)] = int(cstep)
				out_file.write("%s, %s\n" % (op,cstep))
		out_file.close()
		# print("Status:", pulp.LpStatus[prob.status])

	def getOptSchedule(self):
		return self.schedule