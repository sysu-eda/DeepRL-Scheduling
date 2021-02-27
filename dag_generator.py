'''
Copyright 2018 Hongzheng Chen
E-mail: chenhzh37@mail2.sysu.edu.cn

This is the implementation of Deep-reinforcement-learning-based scheduler for High-Level Synthesis.

This file contains the random DAG generator.
'''
from graph import Graph
import random

NUM_GRAPH = 5000

class DAGGen(object):
	def __init__(self, num, tot_node=50, min_per_layer=1, max_per_layer=5, link_rate=0.5, mul_rate=0.3):
		res = "digraph {\n"
		res += "    node [fontcolor=black]\n"
		res += "    property [mul=%d,lf=%.1f]\n" % (random.randint(2,5),random.uniform(1.0,2.0))
		nowNode = 0
		edges = []
		pre_layer = []
		while nowNode < tot_node:
			newNode = random.randint(min_per_layer, max_per_layer)
			if nowNode + newNode > tot_node:
				newNode = tot_node - nowNode
			cur_layer = []
			for i in range(nowNode,nowNode + newNode):
				cur_layer.append(i)
			for j in pre_layer:
				for k in cur_layer:
					if random.random() < link_rate:
						edges.append((j,k))
			pre_layer = cur_layer[:]
			nowNode += newNode
		for i in range(tot_node):
			if random.random() < mul_rate:
				typename = "mul"
			else:
				typename = "add"
			res += "    %d [ label = %s ];\n" % (i, typename)
		for (step,edge) in enumerate(edges):
			res += "    %d -> %d [ name = %d ];\n" % (edge[0],edge[1],step)
		res += "}\n"
		output = open("./DAG/dag_" + str(num) + ".dot","w")
		output.write(res)
		output.close()

for i in range(1,NUM_GRAPH+1):
	DAGGen(i,tot_node=random.randint(10,50),mul_rate=random.uniform(0.3,0.5))
	if i % 100 == 0:
		print("Generated %d / %d DAGs." % (i,NUM_GRAPH))