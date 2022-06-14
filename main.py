import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from collections import deque
import networkx as nx


# random.seed = 2022

# 读取数据
class InputData:
    nodes = set()
    edges = set()
    ltDict = {}
    hcDict = {}
    slaDict = {}
    quaDict = {}
    demDict = {}

    def __init__(self, UrlNodes, UrlEdges, UrlDemand):
        allNodes = pd.read_csv(UrlNodes)
        for index, row in allNodes.iterrows():
            self.nodes.add(row['node_id'])
            self.ltDict[row['node_id']] = row['lt']
            self.hcDict[row['node_id']] = row['hc']
            self.slaDict[row['node_id']] = row['sla']
        allEdges = pd.read_csv(UrlEdges)
        for index, row in allEdges.iterrows():
            self.edges.add((row['predecessor'], row['successor']))
            self.quaDict[(row['predecessor'], row['successor'])] = row['quantity']
        allDemand = pd.read_csv(UrlDemand)
        for index, row in allDemand.iterrows():
            self.demDict[row['node_id']] = (row['mean'], row['std'])


class Digraph:
    nodes = set()
    edges = set()
    ltDict = {}
    hcDict = {}
    slaDict = {}
    quaDict = {}
    demDict = {}
    predDict = {}
    succDict = {}
    inDegree = {}
    outDegree = {}
    inoutDegree = {}

    def __init__(self, incoming_graph_data=None, **attr):
        if incoming_graph_data:
            self.edges = incoming_graph_data

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, edge):
        self.edges.add(edge)

    def add_nodes_from(self, nodes):
        self.nodes.update(nodes)

    def add_edges_from(self, edges):
        self.edges.update(edges)

    def add_ltDict(self, lt):
        self.ltDict.update(lt)

    def add_hcDict(self, hc):
        self.hcDict.update(hc)

    def add_slaDict_from(self, sla):
        self.slaDict.update(sla)

    def add_quaDict_from(self, qua):
        self.quaDict.update(qua)

    def add_demDict_from(self, dem):
        self.demDict.update(dem)

    def remove_node(self, node):
        self.nodes.remove(node)

    def remove_node_from(self, nodes):
        self.nodes.difference_update(nodes)

    def remove_edge(self, edge):
        self.edges.remove(edge)

    def remove_edge_from(self, edges):
        self.edges.difference_update(edges)

    def clear(self):
        self.edges.clear()
        self.nodes.clear()

    def clear_edges(self):
        self.edges.clear()

    def show_nodes(self):
        return self.nodes

    def has_node(self, node):
        if node in self.nodes:
            return True
        else:
            return False

    def show_edges(self):
        return self.edges

    def has_edge(self, u, v):
        if (u, v) in self.edges:
            return True
        else:
            return False

    def iter_nodes(self):
        it_nodes = iter(self.nodes)
        return it_nodes

    def iter_edges(self):
        it_edges = iter(self.edges)
        return it_edges

    def find_pred_nodes(self):
        self.predDict = findPredNodes(self.edges)

    def find_succ_nodes(self):
        self.succDict = findSuccNodes(self.edges)

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        return len(self.edges)

    def cal_degree(self):
        self.inDegree = {node: 0 for node in self.nodes}
        self.outDegree = {node: 0 for node in self.nodes}
        self.inoutDegree = {node: 0 for node in self.nodes}
        for pred, succ in self.edges:
            self.inDegree[succ] += 1
            self.outDegree[pred] += 1
        for node in self.nodes:
            self.inoutDegree[node] = self.inDegree[node] + self.outDegree[node]


def findPredNodes(edges):
    nodes = {node for edge in edges for node in edge}
    predDict = {node: [] for node in nodes}
    for pred, succ in edges:
        predDict[succ].append(pred)
    return predDict


def findSuccNodes(edges):
    nodes = {node for edge in edges for node in edge}
    succDict = {node: [] for node in nodes}
    for pred, succ in edges:
        succDict[pred].append(succ)
    return succDict


# 拓扑排序
def TopologicalSort(graph: Digraph()):
    topoSort = list()
    graph.find_succ_nodes()
    succDict = graph.succDict
    graph.cal_degree()
    inDegree = graph.inDegree
    queue = deque()
    for node in inDegree:
        if not inDegree[node]:
            queue.append(node)
            topoSort.append(node)
    while queue:
        pre = queue.popleft()
        for succ in succDict[pre]:
            inDegree[succ] -= 1
            if not inDegree[succ]:
                queue.append(succ)
                topoSort.append(succ)
    return topoSort


# 统计成品所需要用到的半成品、原材料数目
def calQuantity(graph: Digraph()):
    nodeQuantity = {node: 0 for node in graph.nodes}
    for i in graph.demDict.keys():
        nodeQuantity[i] = 1
    reverseTopoSort = list(reversed(TopologicalSort(graph)))
    graph.cal_degree()
    inDegree = graph.inDegree
    graph.find_pred_nodes()
    predDict = graph.predDict
    for i in reverseTopoSort:
        if inDegree[i] == 0:
            break
        for j in predDict[i]:
            nodeQuantity[j] += nodeQuantity[i] * graph.quaDict[(j, i)]
    return nodeQuantity


# max cumulative lt
def calCumLt(graph: Digraph()):
    cumLtDict = {node: 0 for node in graph.nodes}
    topoSort = TopologicalSort(graph)
    graph.find_pred_nodes()
    predDict = graph.predDict
    for i in topoSort:
        if not predDict[i]:
            cumLtDict[i] = graph.ltDict[i]
        for j in predDict[i]:
            cumLtDict[i] = max(cumLtDict[i], cumLtDict[j] + graph.ltDict[i])
    return cumLtDict


pData = InputData('manufacture_node_df.csv', 'manufacture_edge_df.csv', 'manufacture_demand_df.csv')
G = Digraph()
G.add_nodes_from(pData.nodes)
G.add_edges_from(pData.edges)
G.add_ltDict(pData.ltDict)
G.add_hcDict(pData.hcDict)
G.add_slaDict_from(pData.slaDict)
G.add_quaDict_from(pData.quaDict)
G.add_demDict_from(pData.demDict)
G.cal_degree()
G.find_pred_nodes()
G.find_succ_nodes()
del pData

print(G.predDict['N001901'])
print(G.succDict['N001693'])
print(G.inoutDegree['N001693'])
print(TopologicalSort(G))
print(calQuantity(G)['N000589'])
print(calCumLt(G)['N001777'])



