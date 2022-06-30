import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from collections import deque
import networkx as nx
import dash


# random.seed = 2022

# 读取数据
class InputData:
    def __init__(self, UrlNodes, UrlEdges, UrlDemand):
        self.nodes = set()
        self.edges = set()
        self.ltDict = {}
        self.hcDict = {}
        self.slaDict = {}
        self.quaDict = {}
        self.demDict = {}
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
    def __init__(self, incoming_graph_data=None, **attr):
        self.nodes = set()
        self.edges = set()
        self.ltDict = {}
        self.hcDict = {}
        self.slaDict = {}
        self.quaDict = {}
        self.demDict = {}
        if incoming_graph_data:
            self.edges = incoming_graph_data

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, edge):
        self.edges.add(edge)
        self.nodes.add(x for x in edge)

    def add_nodes_from(self, nodes):
        self.nodes.update(nodes)

    def add_edges_from(self, edges):
        self.edges.update(edges)
        for edge in self.edges:
            for node in edge:
                self.nodes.add(node)

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
        for edge in self.edges:
            if node in edge:
                self.edges.remove(edge)

    def remove_node_from(self, nodes):
        self.nodes.difference_update(nodes)
        for node in nodes:
            for edge in self.edges:
                if node in edge:
                    self.edges.remove(edge)

    def remove_edge(self, edge):
        self.edges.remove(edge)
        for node in edge:
            if self.inoutDegree[node] == 0:
                self.nodes.remove(node)

    def remove_edge_from(self, edges):
        self.edges.difference_update(edges)
        for edge in edges:
            for node in edge:
                if self.inoutDegree[node] == 0:
                    self.nodes.remove(node)

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

    def find_subtree(self, node):
        subtree = set()
        self.find_pred_nodes()
        queue = deque()
        queue.append(node)
        while queue:
            x = queue.popleft()
            for _ in self.predDict[x]:
                subtree.add((_, x))
                queue.append(_)
        return subtree


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


def findLongestPath(graph: Digraph(), node):
    LongestPath = list()
    predDict = graph.predDict
    cumLtDict = calCumLt(graph)
    queue = deque()
    queue.append(node)
    while queue:
        x = queue.popleft()
        y = len(LongestPath)
        for pred in predDict[x]:
            if cumLtDict[pred] + graph.ltDict[x] == cumLtDict[x]:
                LongestPath.append([x, pred])
                queue.append(pred)
        if y == len(LongestPath):
            for pred, succ in LongestPath:
                if pred == x:
                    LongestPath.remove([pred, succ])
    allPath = list()
    stk = list()

    def dfs(_):
        if not predDict[_]:
            allPath.append(stk[:])
            return
        for i, j in LongestPath:
            if i == _:
                stk.append(j)
                dfs(j)
                stk.pop()

    stk.append(node)
    dfs(node)

    return allPath


class SerialGraph(Digraph):
    def __init__(self, numNodes):
        self.nodes = set()
        self.edges = set()
        self.predDict = {}
        self.succDict = {}
        self.inDegree = {}
        self.outDegree = {}
        self.inoutDegree = {}
        for i in range(numNodes):
            self.nodes.add(i)
        for i in range(numNodes - 1):
            self.edges.add((i, i + 1))

        genData = GenData(self.nodes, self.edges)
        self.ltDict = genData.ltDict
        self.hcDict = genData.hcDict
        self.slaDict = genData.slaDict
        self.quaDict = genData.quaDict
        self.demDict = genData.demDict


class DistributionGraph(Digraph):
    def __init__(self, numNodes):
        self.nodes = set()
        self.edges = set()
        self.predDict = {}
        self.succDict = {}
        self.inDegree = {}
        self.outDegree = {}
        self.inoutDegree = {}
        for i in range(numNodes):
            self.nodes.add(i)
        for i in range(numNodes - 1, 0, -1):
            x = random.randint(0, i - 1)
            self.edges.add((x, i))

        genData = GenData(self.nodes, self.edges)
        self.ltDict = genData.ltDict
        self.hcDict = genData.hcDict
        self.slaDict = genData.slaDict
        self.quaDict = genData.quaDict
        self.demDict = genData.demDict


class AssemblyGraph(Digraph):
    def __init__(self, numNodes):
        self.nodes = set()
        self.edges = set()
        self.predDict = {}
        self.succDict = {}
        self.inDegree = {}
        self.outDegree = {}
        self.inoutDegree = {}
        for i in range(numNodes):
            self.nodes.add(i)
        for i in range(numNodes - 1):
            x = random.randint(i + 1, numNodes - 1)
            self.edges.add((i, x))

        genData = GenData(self.nodes, self.edges)
        self.ltDict = genData.ltDict
        self.hcDict = genData.hcDict
        self.slaDict = genData.slaDict
        self.quaDict = genData.quaDict
        self.demDict = genData.demDict


class GeneralGraph(Digraph):
    def __init__(self, numNodes, numEdges):
        self.nodes = set()
        self.edges = set()
        self.predDict = {}
        self.succDict = {}
        self.inDegree = {}
        self.outDegree = {}
        self.inoutDegree = {}
        for i in range(numNodes):
            self.nodes.add(i)
        '''
        if numEdges < numNodes or numEdges > numNodes*(numNodes-1)/2:
            raise Exception()
        '''
        while numEdges:
            x = random.randint(0, numNodes - 1)
            y = random.randint(0, numNodes - 1)
            if x < y and (x, y) not in self.edges:
                self.edges.add((x, y))
                numEdges -= 1
            if x > y and (y, x) not in self.edges:
                self.edges.add((y, x))
                numEdges -= 1

        genData = GenData(self.nodes, self.edges)
        self.ltDict = genData.ltDict
        self.hcDict = genData.hcDict
        self.slaDict = genData.slaDict
        self.quaDict = genData.quaDict
        self.demDict = genData.demDict


class GenData:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.ltDict = {}
        self.hcDict = {}
        self.slaDict = {}
        self.quaDict = {}
        self.demDict = {}
        for node in nodes:
            self.ltDict[node] = random.randint(1, 20)
            self.hcDict[node] = random.random()
        for edge in edges:
            self.quaDict[edge] = random.uniform(0.0, 2.0)
        products = list()
        products.append(len(nodes) - 1)
        for i in range(len(nodes) - 2, 0, ):
            flag = True
            for j in products:
                if (i, j) in edges:
                    flag = False
            if flag:
                products.append(i)
            else:
                break
        for _ in products:
            self.hcDict[_] = random.uniform(50, 100)
            self.slaDict[_] = random.randint(5, 20)
            self.demDict[_] = (random.uniform(10, 100), random.uniform(1, 10))


if __name__ == '__main__':
    pData = InputData('manufacture_node_df.csv', 'manufacture_edge_df.csv', 'manufacture_demand_df.csv')
    G = Digraph()
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
    print(G.succDict['N001661'])
    print(G.inDegree['N001693'])
    print(TopologicalSort(G))
    print(calQuantity(G)['N000589'])
    print(calCumLt(G)['N001777'])
    print(len(G.find_subtree('N001693')))
    print(findLongestPath(G, 'N001661'))
    print(len(G.predDict['N001273']))
    print(len(findLongestPath(G, 'N001661')))

    g1 = GeneralGraph(5, 8)
    print(g1.edges)
    print(g1.nodes)
    print(g1.demDict)
