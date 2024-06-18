# 生成社交网络
import math

import networkx as nx
import numpy as np



def Get_SocNetwork(node_file, graph_file):
    nodedict = {}
    with open(node_file) as f:
        lines = f.readlines()
        for line in lines:
            curLine = line.strip().split(" ")
            nodedict[int(curLine[0])] = [float(curLine[1]), float(curLine[2]), float(curLine[3])]
    G = nx.Graph()
    edgelist = []
    with open(graph_file) as f:
        lines = f.readlines()
        for line in lines:
            curLine = line.strip().split(" ")
            edgelist.append((int(curLine[0]), int(curLine[1])))
    G.add_edges_from(edgelist)
    return G, nodedict

# 读取社区节点
def ReadCom(data, query):
    C= []
    f = open("community_com_attr/" + str(data) + "/" + str(query) + ".txt", 'r')
    nodes = f.read().strip().split(",")
    nodes.pop(-1)
    for i in nodes:
        C.append(int(i))
    return C

def Subgraph(g,c):
    sub_g = nx.Graph()
    edgelist = set()
    for i in c:
        for j in c:
            if (i, j) in g.edges():
                edgelist.add((i,j))
    sub_g.add_edges_from(list(edgelist))
    return sub_g

def GetNc(g, c):
    """求社区在社交网络上的邻居节点"""
    community_neighbors_nodes = set()
    for u in set(c):
        Nu = list(g.neighbors(u))
        community_neighbors_nodes = set(Nu) | set(community_neighbors_nodes)
    C_n = community_neighbors_nodes - set(C)
    return C_n

def Compute_Sim(nodedict, c):
    sum_sim = 0
    for i in c:
        W_i = set(nodedict[i])
        for j in c:
            W_j = set(nodedict[j])
            sim = len(W_i & W_j) / len(W_i | W_j)
            sum_sim = sum_sim + sim
    return round(sum_sim / (len(c)*len(c)), 4)

# def Compute_Sim(g, nodedict, c):
#     sum_sim_in = 0
#     for i in c:
#         W_i = set(nodedict[i])
#         for j in c:
#             W_j = set(nodedict[j])
#             sim = len(W_i & W_j) / len(W_i | W_j)
#             sum_sim_in = sum_sim_in + sim
#     sum_sim_out = 0
#     Nc = GetNc(g, c)
#     for i in c:
#         W_i = set(nodedict[i])
#         for j in Nc:
#             W_j = set(nodedict[j])
#             sim = len(W_i & W_j) / len(W_i | W_j)
#             sum_sim_out = sum_sim_out + sim
#     if sum_sim_out == 0:
#         sum_sim_out = 0.1
#     return round(sum_sim_in * len(Nc) / (sum_sim_out * len(c)), 4)


# def Compute_Sim(g, nodedict, c):
#     sum_sim_in = 0
#     for i in c:
#         i_arr = np.array([nodedict[i][0], nodedict[i][1], nodedict[i][2]])
#         for j in c:
#             j_arr = np.array([nodedict[j][0], nodedict[j][1], nodedict[j][2]])
#             if np.linalg.norm(i_arr) * np.linalg.norm(j_arr) != 0:
#                 sim = i_arr.dot(j_arr) / (np.linalg.norm(i_arr) * np.linalg.norm(j_arr))
#                 sum_sim_in = sum_sim_in + sim
#     sum_sim_out = 0
#     Nc = GetNc(g, c)
#     for i in c:
#         i_arr = np.array([nodedict[i][0], nodedict[i][1], nodedict[i][2]])
#         for j in Nc:
#             j_arr = np.array([nodedict[j][0], nodedict[j][1], nodedict[j][2]])
#             if np.linalg.norm(i_arr) * np.linalg.norm(j_arr) != 0:
#                 sim = i_arr.dot(j_arr) / (np.linalg.norm(i_arr) * np.linalg.norm(j_arr))
#                 sum_sim_out = sum_sim_out + sim
#     if sum_sim_out == 0:
#         sum_sim_out = 0.01
#     return round(sum_sim_in * len(Nc) / (sum_sim_out * len(c)), 4)


if __name__ == '__main__':
    community = "kite"
    soc = "kite"
    print(soc)
    # 生成社交网络
    node_file = "../dataset/soc/" + str(soc) + "/node.txt"
    graph_file = "../dataset/soc/" + str(soc) + "/edge.txt"
    G, Nodedict = Get_SocNetwork(node_file, graph_file)
    query_list = []
    query_file = "../dataset/query/new_" + str(soc) + "_query.txt"
    with open(query_file) as qf:
        lines = qf.readlines()
        for line in lines:
            query_list.append(int(line))
    # query_list = {17153, 45444, 19335, 25608, 20105, 26504, 30091, 34567, 38921, 19598, 28305, 24212, 30747, 17435, 22811, 31006, 17662, 24608, 35362, 14118, 24359, 25255, 34601, 22318, 20527, 25264, 16818, 36915, 23477, 22841, 18363, 21563, 19773, 23613, 37436, 21828, 16069, 19014, 16713, 20681, 35275, 18764, 26573, 31820, 38481, 19027, 28245, 22614, 25046, 30170, 32988, 21085, 24926, 19682, 33381, 30311, 30440, 26862, 20731, 36592, 25586, 27250, 23544, 18297, 29563, 28414}
    print(query_list)
    sum_sim = 0
    for query in query_list:
        C = ReadCom(community, query)  # 节点query的社区
        sim = Compute_Sim(Nodedict, C)
        sum_sim = sum_sim + sim
    avg_sim = sum_sim / len(query_list)
    print("avg sim：", round(avg_sim, 4))
    exit()
