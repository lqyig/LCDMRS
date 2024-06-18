import math
import operator
import time
import func_timeout
import networkx as nx
import numpy as np
from func_timeout import func_set_timeout
from sklearn.decomposition import NMF
import sys
import gc

import warnings
warnings.filterwarnings("ignore")

gc.enable()

r_c = 0
# 生成社交网络
def Get_SocNetwork(node_file, graph_file):
    G = nx.Graph()
    nodelist = []
    with open(node_file) as f:
        lines = f.readlines()
        for line in lines:
            curLine = line.strip().split(" ")
            nodelist.append(int(curLine[0]))
    G.add_nodes_from(nodelist)
    edgelist = []
    with open(graph_file) as f:
        lines = f.readlines()
        for line in lines:
            curLine = line.strip().split(" ")
            edgelist.append((int(curLine[0]), int(curLine[1])))
    G.add_edges_from(edgelist)
    return G

# 读取社交网络-道路网络节点之间的映射
def Mapping_node(mapping_file):
    node_mapping = dict()
    with open(mapping_file) as f:
        lines = f.readlines()
        for line in lines:
            curLine = line.strip().split(" ")
            node_mapping[int(curLine[0])] = int(curLine[1])
    return node_mapping

# 生成道路网络
def Get_Roadnetwork(node_file,graph_file):
    """
    道路网络中节点的位置
    """
    road_node_loc = {}
    with open(node_file) as f:
        lines = f.readlines()
        for line in lines:
            curLine = line.strip().split(" ")
            road_node_loc.update({int(curLine[0]): [float(curLine[1]), float(curLine[2])]})
    """
    道路网络边的信息
    """
    G = nx.Graph()
    edgelist = []
    with open(graph_file) as f:
        lines = f.readlines()
        for line in lines:
            curLine = line.strip().split(" ")
            edgelist.append((int(curLine[1]), int(curLine[2])))
    G.add_edges_from(edgelist)
    return G, road_node_loc

# 生成由M方法得到的社区的子图
def Subgraph(g,M_com):
    sub_g = nx.Graph()
    edgelist = set()
    for i in M_com:
        for j in M_com:
            if (i, j) in g.edges():
                edgelist.add((i,j))
    sub_g.add_edges_from(list(edgelist))
    return sub_g

# 生成属性矩阵X
def Gen_attr_mat(file,M_Com):
    nodedict = {}
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            curLine = line.strip().split(" ")
            if int(curLine[0]) in M_Com:
                nodedict[int(curLine[0])] = [round(float(float(curLine[1])*1000000/15),1),round(float(float(curLine[2])*1000000/15),1),round(float(float(curLine[3])*1000000/15),1)]
    new_nodedict = {}
    for i in M_Com:
        new_nodedict[i] = nodedict[i]
    nodelist = []
    for i in new_nodedict.keys():
        nodelist.append(new_nodedict[i])
    X = np.array(nodelist)
    return X

def calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma):
    temp1 = A - np.matmul(G, H)
    temp1 = np.multiply(temp1, temp1)
    temp1 = np.multiply(np.log(np.reciprocal(outl1)), np.sum(temp1, axis=1))
    temp1 = np.sum(temp1)

    temp2 = C - np.matmul(U, V)
    temp2 = np.multiply(temp2, temp2)
    temp2 = np.multiply(np.log(np.reciprocal(outl2)), np.sum(temp2, axis=1))
    temp2 = np.sum(temp2)

    temp3 = G.T - np.matmul(W, U.T)
    temp3 = np.multiply(temp3, temp3)
    temp3 = np.multiply(np.log(np.reciprocal(outl3)), np.sum(temp3, axis=0).T)
    temp3 = np.sum(temp3)

    # print('\t Component values: {},{} and {}'.format(temp1, temp2, temp3))

    func_value = alpha * temp1 + beta * temp2 + gamma * temp3

    # print('\t Total Function value {}'.format(func_value))

# 生成嵌入矩阵
def Matrix_decom(A, C):
    K = 2
    W = np.eye(K)
    # print('Dimension of C: {}, {}'.format(C.shape[0], C.shape[1]))
    gc.collect()
    mu = 1
    gc.collect()
    model = NMF(n_components=K, init='random', random_state=0)
    G = model.fit_transform(A)
    H = model.components_
    model = NMF(n_components=K, init='random', random_state=0)
    U = model.fit_transform(C)
    V = model.components_
    outl1 = np.ones((A.shape[0]))
    outl2 = np.ones((A.shape[0]))
    outl3 = np.ones((A.shape[0]))
    Graph = nx.from_numpy_array(A)
    bet = nx.betweenness_centrality(Graph)
    for i in range(len(outl1)):
        outl1[i] = float(1) / A.shape[0] + bet[i]
        outl2[i] = float(1) / A.shape[0]
        outl3[i] = float(1) / A.shape[0] + bet[i]
    outl1 = outl1 / sum(outl1)
    outl2 = outl2 / sum(outl2)
    outl3 = outl3 / sum(outl3)

    count_outer = 5  # Number of outer Iterations for optimization

    temp1 = A - np.matmul(G, H)
    temp1 = np.multiply(temp1, temp1)
    temp1 = np.multiply(np.log(np.reciprocal(outl1)), np.sum(temp1, axis=1))
    temp1 = np.sum(temp1)

    temp2 = C - np.matmul(U, V)
    temp2 = np.multiply(temp2, temp2)
    temp2 = np.multiply(np.log(np.reciprocal(outl2)), np.sum(temp2, axis=1))
    temp2 = np.sum(temp2)

    temp3 = G.T - np.matmul(W, U.T)
    temp3 = np.multiply(temp3, temp3)
    temp3 = np.multiply(np.log(np.reciprocal(outl3)), np.sum(temp3, axis=0).T)
    temp3 = np.sum(temp3)

    alpha = 1
    beta = temp1 / temp2
    gamma = min(2 * beta, temp3)

    for passNo in range(1):  # mu = 1 is good enough for all datasets
        # print('Pass {} Started'.format(passNo))
        for opti_iter in range(count_outer):
            # print('Loop {} started: \n'.format(opti_iter))
            # print("The function values which we are interested are : ")
            calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma)

            # The Update rule for G[i,k]
            for i in range(G.shape[0]):
                for k in range(G.shape[1]):
                    Gik_numer = alpha * np.log(np.reciprocal(outl1[i])) * np.dot(H[k, :], (
                                A[i, :] - (np.matmul(G[i], H) - np.multiply(G[i, k], H[k, :])))) + \
                                gamma * np.log(np.reciprocal(outl3[i])) * np.dot(U[i], W[k, :])
                    Gik_denom = alpha * np.log(np.reciprocal(outl1[i])) * np.dot(H[k, :], H[k, :]) + gamma * np.log(
                        np.reciprocal(outl3[i]))
                    G[i, k] = Gik_numer / Gik_denom
            calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma)
            # print('Done for G')

            # The update rule for H[k,j]
            for k in range(H.shape[0]):
                for j in range(H.shape[1]):
                    Hkj_numer = alpha * np.dot(np.multiply(np.log(np.reciprocal(outl1)), G[:, k]),
                                               (A[:, j] - (np.matmul(G, H[:, j]) - np.multiply(G[:, k], H[k, j]))))
                    Hkj_denom = alpha * (np.dot(np.log(np.reciprocal(outl1)), np.multiply(G[:, k], G[:, k])))
                    H[k, j] = Hkj_numer / Hkj_denom

            calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma)
            # print('Done for H')

            # The update rule for U[i,k]
            for i in range(U.shape[0]):
                for k in range(U.shape[1]):
                    Uik_numer_1 = beta * np.log(np.reciprocal(outl2[i])) * (
                        np.dot(V[k, :], (C[i] - (np.matmul(U[i, :], V) - np.multiply(U[i, k], V[k, :])))))
                    Uik_numer_2 = gamma * np.log(np.reciprocal(outl3[i])) * np.dot(
                        (G[i, :] - (np.matmul(U[i, :], W) - np.multiply(U[i, k], W[:, k]))), W[:, k])
                    Uik_denom = beta * np.log(np.reciprocal(outl2[i])) * np.dot(V[k, :], V[k, :]) + gamma * np.log(
                        np.reciprocal(outl3[i])) * np.dot(W[:, k], W[:, k])
                    U[i, k] = (Uik_numer_1 + Uik_numer_2) / Uik_denom

            calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma)
            # print('Done for U')

            # The update rule for V[k,d]
            for k in range(V.shape[0]):
                for d in range(V.shape[1]):
                    Vkd_numer = beta * np.dot(np.multiply(np.log(np.reciprocal(outl2)), U[:, k]),
                                              (C[:, d] - (np.matmul(U, V[:, d]) - np.multiply(U[:, k], V[k, d]))))
                    Vkd_denom = beta * (np.dot(np.log(np.reciprocal(outl2)), np.multiply(U[:, k], U[:, k])))
                    V[k][d] = Vkd_numer / Vkd_denom

            calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma)
            # print('Done for V')

            # The Update rule for W[p,q]
            logoi = np.log(np.reciprocal(outl3))
            sqrt_logoi = np.sqrt(logoi)
            sqrt_logoi = np.tile(sqrt_logoi, (K, 1))
            assert (sqrt_logoi.shape == G.T.shape)

            term1 = np.multiply(sqrt_logoi, G.T)
            term2 = np.multiply(sqrt_logoi, U.T)

            svd_matrix = np.matmul(term1, term2.T)


            svd_u, svd_sigma, svd_vt = np.linalg.svd(svd_matrix)

            W = np.matmul(svd_u, svd_vt)

            calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma)
            # print('Done for W')

            # The update rule for outl
            GH = np.matmul(G, H)
            UV = np.matmul(U, V)
            WUTrans = np.matmul(W, U.T)

            outl1_numer = alpha * (np.multiply((A - GH), (A - GH))).sum(axis=1)

            outl1_denom = alpha * pow(np.linalg.norm((A - GH), 'fro'), 2)

            outl1_numer = outl1_numer * mu
            outl1 = outl1_numer / outl1_denom

            outl2_numer = beta * (np.multiply((C - UV), (C - UV))).sum(axis=1)

            outl2_denom = beta * pow(np.linalg.norm((C - UV), 'fro'), 2)

            outl2_numer = outl2_numer * mu
            outl2 = outl2_numer / outl2_denom

            outl3_numer = gamma * (np.multiply((G.T - WUTrans), (G.T - WUTrans))).sum(axis=0).T

            outl3_denom = gamma * pow(np.linalg.norm((G.T - WUTrans), 'fro'), 2)

            outl3_numer = outl3_numer * mu
            outl3 = outl3_numer / outl3_denom

            calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma)
            # print('Done for outlier score')
    AX = (G + np.dot(U, W.T)) / 2
    return AX

# 生成“节点-向量”字典
def Node_Vec(M_C,AX):
    node_vecs = {}
    for i in range(0, len(M_C)):
        node_vecs[M_C[i]] = AX[i]

    return node_vecs

# 计算两点之间线段的距离
def __line_magnitude(x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude

# 求某点到某线段的距离
def __point_to_line_distance(px,py,x1,x2,y1,y2):
    line_magnitude = __line_magnitude(x1, y1, x2, y2)
    if line_magnitude < 0.00000001:
        return 9999,0,0
    else:
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (line_magnitude * line_magnitude)
        if (u < 0.00001) or (u > 1):
            # 点到直线的投影不在线段内, 计算点到两个端点距离的最小值即为"点到线段最小距离"
            ix = __line_magnitude(px, py, x1, y1)
            iy = __line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance = iy
            else:
                distance = ix
        else:
            # 投影点在线段内部, 计算方式同点到直线距离, u 为投影点距离x1在x1x2上的比例, 以此计算出投影点的坐标
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = __line_magnitude(px, py, ix, iy)
        return distance,ix,iy

# 求点到线段的投影点
def point_to_segment_projection(px,py,x1,y1,x2,y2):
    """
    求点到线段的投影点
    参数：
    point: 二元组，表示点的坐标（x, y）
    segment: 二元组，表示线段的两个端点坐标，格式为 ((x1, y1), (x2, y2))
    返回值：
    二元组，表示投影点的坐标（x, y）
    """
    # 计算线段的长度和方向
    dx = x2 - x1
    dy = y2 - y1
    segment_length_squared = dx ** 2 + dy ** 2
    # 如果线段长度为0，则返回线段的起点作为投影点
    if segment_length_squared == 0:
        return x1, y1
    # 计算点到线段起点的向量
    u = ((px - x1) * dx + (py - y1) * dy) / segment_length_squared
    # 如果点在线段的起点左侧，则返回线段的起点作为投影点
    if u < 0:
        return x1, y1
    # 如果点在线段的终点右侧，则返回线段的终点作为投影点
    if u > 1:
        return x2, y2
    # 计算投影点的坐标
    proj_x = x1 + u * dx
    proj_y = y1 + u * dy
    return proj_x, proj_y


"""
——————————————————————————————————————————————————————————————————
参数：
    Gr  道路网络
    graph_nodeloc  道路网络中的节点位置
    start_node  开始节点
    end_node  目标节点

return：
    path  从开始节点到目标节点的最短路径
———————————————————————————————————————————————————————————————————
"""
# 计算两点之间的最短路径
def Shortest_path(Gr,graph_nodeloc,start_node,end_node):
    d_length,p_length,f_length,opentable,closetable = {},{},{},{},{}
    C_n = set(Gr.neighbors(start_node))- {start_node}
    parent = {}
    for i in C_n:
        x1 = graph_nodeloc[start_node][0]-graph_nodeloc[i][0]
        y1 = graph_nodeloc[start_node][1]-graph_nodeloc[i][1]
        x2 = graph_nodeloc[i][0]-graph_nodeloc[end_node][0]
        y2 = graph_nodeloc[i][1]-graph_nodeloc[end_node][1]
        p_length[i] = math.sqrt(x1**2+y1**2)
        d_length[i] = math.sqrt(x2**2+y2**2)
        f_length[i] = p_length[i] + d_length[i]
        opentable[i] = f_length[i]
        parent[i] = start_node
    while(True):
        opentable = sorted(opentable.items(), key=lambda x: x[1])
        opentable = dict(opentable)
        min_node = list(opentable.keys())[0]
        closetable[min_node] = opentable[min_node]
        if (min_node == end_node):
            break
        opentable.pop(min_node)
        C_n = set(Gr.neighbors(min_node))-{start_node}
        for i in C_n:
            x1 = graph_nodeloc[min_node][0] - graph_nodeloc[i][0]
            y1 = graph_nodeloc[min_node][1] - graph_nodeloc[i][1]
            x2 = graph_nodeloc[i][0] - graph_nodeloc[end_node][0]
            y2 = graph_nodeloc[i][1] - graph_nodeloc[end_node][1]
            upgrade_p = p_length[min_node]+math.sqrt(x1**2+y1**2)
            d_length[i] = math.sqrt(x2**2+y2**2)
            upgrade = upgrade_p + d_length[i]
            if i in opentable.keys() and upgrade < f_length[i]:
                p_length[i] = upgrade_p
                f_length[i] = upgrade
                parent[i] = min_node
            if i in closetable.keys() and upgrade < f_length[i]:
                p_length[i] = upgrade_p
                closetable.pop(i)
                opentable[i] = upgrade
                f_length[i] = upgrade
                parent[i] = min_node
            if i not in opentable.keys() and i not in closetable.keys():
                p_length[i] = upgrade_p
                f_length[i] = upgrade
                opentable[i] = f_length[i]
                parent[i] = min_node
    pathlength = closetable[end_node]
    path = [end_node]
    while True:
        end = parent[end_node]
        path.append(end)
        end_node = end
        if end_node == start_node:
            break
    return round(pathlength, 8), path

"""
——————————————————————————————————————————————————————————————————
作用：找到社区的中心位置在最近边上的投影点r_c
参数：
    Gr  道路网络
    graph_nodeloc  道路网络中的节点位置
    c  社区
    path  从社区的道路中心位置到节点i的最短路径
    px,py  社区中心坐标
    com_nodeloc_x,com_nodeloc_y  社区内所有节点的横、纵坐标

return：
    Gr  道路网络
    r_c  社区的道路中心位置在最近边上的投影点
    add_edges  将投影点加入道路网络后的新增边
———————————————————————————————————————————————————————————————————
"""
def Find_all_roads(Gr,road_node_loc,c,path,px,py,com_nodeloc_x,com_nodeloc_y):
    global r_c
    """计算C ∪ {path}的边界"""
    C_Nodes_x, C_Nodes_y = com_nodeloc_x[:], com_nodeloc_y[:]
    for j in path:
        C_Nodes_x.append(road_node_loc[j][0])
        C_Nodes_y.append(road_node_loc[j][1])
    C_Nodes_x_arr, C_Nodes_y_arr = np.array(C_Nodes_x), np.array(C_Nodes_y)
    x_min, x_max = C_Nodes_x_arr.min(), C_Nodes_x_arr.max()
    y_min, y_max = C_Nodes_y_arr.min(), C_Nodes_y_arr.max()
    """得到边界内的所有节点"""
    c_r = set()
    for i in c:
        c_r.add(node_mapping[i])
    opentable = c_r | path
    opentable_temp = opentable
    while True:
        N = set()
        for u in opentable:
            Nu = list(Gr.neighbors(u))
            N = set(Nu) | set(N)
        N = N - opentable
        if len(N) == 0:
            break
        for v in N:
            if x_min <= road_node_loc[v][0] <= x_max and y_min <= road_node_loc[v][1] <= y_max:
                opentable.add(v)
        if operator.eq(opentable, opentable_temp):
            break
        opentable_temp = opentable
    # print("opentable", opentable)
    """得到由边界内所有节点组成的边"""
    edgelist = []
    for u in opentable:
        for v in opentable:
            if (u, v) in Gr.edges():
                edgelist.append((u, v))
    # print("edgelist", edgelist)
    """求社区中心坐标距道路的最短距离的投影坐标"""
    edge_dist_dict = {}
    for (i, j) in edgelist:
        r_x, r_y = point_to_segment_projection(px, py, road_node_loc[i][0], road_node_loc[i][1], road_node_loc[j][0], road_node_loc[j][1])
        distance = math.sqrt((px-r_x)**2+(py-r_y)**2)
        edge_dist_dict[(i, j)] = [distance, r_x, r_y]
    edge_dist_dict = sorted(edge_dist_dict.items(), key=lambda x: x[1][0])
    # print(edge_dist_dict)
    r_c = r_c - 1
    # print("edge_dist_dict", edge_dist_dict)
    add_edges = [(edge_dist_dict[0][0][0], r_c), (r_c, edge_dist_dict[0][0][1])]
    Gr.add_edges_from([(edge_dist_dict[0][0][0], r_c), (r_c, edge_dist_dict[0][0][1])])
    road_node_loc[r_c] = [round(float(edge_dist_dict[0][1][1]), 8), round(float(edge_dist_dict[0][1][2]), 8)]  # 投影坐标
    return Gr, r_c, road_node_loc, add_edges

"""
——————————————————————————————————————————————————————————————————
作用：计算目标值D
参数：
    Gr  道路网络
    c  社区
    graph_nodeloc  道路网络中的节点位置
    r_c  社区的道路中心位置在最近边上的投影点

return：D
———————————————————————————————————————————————————————————————————
"""
# 计算目标D
def RecomputeD(Gr, road_node_loc, node_mapping, c, r_c):
    """
    作用：增量计算目标值D
    参数：
        Gr  道路网络
        road_node_loc  道路网络中的节点位置
        node_mapping 社交-道路的节点映射关系
        c  当前社区
        j  待加入社区的节点
        sum_path_distance  当前社区节点之间的最短路径之和
    return：D
    """
    c_path_len = 0
    for i in c:
        if r_c == node_mapping[i]:
            short_path = 0
        else:
            short_path, parent = Shortest_path(Gr, road_node_loc, r_c, node_mapping[i])
        c_path_len = c_path_len + short_path
    num = len(c)
    if c_path_len == 0:
        c_path_len = 0.0001
    # print(c_path_len)
    D = round(num**3/(3.14*(c_path_len**2)), 5)
    # D = - round(c_path_len**2/num**2, 5)
    # D = - round(c_path_len/num, 8)
    return D
"""
——————————————————————————————————————————————————————————————————
作用：计算目标值E
return：E
———————————————————————————————————————————————————————————————————
"""
def RecomputeMS(gc, c, j, node_vecs, sum_sim):
    """
    作用：计算目标值M
    return：M
    """
    """社区节点之间的余弦相似度之和"""
    j_arr = np.array([node_vecs[j][0], node_vecs[j][1]])
    cos_in = 0
    for i in c:
        i_arr = np.array([node_vecs[i][0], node_vecs[i][1]])
        cos_sim = i_arr.dot(j_arr) / (np.linalg.norm(i_arr) * np.linalg.norm(j_arr))
        cos_in = cos_in + cos_sim
    cos_in_sum = cos_in + sum_sim
    M = round((cos_in_sum-1) / (len(c) * (len(c)-1)), 5)
    return M, cos_in
# def RecomputeMS(gc, c, j, node_vecs, sum_sim):
#     """
#     作用：计算目标值M
#     return：M
#     """
#     """社区节点之间的余弦相似度之和"""
#     j_arr = np.array([node_vecs[j][0], node_vecs[j][1]])
#     cos_in = 0
#     for i in c:
#         if (i,j) in gc.edges():
#             i_arr = np.array([node_vecs[i][0], node_vecs[i][1]])
#             cos_sim = i_arr.dot(j_arr) / (np.linalg.norm(i_arr) * np.linalg.norm(j_arr))
#             cos_in = cos_in + cos_sim
#     cos_in_sum = cos_in + sum_sim
#
#     """社区邻居节点与社区节点之间的余弦相似度之和"""
#     cos_out = 0
#     for u in c:
#         u_arr = np.array([node_vecs[u][0], node_vecs[u][1]])
#         for v in gc.neighbors(u):
#             v_arr = np.array([node_vecs[v][0], node_vecs[v][1]])
#             cos_sim = u_arr.dot(v_arr) / (np.linalg.norm(u_arr) * np.linalg.norm(v_arr))
#             cos_out = cos_out + cos_sim
#     M = round(cos_in_sum / cos_out, 5)
#     return abs(M), cos_in
#
# def ComSpa(Gr, road_nodes_loc, c):
#     sum_path_len = 0
#     for i in c:
#         for j in c:
#             if node_mapping[i] != node_mapping[j]:
#                 shortest_path, parent = Shortest_path(Gr, road_nodes_loc, node_mapping[i], node_mapping[j])
#             else:
#                 shortest_path = 0
#             sum_path_len = sum_path_len + shortest_path
#     D = round(sum_path_len / (len(c) * (len(c) - 1)), 8)
#     return D

# def ComComm(Gs, c):
#     Ec_in = 0
#     for u in c:
#         for v in c:
#             if v in Gs.neighbors(u):
#                 Ec_in = Ec_in + 1
#     Ec_in = Ec_in / 2
#     N = len(c)
#     Q = 2 * Ec_in / (N * (N - 1))
#     sum_edges, sum_deg = 0, 0
#     for i in Gs.nodes():
#         sum_edges = sum_edges + len(list(Gs.neighbors(i)))
#     sum_edges = sum_edges / 2
#     for u in c:
#         u_Degree = len(list(Gs.neighbors(u)))
#         sum_deg = sum_deg + u_Degree
#     x = (sum_deg / (2 * sum_edges)) ** 2
#     communitude = ((Ec_in / sum_edges) - x) / math.sqrt(x * (1 - x))
#     return round(communitude, 4)

"""
获得候选节点
"""
def obtain_candidates(Gc, Gr, C_n, c, MS_old, D_old, node_vecs, road_node_loc, node_mapping, sum_sim, r_c, X, Y):
    candidate_nodes = {}
    for i in C_n:
        c.append(i)
        """求上一轮的社区中心映射点到节点i的最短路径"""
        # 映射中心不是新节点
        if r_c == node_mapping[i]:
            D = RecomputeD(Gr, road_node_loc, node_mapping, c, r_c)
            MS, cos_in = RecomputeMS(Gc, c, i, node_vecs, sum_sim)
            # DDD = ComSpa(Gr, road_node_loc, c)
            # MMM = ComComm(G_S, c)
            c.remove(i)
        # 映射中心是新节点
        else:
            short_path, path = Shortest_path(Gr, road_node_loc, r_c, node_mapping[i])
            """社区C∪{i}中心的坐标"""
            X.append(road_node_loc[node_mapping[i]][0])
            Y.append(road_node_loc[node_mapping[i]][1])
            px, py = np.array(X).mean(), np.array(Y).mean()
            """计算C∪{i}的中心道路位置到边界内节点组成的每条边的距离，并得出最近距离在边上的映射点"""
            Gr, r_c_temp, road_node_loc, add_edges = Find_all_roads(Gr, road_node_loc, c, set(path), px, py, X, Y)
            """计算D和MS"""
            MS, cos_in = RecomputeMS(Gc, c, i, node_vecs, sum_sim)
            D = RecomputeD(Gr, road_node_loc, node_mapping, c, r_c_temp)
            # DDD = ComSpa(Gr, road_node_loc, c)
            # MMM = ComComm(G_S, c)
            c.remove(i)
            X.remove(road_node_loc[node_mapping[i]][0])
            Y.remove(road_node_loc[node_mapping[i]][1])
            Gr.remove_node(r_c_temp)  # 删除节点r_c_temp以及与节点r_c_temp相关联的边
        # print("add node " + str(i) + "，MS and D are :" + str(round(MS, 5)) + "；" + str(round(D, 5)))
        if D_old == 0 and D == 0:
            candidate_nodes[i] = [MS, D]
        if not ((MS_old >= MS and D_old > D) or (MS_old > MS and D_old >= D)):
            candidate_nodes[i] = [MS, D]
    return candidate_nodes

"""
社区扩张
"""
def community_expansion(Gc, Gr, candidate_nodes, c, MS_old, D_old, road_node_loc, node_mapping, node_vecs, sum_sim, flag, History_Opt_community, r_c, X, Y):
    C_old = c[:] # 记录上一轮的社区
    candidate_nodes_mmd = {}
    """对候选结点按降序排序"""
    # 如果是第一轮扩张，则按 D 值排序
    if flag == 1:
        for i in candidate_nodes:
            candidate_nodes_mmd[i] = [candidate_nodes[i][1], candidate_nodes[i][0]]
            # print("MS'、D' and MMD of add node " + str(i) + " :", str("M'="), str(round(ms_n, 5)), str("D'="), str(round(d_n, 5)), str("MMD="), str(round(MMD, 5)))
        candidate_nodes_mmd = sorted(candidate_nodes_mmd.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)  # 对候选节点按其D、M值降序排序
    # 如果是第二轮及以后的扩张，则按 SD 值排序
    else:
        for i in candidate_nodes:
            ms_n = (candidate_nodes[i][0] - MS_old) / MS_old
            d_n = (candidate_nodes[i][1] - D_old) / D_old
            MMD = round(ms_n + d_n, 5)
            # print("MS'、D' and MMD of add node " + str(i) + " :", str("M'="), str(round(ms_n, 5)), str("D'="), str(round(d_n, 5)), str("MMD="), str(round(MMD, 5)))
            candidate_nodes_mmd[i] = [MMD, candidate_nodes[i][1], candidate_nodes[i][0]]
        candidate_nodes_mmd = sorted(candidate_nodes_mmd.items(), key=lambda x: (x[1][0], x[1][1], x[1][2]), reverse=True)  # 对候选节点按其MMD、D值降序排序
    candidate_nodes_sorted = [key for key, value in candidate_nodes_mmd]

    """社区扩张"""
    opt_communities = {}  # 记录各扩张社区的信息
    all_node = set()
    for i in candidate_nodes_sorted:
        c.append(i)
        # 映射中心不是新节点
        if r_c == node_mapping[i]:
            D = RecomputeD(Gr, road_node_loc, node_mapping, c, r_c)
            MS, cos_in = RecomputeMS(Gc, c, i, node_vecs, sum_sim)
            # DDD = ComSpa(Gr, road_node_loc, c)
            # MMM = ComComm(G_S, c)
            sum_sim = cos_in + sum_sim
            opt_communities[tuple(c)] = [MS, D, sum_sim, X[:], Y[:], r_c, []]
        # 映射中心是新节点
        else:
            short_path, path = Shortest_path(Gr, road_node_loc, r_c, node_mapping[i])
            all_node = all_node | set(path)
            """社区C ∪ {i}的中心位置Vc"""
            X.append(road_node_loc[node_mapping[i]][0])
            Y.append(road_node_loc[node_mapping[i]][1])
            px, py = np.array(X).mean(), np.array(Y).mean()
            """计算C的中心坐标到边界内节点组成的图上每条边的距离，并得出最近距离在边上的映射点"""
            Gr, r_c, road_node_loc, add_edges = Find_all_roads(Gr, road_node_loc, c, all_node, px, py, X, Y)
            """计算D和MS"""
            D = RecomputeD(Gr, road_node_loc, node_mapping, c, r_c)
            MS, cos_in = RecomputeMS(Gc, c, i, node_vecs, sum_sim)
            # DDD = ComSpa(Gr, road_node_loc, c)
            # MMM = ComComm(G_S, c)
            sum_sim = cos_in + sum_sim
            opt_communities[tuple(c)] = [MS, D, sum_sim, X[:], Y[:], r_c, add_edges]
            # opt_communities[tuple(c)] = [MS, D, sum_sim, X[:], Y[:], r_c, add_edges, MMM, DDD]
        # print("the community ", c, " MS and D are :" + str(round(MS, 5)) + "；" + str(round(D, 5)))
    c_opt_comms = {}
    if flag == 1:
        for i in opt_communities:
            c_opt_comms[i] = [opt_communities[i][1], opt_communities[i][0]]
    else:
        # print("M_old", MS_old)
        # print("D_old", D_old)
        for i in opt_communities:
            ms_n = (opt_communities[i][0] - MS_old) / MS_old
            d_n = (opt_communities[i][1] - D_old) / D_old
            MMD = round(ms_n + d_n, 5)
            # print("MS'、D' and MMD of add node " + str(i) + " :", str("M'="), str(round(ms_n, 5)), str("D'="), str(round(d_n, 5)), str("MMD="), str(round(MMD, 5)))
            c_opt_comms[i] = [MMD, d_n]
    c_opt_mmd = sorted(c_opt_comms.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)  # 对候选节点按其MMD、D值降序排序
    C_opt = [key for key, value in c_opt_mmd][0]  # 本轮的最优社区
    C_opts = {}
    """如果最优社区集为空，则直接将本轮最优社区加入到最优社区集"""
    if len(History_Opt_community) == 0:
        History_Opt_community[C_opt] = [opt_communities[C_opt][0], opt_communities[C_opt][1], opt_communities[C_opt][2],
                                        opt_communities[C_opt][3], opt_communities[C_opt][4], opt_communities[C_opt][5]]
        C_opts[C_opt] = [1]
        print("本轮社区：", C_opt, " M ", opt_communities[C_opt][0], " D ", opt_communities[C_opt][1], " MMD_C_opt ", 1)
        # print("本轮社区：", C_opt, " M ", opt_communities[C_opt][0], " D ", opt_communities[C_opt][1], " MMD_C_opt ", 1, "MMM", opt_communities[C_opt][7], "DDD", opt_communities[C_opt][8])
        return C_opt, C_opts, History_Opt_community, opt_communities
    else:
        # print("M_old", MS_old)
        # print("D_old", D_old)
        """计算本轮最优社区的MD"""
        m_curr_n = (opt_communities[C_opt][0] - MS_old) / MS_old
        d_curr_n = (opt_communities[C_opt][1] - D_old) / D_old
        MMD_C_opt = round(m_curr_n + d_curr_n, 5)
        print("本轮社区：", C_opt, " M ", opt_communities[C_opt][0], " D ", opt_communities[C_opt][1], " MMD_C_opt ", MMD_C_opt)
        # print("本轮社区：", C_opt, " M ", opt_communities[C_opt][0], " D ", opt_communities[C_opt][1], " MMD_C_opt ", MMD_C_opt, "MMM", opt_communities[C_opt][7], "DDD", opt_communities[C_opt][8])
        """计算上一轮社区的MD"""
        m_n = (History_Opt_community[tuple(C_old)][0] - MS_old) / MS_old
        d_n = (History_Opt_community[tuple(C_old)][1] - D_old) / D_old
        mmd = round(m_n + d_n, 5)
        print("上轮社区：", C_old, " M ", History_Opt_community[tuple(C_old)][0], " D ", History_Opt_community[tuple(C_old)][1], " MMD_C_old ", mmd)
        # print("上轮社区：", C_old, " M ", History_Opt_community[tuple(C_old)][0], " D ", History_Opt_community[tuple(C_old)][1], " MMD_C_old ", mmd, "MMM", History_Opt_community[tuple(C_old)][6], "DDD", History_Opt_community[tuple(C_old)][7])
        """如果上一轮社区的MD值大于本轮社区的MD值，则返回上一轮社区作为最终社区"""
        if mmd > MMD_C_opt:
            C_opts[tuple(C_old)] = [mmd]
    return C_opt, C_opts, History_Opt_community, opt_communities

"""
——————————————————————————————————————————————————————————————————
作用：一轮社区扩张的全过程
参数：
    Gc  抽样网络
    Gr  道路网络
    node_vecs  节点向量
    road_node_loc  道路网络中的节点位置
    node_mapping  社交-道路的节点映射关系
    s  查询节点
    flag  记录第几次扩张
return：
    C  最终社区
———————————————————————————————————————————————————————————————————

"""
@func_set_timeout(7200)
def Gen_Community(Gc, Gr, node_vecs, road_node_loc, node_mapping, s):
    """初始化"""
    C, VCTT, C_opt_temp, node_vec = [s], [s], [s], []
    MS_old, D_old, sum_path_dist, sum_sim = 0, 1, 0, 0
    History_Opt_community = {}  # 记录每轮最优社区的信息
    """社区在抽样社交网络上的邻居节点"""
    C_n = list(Gc.neighbors(s))
    r_c = node_mapping[s]
    X, Y = [road_node_loc[node_mapping[s]][0]], [road_node_loc[node_mapping[s]][1]]
    flag = 1
    while True:
        if len(C_n) == 0:
            break

        """step1：求候选节点"""
        candidate_nodes = obtain_candidates(Gc, Gr, C_n, C, MS_old, D_old, node_vecs, road_node_loc, node_mapping, sum_sim, r_c, X, Y)
        if len(candidate_nodes) == 0:
            break

        """step2：对候选节点排序"""
        C_opt, C_opts, History_Opt_community, opt_communities = community_expansion(Gc, Gr, candidate_nodes, C, MS_old, D_old, road_node_loc, node_mapping, node_vecs, sum_sim, flag, History_Opt_community, r_c, X, Y)

        """如果不是第一轮扩张且上一轮社区的MD值大于本轮社区的MD值，则返回上一轮社区作为最终社区"""
        if C_opt not in History_Opt_community.keys() and len(C_opts) != 0:
            C = list(list(C_opts.keys())[0])
            break

        """将本轮最优社区加入到历史最优社区集中"""
        History_Opt_community[C_opt] = [opt_communities[C_opt][0], opt_communities[C_opt][1], opt_communities[C_opt][2],
                                            opt_communities[C_opt][3], opt_communities[C_opt][4], opt_communities[C_opt][5]]
        """记录本轮最优社区的信息"""
        MS_old, D_old, r_c = abs(History_Opt_community[C_opt][0]), History_Opt_community[C_opt][1], History_Opt_community[C_opt][5]
        sum_sim, X, Y = History_Opt_community[C_opt][2], History_Opt_community[C_opt][3], History_Opt_community[C_opt][4]

        C = list(C_opt)
        if operator.eq(VCTT, C):
            break
        """求社区在社交网络上的邻居节点"""
        community_neighbors_nodes = set()
        for u in set(C):
            Nu = list(Gc.neighbors(u))
            community_neighbors_nodes = set(Nu) | set(community_neighbors_nodes)
        C_n = community_neighbors_nodes - set(C)
        VCTT = C_opt[:]
        flag = flag + 1
    return C

# 将得到的社区写入文件
def Write_nodes(community_nodes, i, datas):
    f = open("community_0407/"+str(datas)+"/" + str(i)+".txt", 'w')
    for j in community_nodes:
        f.write(str(j) + ",")
    f.close()

# 读取社区节点
def Read_nodes(i, datas):
    C = []
    f = open("../DMF_M/community/"+str(datas)+"/"+str(i)+".txt", 'r')
    nodes = f.read().strip().split(",")
    nodes.pop(-1)
    for i in nodes:
        C.append(int(i))
    C.sort()
    return C

# 读取社区节点
# def ReadCom(data, query):
#     C= []
#     f = open("community_1110/" + str(data) + "/" + str(query) + ".txt", 'r')
#     nodes = f.read().strip().split(",")
#     nodes.pop(-1)
#     for i in nodes:
#         C.append(int(i))
#     C.sort()
#     return C
#
# def show():
#     soc = "weep"
#     file1 = "../dataset/soc/" + str(soc) + "/edge.txt"
#     file2 = "../dataset/soc/" + str(soc) + "/node.txt"
#     G_S = Get_SocNetwork(file2, file1)
#     C = ReadCom(soc, 15363)
#     GC = Subgraph(G_S, C)  # 得到子图
#     print(GC.nodes)
#     A = np.array(nx.adjacency_matrix(GC).todense())  # 生成子图的邻接矩阵A
#     X = Gen_attr_mat(file2, list(GC.nodes()))  # 生成子图的属性矩阵X
#     AX = Matrix_decom(A, X)  # 生成A与X的嵌入矩阵
#     print(A)
#     print(X)
#     print(AX)

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

if __name__ == '__main__':
    community = "kite"
    soc = "kite"
    road = "nak"
    print(soc)
    # 生成社交网络
    file1 = "../dataset/soc/" + str(soc) + "/edge.txt"
    file2 = "../dataset/soc/" + str(soc) + "/node.txt"
    G_S = Get_SocNetwork(file2, file1)
    # 生成 社交-道路 节点之间的映射
    mapping_file = "../dataset/mapping/" + str(soc) + "_" + str(road) + "_mapping.txt"
    node_mapping = Mapping_node(mapping_file)
    # 生成道路网络
    edge_file = "../dataset/road/" + str(road) + "/edge.txt"
    node_file = "../dataset/road/" + str(road) + "/node.txt"
    G_R, road_node_loc = Get_Roadnetwork(node_file, edge_file)
    # query_list = [40258, 1507, 12581, 4171, 26573, 5935]
    # query_file = "../dataset/query/new_" + str(soc) + "_query.txt"
    # with open(query_file) as qf:
    #     lines = qf.readlines()
    #     for line in lines:
    #         query_list.append(int(line))
    query_list = [2162, 1143, 633, 2172, 2177, 643, 644, 1673, 1674, 2185, 649]
    print(query_list)
    for query in query_list:
        print("query node：", query)
        M_Com = Read_nodes(query, soc)
        GC = Subgraph(G_S, M_Com) # 得到子图
        A = np.array(nx.adjacency_matrix(GC).todense())  # 生成子图的邻接矩阵A
        X = Gen_attr_mat(file2, list(GC.nodes()))  # 生成子图的属性矩阵X
        AX = Matrix_decom(A, X)  # 生成A与X的嵌入矩阵
        node_vecs = Node_Vec(list(GC.nodes()), AX)
        start = time.time()
        try:
            C = Gen_Community(GC, G_R, node_vecs, road_node_loc, node_mapping, query)
            print("最终社区：", list(C), end=" ")
            print("耗时：", time.time() - start)
            Write_nodes(list(C), query, soc)
        except func_timeout.exceptions.FunctionTimedOut:
            pass
    exit()