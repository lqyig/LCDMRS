# 生成社交网络
import math
import networkx as nx


r_c = 0
def Get_SocNetwork(graph_file):
    G = nx.Graph()
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
    """道路网络中节点的位置"""
    road_node_loc = {}
    with open(node_file) as f:
        lines = f.readlines()
        for line in lines:
            curLine = line.strip().split(" ")
            road_node_loc.update({int(curLine[0]): [float(curLine[1]), float(curLine[2])]})
    # """社交网络中节点在道路网络中的位置"""
    # soc_node_loc = {}
    # for i in node_mapping:
    #     soc_node_loc.update({i: [road_node_loc[node_mapping[i]][0], road_node_loc[node_mapping[i]][1]]})
    """道路网络边的信息"""
    G = nx.Graph()
    edgelist = []
    with open(graph_file) as f:
        lines = f.readlines()
        for line in lines:
            curLine = line.strip().split(" ")
            edgelist.append((int(curLine[1]), int(curLine[2])))
    G.add_edges_from(edgelist)
    return G, road_node_loc

# 读取社区节点
def ReadCom(data, query):
    C= []
    f = open("community_com_attr/" + str(data) + "/" + str(query) + ".txt", 'r')
    nodes = f.read().strip().split(",")
    nodes.pop(-1)
    for i in nodes:
        C.append(int(i))
    return C

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

# 计算空间内聚性指标（两节点之间的平均最短路径）
def ComSpa(Gr, road_nodes_loc, c):
    sum_path_len = 0
    for i in c:
        for j in c:
            if node_mapping[i] != node_mapping[j]:
                shortest_path, parent = Shortest_path(Gr, road_nodes_loc, node_mapping[i], node_mapping[j])
            else:
                shortest_path = 0
            sum_path_len = sum_path_len + shortest_path
    D = round(sum_path_len / (len(c) * (len(c) - 1)), 8)
    return D

def ComComm(Gs, c):
    Ec_in = 0
    for u in c:
        for v in c:
            if v in Gs.neighbors(u):
                Ec_in = Ec_in + 1
    Ec_in = Ec_in / 2
    N = len(c)
    Q = 2 * Ec_in / (N * (N - 1))
    sum_edges, sum_deg = 0, 0
    for i in Gs.nodes():
        sum_edges = sum_edges + len(list(Gs.neighbors(i)))
    sum_edges = sum_edges / 2
    for u in c:
        u_Degree = len(list(Gs.neighbors(u)))
        sum_deg = sum_deg + u_Degree
    x = (sum_deg / (2 * sum_edges)) ** 2
    communitude = ((Ec_in / sum_edges) - x) / math.sqrt(x * (1 - x))
    return round(communitude, 4), round(Q, 4)

if __name__ == '__main__':
    community = "kite"
    soc = "kite"
    road = "nak"
    print(soc)
    # 生成社交网络
    file1 = "../dataset/soc/" + str(soc) + "/edge.txt"
    file2 = "../dataset/soc/" + str(soc) + "/node.txt"
    G_S = Get_SocNetwork(file1)
    # 生成 社交-道路 节点之间的映射
    mapping_file = "../dataset/mapping/" + str(soc) + "_" + str(road) + "_mapping.txt"
    node_mapping = Mapping_node(mapping_file)
    # 生成道路网络
    edge_file = "../dataset/road/" + str(road) + "/edge.txt"
    node_file = "../dataset/road/" + str(road) + "/node.txt"
    G_R, road_nodes_loc = Get_Roadnetwork(node_file, edge_file)
    query_list = []
    query_file = "../dataset/query/new_" + str(soc) + "_query.txt"
    with open(query_file) as qf:
        lines = qf.readlines()
        for line in lines:
            query_list.append(int(line))
    # query_list = []
    print(query_list)
    print(len(query_list))
    sum_comm, sum_coef, sum_D, sum_size = 0, 0, 0, 0
    for query in query_list:
        C = ReadCom(community, query)  # 节点query的社区
        D = ComSpa(G_R, road_nodes_loc, C)  # 计算空间内聚性
        comm, coef = ComComm(G_S, C)  # 计算结构内聚性
        # print("query：", query, " comm：", round(comm, 4), " coef：", round(coef, 4))
        print("query：", query, " comm：", round(comm, 4), " coef：", round(coef, 4), " D：", round(D, 4))
        sum_comm = sum_comm + comm
        sum_coef = sum_coef + coef
        sum_D = sum_D + D
        sum_size = sum_size + len(C)
    avg_comm = sum_comm / len(query_list)
    avg_coef = sum_coef / len(query_list)
    avg_D = sum_D / len(query_list)
    avg_size = sum_size / len(query_list)
    print("avg comm：", round(avg_comm, 4), "avg coef：", round(avg_coef, 4), "avg D：", round(avg_D, 4), "avg size：", round(avg_size, 0))
    exit()
