import copy
import networkx as nx
import matplotlib.pyplot as plt

def LT(G, node, steps=0):  # LT线性阈值算法
    temp = copy.deepcopy(G)

    # 初始化节点阈值
    for n in temp.nodes():
        temp.nodes[n]['threshold'] = 0.5

    # 初始化边的权重
    in_deg = temp.in_degree()  # 获取所有节点的入度
    for edge in temp.edges():
        temp[edge[0]][edge[1]]['weight'] = 1.0 / in_deg[edge[1]]  # 计算边的权重

    Node = copy.deepcopy(node)
    layer_nodes = []
    layer_nodes.append([i for i in Node])
    while steps > 0 and len(Node) < len(temp):
        origin = len(Node)
        nodes, act_nodes = diffuse(temp, Node)
        layer_nodes.append(act_nodes)
        if len(nodes) == origin:
            break
        steps -= 1
    return layer_nodes

def diffuse(G, nodes):
    Act_nodes = set()
    for node in nodes:
        nodes_aft = G.successors(node)
        for node_aft in nodes_aft:
            if node_aft in nodes:  # 若节点已经被激活则不再考虑
                continue

            act_nodes = list(set(G.predecessors(node_aft)).intersection(set(nodes)))
            if weight(G, act_nodes, node_aft) >= G.nodes[node_aft]['threshold']:
                Act_nodes.add(node_aft)
    nodes.extend(list(Act_nodes))
    return nodes, list(Act_nodes)

def weight(G, starts, end):
    weight_sum = 0.0
    for start in starts:
        weight_sum += G[start][end]['weight']
    return weight_sum

if __name__ == '__main__':
    n = 10
    G = nx.generators.directed.gnc_graph(n)

    for i in range(n):
        layers = LT(G, [i], n)  # 调用LT线性阈值算法，返回子节点集和该子节点集的最大激活节点集
        del layers[-1]
        length = 0
        for i in range(len(layers)):
            length = length + len(layers[i])
        lengths = length - len(layers[0])  # 获得子节点的激活节点的个数（长度）
        # 输出能够激活其他节点的结果
        if lengths > 0:
            init_node = layers[0]
            layers.remove(layers[0])
            print(f"节点{init_node}激活的节点为{layers}")
    plt.figure()
    plt.subplot(2, 2, 1)
    nx.draw(G, with_labels=True)
    plt.title("origin")
    plt.show()