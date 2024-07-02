import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

steps = 3
num_agents = 8

def interact(G):
    for node in G.nodes:
        # 找到度最大的节点
        degrees = dict(G.degree())
        max_degree_node = max(degrees, key=degrees.get)
        if max_degree_node not in G.neighbors(node) and max_degree_node != node:
            G.add_edge(max_degree_node, node)
            # 找到度最小的邻居
            neighbors = list(G.neighbors(node))
            neighbor_degrees = {node: G.degree(node) for node in neighbors}
            min_degree_neighbor = min(neighbor_degrees, key=neighbor_degrees.get)
            G.remove_edge(min_degree_neighbor, node)
if __name__ == '__main__':
    # 生成一个随机无向图
    G = nx.Graph()
    for i in range(num_agents):
        G.add_node(i)

    # 随机生成初始连接
    initial_edges = np.random.choice(num_agents, (num_agents, 2), replace=True)
    for edge in initial_edges:
        if edge[0] != edge[1]:
            G.add_edge(edge[0], edge[1])

    plt.figure()
    plt.subplot(2, 2, 1)
    nx.draw(G, with_labels=True)
    plt.title("origin")
    for i in range(steps):
        interact(G)
        plt.subplot(2, 2, i + 2)
        nx.draw(G, with_labels=True)
        plt.title(f'Step {i + 1}')
    plt.show()