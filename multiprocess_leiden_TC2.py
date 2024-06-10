#!/usr/bin/env python
# coding: utf-8
# In[1]:
import numpy as np
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
from multiprocessing import Process, Manager
from sklearn.metrics import normalized_mutual_info_score
iter = 2
data_number = 2
report = {
    'Data number': [],
    'First leiden time': [],
    'Second leiden time': [],
}
def refine_cluster(cluster_index, cluster_vertices, data):
    n_of_valid_vertices = 0
    new_graph = []
    for index in range(len(data)):
        node1, node2 = data[index]
        # assert node1 != node2
        if node1 in cluster_vertices and node2 in cluster_vertices:
            new_graph.append([node1, node2])
    new_graph = np.array(new_graph)
    unique_nodes, count = np.unique(new_graph, return_counts=True)
    # Mapping
    old2new = {}
    new2old = {}
    for i, node in enumerate(unique_nodes):
        old2new[node] = i
        new2old[i] = node
    new_vertices = np.array([[old2new[node1], old2new[node2]] for node1, node2 in new_graph])
    if len(new_vertices) == 0:
        print(f'{cluster_index} is empty')
        print(f"Vertices: {len(cluster_vertices)}, {cluster_vertices}")
        return
    # Generate sub-graphs
    new_graph = ig.Graph(new_vertices, directed=False)
    sub_partition = la.find_partition(new_graph, la.ModularityVertexPartition, n_iterations=iter, seed=7777)
    if sub_partition.modularity < mother_modularity:
        cluster_set.append(cluster_vertices)
        return
    n_of_valid_vertices += len(cluster_vertices)
    check_assert_list = []
    for index, sub_vertices in enumerate(sub_partition):
        old_sub_vertices = []
        for new_vertex in sub_vertices:
            old_sub_vertices.append(new2old[new_vertex])
        check_assert_list.extend(old_sub_vertices)
        if len(old_sub_vertices) != 0:
            cluster_set.append(old_sub_vertices)
            print(f"Cluster {cluster_index} has {len(old_sub_vertices)} nodes")
for data_number in range(11, 16):
    division_count = 0
    mother_n_clusters = 0
    child_n_clusters = 0
    data_path = f'dataset/tc{data_number}.dat'
    data = np.loadtxt(data_path).astype(int)
    graph = ig.Graph(data, directed=False)
    start = time.time()
    partition = la.find_partition(graph, la.ModularityVertexPartition, n_iterations=iter, seed=7777,)
    end = time.time()
    # print(f"First leiden time: {end - start}")
    report['First leiden time'].append(end - start)
    vertices = graph.vcount() - 1
    pred_labels = [0] * (vertices)
    for id, nodes in enumerate(partition):
        for node in nodes:
            if node != 0:
                pred_labels[node - 1] = id
    pred_labels = np.array(pred_labels)
    original_modularity = partition.modularity
    cluster_set = []
    n_of_valid_vertices = 0
    mother_modularity = original_modularity
    manager = Manager()
    start = time.time()
    process = [Process(target=refine_cluster, args=(index, cluster, data)) for index, cluster in enumerate(partition)]
    for p in process:
        p.start()
    for p in process:
        p.join()
    end = time.time()
    report['Second leiden time'].append(end - start + report['First leiden time'][-1])
    total_nodes = 0
    for index, cluster in enumerate(cluster_set):
        total_nodes += len(cluster)
    pred_labels = [0] * (vertices)
    for id, nodes in enumerate(cluster_set):
        for node in nodes:
            if node != 0:
                pred_labels[node - 1] = id
    pred_labels = np.array(pred_labels)
    report['Data number'].append(data_number)
df = pd.DataFrame(report)
df.to_csv('refine_leiden-TC2-multi.csv', index=False)