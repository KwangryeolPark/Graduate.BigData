#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score

iter = 2

report = {
    'Data name': [],
    'Original NMI': [],
    'Sub NMI': [],
    'Improvement': [],
    'Percentage': [],
    'Original modularity': [],
    'Division count': [],
    'Mother clusters': [],
    'Child clusters': [],
}

for data_name in ['dolphin', 'football', 'karate', 'mexican', 'polbooks', 'railway', 'strike']:
    division_count = 0
    mother_n_clusters = 0
    child_n_clusters = 0
    
    data_path = f'dataset/{data_name}/network.dat'
    label_path = f'dataset/{data_name}/community.dat'

    data = np.loadtxt(data_path).astype(int)
    graph = ig.Graph(data, directed=False)
    partition = la.find_partition(graph, la.ModularityVertexPartition, n_iterations=iter, seed=7777,)

    vertices = graph.vcount() - 1

    answer = pd.read_csv(label_path, sep='\t', header=None)
    labels = answer[1].values

    print(f"{len(partition)} clusters found")
    print(f"Number of vertexes: {vertices}")


    # In[3]:


    pred_labels = [0] * (vertices)
    for id, nodes in enumerate(partition):
        for node in nodes:
            if node != 0:
                pred_labels[node - 1] = id

    pred_labels = np.array(pred_labels)
    original_nmi = normalized_mutual_info_score(answer[1], pred_labels)

    original_modularity = partition.modularity
    print(f"Original NMI: {original_nmi}")
    print(f"Original modularity: {original_modularity}")


    # In[ ]:


    cluster_set = []

    n_of_valid_vertices = 0
    mother_modularity = original_modularity
    for cluster_index, cluster_vertices in tqdm(enumerate(partition), total=len(partition), desc=f'Data name: {data_name}'):
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
            continue
        
        mother_n_clusters += 1
        
        # Generate sub-graphs
        new_graph = ig.Graph(new_vertices, directed=False)
        sub_partition = la.find_partition(new_graph, la.ModularityVertexPartition, n_iterations=iter, seed=7777)
        if sub_partition.modularity < mother_modularity:
            cluster_set.append(cluster_vertices)
            continue
        n_of_valid_vertices += len(cluster_vertices)

        # if len(unique_nodes) != len(cluster_vertices):
        #     print(list(unique_nodes))
        #     print(cluster_vertices)

        check_assert_list = []
        for index, sub_vertices in enumerate(sub_partition):
            old_sub_vertices = []
            for new_vertex in sub_vertices:
                old_sub_vertices.append(new2old[new_vertex])
            check_assert_list.extend(old_sub_vertices)
            if len(old_sub_vertices) != 0:
                cluster_set.append(old_sub_vertices)
                child_n_clusters += 1
        print(f"Cluster {cluster_index} has been divided into {len(sub_partition)} clusters")
        division_count += 1
        #     assert len(old_sub_vertices) != 0
        
        # check_assert = False
        # assert len(check_assert_list) == len(list(old2new.keys()))
        # for node1, node2 in zip(sorted(check_assert_list), sorted(list(old2new.keys()))):
        #     assert node1 == node2
            


    # In[ ]:


    print(f"Found clusters: {len(cluster_set)}")
    total_nodes = 0
    for index, cluster in enumerate(cluster_set):
        total_nodes += len(cluster)
    print(f"Total nodes: {total_nodes}")


    # In[ ]:


    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    counts = sorted([len(cluster) for cluster in partition])
    axes[2].bar(range(len(counts)), counts, color='b')

    counts = sorted([len(cluster) for cluster in cluster_set])
    axes[1].bar(range(len(counts)), counts, color='g')

    unique, counts = np.unique(labels, return_counts=True)
    counts = sorted(counts)
    axes[0].bar(range(len(counts)), counts, color='r')



    # In[ ]:


    pred_labels = [0] * (vertices)
    for id, nodes in enumerate(cluster_set):
        for node in nodes:
            if node != 0:
                pred_labels[node - 1] = id

    pred_labels = np.array(pred_labels)
    sub_nmi = normalized_mutual_info_score(answer[1], pred_labels)
    print(f"Data number: {data_name}")
    print(f"Ori NMI: {original_nmi}")
    print(f"Sub NMI: {sub_nmi}")

    print("Improvement: ", sub_nmi - original_nmi, "Percentage: ", (sub_nmi - original_nmi) / original_nmi * 100)
    print(f"Original modularity: {original_modularity}")
    
    report['Data name'].append(data_name)
    report['Original NMI'].append(original_nmi)
    report['Sub NMI'].append(sub_nmi)
    report['Improvement'].append(sub_nmi - original_nmi)
    report['Percentage'].append((sub_nmi - original_nmi) / original_nmi * 100)
    report['Original modularity'].append(original_modularity)
    report['Division count'].append(division_count)
    report['Mother clusters'].append(mother_n_clusters)
    report['Child clusters'].append(child_n_clusters)
    

df = pd.DataFrame(report)
df.to_csv('refine_leiden-real.csv', index=False)