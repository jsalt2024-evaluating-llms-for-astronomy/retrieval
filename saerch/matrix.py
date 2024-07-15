from tqdm import tqdm
import torch
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, TensorDataset
from topk_sae import FastAutoencoder  # Assuming train.py contains your FastAutoencoder class
import matplotlib.pyplot as plt
import clamp
import networkx as nx

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def co_occurrence(topk_indices, k = 64, ndir=9216):
    co_occurrence = np.zeros((ndir, ndir))
    norms = np.zeros(ndir)
    for i in tqdm(range(topk_indices.shape[0]), desc="Calculating co-occurrence"):
        for j in range(k):
            norms[topk_indices[i, j]] += 1
            for l in range(j + 1, k):
                co_occurrence[topk_indices[i, j], topk_indices[i, l]] += 1
                co_occurrence[topk_indices[i, l], topk_indices[i, j]] += 1

    return co_occurrence, norms

def kill_symmetry(mat, node_vals):
    for i in range(len(mat.shape[0])):
        for j in range(len(mat.shape[1])):
            if node_vals[i] >= node_vals[j]:
                mat[i, j] = 0
    return mat

def node_neighbors(G, node, direction):
    if direction == "up":
        in_edges = list(G.in_edges(node))
        nodes = [edge[0] for edge in in_edges]
    else:
        out_edges = list(G.out_edges(node))
        nodes = [edge[1] for edge in out_edges]
    
    return nodes

def make_graph(mat, clean_results, filename = 'graph.graphml'):
    G = nx.from_numpy_array(mat, create_using=nx.Graph())
    #auto_labels_indexed = {label['index']: label for label in auto_results}
    
    node_names = [feature['label'] for feature in clean_results]
    mapping_name = {i: node_names[i] for i in range(len(node_names))}
    node_density = [np.log10(feature['density']) for feature in clean_results]
    mapping_density = {i: node_density[i] for i in range(len(node_density))}

    nx.set_node_attributes(G, mapping_density, 'density')
    G = nx.relabel_nodes(G, mapping_name, 'label')

    nx.write_graphml(G, filename)

    return G

def make_MST(mat, clean_results, filename = 'mst.graphml', algorithm = 'kruskal', add_noise = False):
    G = nx.from_numpy_array(mat, create_using=nx.Graph())

    node_names = [feature for feature in clean_results]
    mapping_name = {i: node_names[i] for i in range(len(node_names))}
    node_density = [np.log10(feature['density']) for feature in clean_results.values()]
    mapping_density = {i: node_density[i] for i in range(len(node_density))}
    nx.set_node_attributes(G, mapping_density, 'density')
    
    G_tree = nx.maximum_spanning_tree(G, algorithm = algorithm)
    G_tree_directed = nx.DiGraph()
    
    for node in G_tree.nodes(data=True):
        G_tree_directed.add_node(node[0], **node[1])

    for u, v, data in G_tree.edges(data=True):
        if node_density[u] < node_density[v]:
            G_tree_directed.add_edge(u, v, **data)
        else:
            G_tree_directed.add_edge(v, u, **data)

    G_tree_directed = nx.relabel_nodes(G_tree_directed, mapping_name, 'label')

    nx.write_graphml(G_tree_directed, filename)

    return G_tree, G_tree_directed

def get_norm_cooc(unnorm_mat, norm, clean_labels, direction = "vertical", threshold = 0.1, poisson = False, exp = True):
    if poisson:
        unnorm_mat += np.random.normal(0, np.sqrt(unnorm_mat))
    
    if direction == "vertical":
        norm_mat = unnorm_mat/(norm + 1)[:, None]
    elif direction == "horizontal":
        norm_mat = unnorm_mat/(norm + 1)[None, :]
    
    clean_ids = [label['index'] for label in clean_labels.values()]
    clean_mat = norm_mat[clean_ids][:, clean_ids]

    clean_mat[clean_mat < threshold] = 0
    if exp:
        clean_mat[clean_mat > 0] = np.exp(clean_mat[clean_mat > 0])

    return clean_mat


def main():
    # hypers
    d_model = 1536
    n_dirs = d_model * 6
    k = 64
    auxk = 128
    multik = 256
    batch_size = 1024

    # Load the pre-trained model
    ae = FastAutoencoder(n_dirs, d_model, k, auxk, multik = multik).to(device)
    model_path = 'checkpoints/64_9216_128_auxk_epoch_50.pth'
    ae.load_state_dict(torch.load(model_path))
    ae.eval()

    # Load abstract embeddings
    abstract_embeddings = np.load("../data/vector_store/abstract_embeddings.npy")
    abstract_embeddings = abstract_embeddings.astype(np.float32)

    topk_indices = np.load("sae_data/topk_indices.npy")
    topk_values = np.load("sae_data/topk_values.npy")
    
    mat, norms = co_occurrence(topk_indices)
    # mat_vert = mat/(norms + 1)[:, None]
    # mat_horz = mat/(norms + 1)[None, :]
    # mat_herm = (mat_vert + mat_horz)/2

    np.save("unnorm_cooccurrence.npy", mat)
    np.save('occurrence_norms.npy', norms)

if __name__ == "__main__":
    main()
