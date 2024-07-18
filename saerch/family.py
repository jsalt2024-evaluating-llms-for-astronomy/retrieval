import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
import matrix
import clamp
import pandas as pd
from topk_sae import FastAutoencoder, loss_fn, unit_norm_decoder_grad_adjustment_, unit_norm_decoder_, init_from_data_
from autointerp import NeuronAnalyzer
import networkx as nx
import os

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1024
CONFIG = '../config.yaml'
neuron = NeuronAnalyzer(CONFIG, 1, 10) # default settings

def nn_hierarchy(ae_large, ae_small, save_path = None): # nearest neighbors hierarchy
    feats = np.arange(ae_small.n_dirs)
    nns = {int(feat): [] for feat in feats}

    # nearest neighbors of features (decoder weights)
    sims = np.dot(ae_small.decoder.weight.data.cpu().numpy().T, ae_large.decoder.weight.data.cpu().numpy()) # 3072 x 9216
    matches = np.argmax(sims, axis = 0)
    
    #print(matches.shape, np.max(matches), np.min(matches))

    for i, match in enumerate(matches):
        if match in nns.keys():
            nns[int(match)].append(i)
    
    if save_path is not None:
        json.dump(nns, open(save_path, 'w'))

    return nns

def get_cosine_sim(weight1, weight2, targets1, targets2):
    A = weight1[targets1] # n1 x 1536
    B = weight2[targets2] # 1536 x n2

    return np.dot(A, B.T)


class multi_model():
    def __init__(self, model_list, all_onehots, all_acts):
        self.model_list = model_list
        self.index, self.feature_index = self.generate_index() # matrix index to feature

        self.norms = np.hstack(tuple([model.norms[list(model.clean_labels_by_id.keys())] for model in model_list])) # for one-hot co-occurrence
        self.feature_vectors = np.vstack(tuple([model.feature_vectors[list(model.clean_labels_by_id.keys())] for model in model_list]))

        if os.path.exists(all_acts) and os.path.exists(all_onehots):
            self.actsims = np.load(all_acts)
            self.mat = np.load(all_onehots)
        
        else: # acts are already normalized by feature (dim = 0)
            acts = torch.concat(tuple([model.acts[:, list(model.clean_labels_by_id.keys())] for model in model_list]), dim = 1)
            onehots = torch.concat(tuple([model.onehots[:, list(model.clean_labels_by_id.keys())] for model in model_list]), dim = 1)

            self.actsims, norms = matrix.co_occurrence(acts)
            np.save(all_acts, self.actsims)
            
            self.mat, norms = matrix.co_occurrence(onehots)
            np.save(all_onehots, self.mat)
            print('Generated and saved matrices.')
    
    def generate_index(self):
        ids = []
        index = []
        feature_index = {}

        tally = 0
        for i, model in enumerate(self.model_list):
            model_ids = model.clean_labels_by_id.keys()
            ids += list()

            for j, id in enumerate(model_ids):
                index.append(model.clean_labels_by_id[id]['label'])
                feature_index[(i, id)] = tally
                tally += 1
        
        return index, feature_index
    

    def explore_splitting(self, ind, nns_list):
        targets = [ind]
        matrix_indices = [] # initial index
        feature_names = []

        for i, model in enumerate(self.model_list):
            print(i, targets, model.get_feature_names(targets))
            
            next_targets = []
            
            for target in targets:
                try:
                    matrix_indices.append(self.feature_index[(i, target)])
                    feature_names.append((i, model.clean_labels_by_id[target]['label']))
                except:
                    print('{} not in clean feature list'.format(target))
                
                if i < len(self.model_list) - 1:
                    matches = nns_list[i][target]
                    if len(matches) > 0:
                        # print(get_cosine_sim(model.feature_vectors, self.model_list[i + 1].feature_vectors, target, matches))
                        next_targets += matches
                
            targets = next_targets
        
        # 16 --> 64 directly
        if len(nns_list) == len(self.model_list):
            print('self consistency ', nns_list[-1][ind], self.model_list[-1].get_feature_names(nns_list[-1][ind]))
            print(get_cosine_sim(self.model_list[0].feature_vectors, self.model_list[-1].feature_vectors, ind, nns_list[-1][ind]))

        return matrix_indices, feature_names
        

class model():
    def __init__(self, model_path, topk_indices, topk_values, autointerp_results, 
                 mat, norms, actsims, d_model = 1536):
        self.k, self.n_dirs, auxk = model_path.split('/')[-1].split('_')[:3]
        
        self.ae = FastAutoencoder(n_dirs = int(self.n_dirs), k = int(self.k), d_model = d_model, auxk = int(auxk), multik = 0)
        self.ae.load_state_dict(torch.load(model_path))
        self.ae.eval()
        self.feature_vectors = self.ae.decoder.weight.data.cpu().numpy().T

        if os.path.exists(topk_indices) and os.path.exists(topk_values):
            self.topk_indices = np.load(topk_indices)
            self.topk_values = np.load(topk_values)
        else:
            self.topk_indices, self.topk_values = self.generate_topk(topk_indices, topk_values)

        self.auto_results = json.load(open(autointerp_results)) 
        self.clean_labels = clamp.get_clean_labels(self.auto_results)
        
        for label in self.clean_labels:
            self.clean_labels[label]['label'] = label
        
        self.clean_labels_by_id = {label['index']: label for label in self.clean_labels.values()}
        
        # one-hot co-occurrences
        if os.path.exists(mat) and os.path.exists(norms):
            self.mat = np.load(mat)
            self.norms = np.load(norms)
            self.onehots = matrix.activations(self.topk_indices, self.topk_values, ndir = self.n_dirs, mode = 'onehot')
        else:
            self.onehots, self.mat, self.norms = self.generate_matrix(mat, norms)
        
        # activation dot product
        if os.path.exists(actsims):
            self.actsims = np.load(actsims)
            self.acts = matrix.activations(self.topk_indices, self.topk_values, ndir = self.n_dirs, mode = 'value')
        else:
            self.acts, self.actsims = self.generate_activation_sims(actsims)

        self.clean_cooc = matrix.get_norm_cooc(self.mat, self.norms, self.clean_labels, threshold = 0.07, poisson = False)

    def generate_topk(self, topk_indices_path, topk_values_path):
        topk_indices = np.zeros((num_abstracts, self.k), dtype=np.int64)
        topk_values = np.zeros((num_abstracts, self.k), dtype=np.float32)

        # Process batches
        with torch.no_grad():
            for i, (batch,) in enumerate(tqdm(dataloader, desc="Processing abstracts")):
                batch = batch.to(device)
                _, info = self.ae(batch)
                
                start_idx = i * BATCH_SIZE
                end_idx = start_idx + batch.size(0)
                
                topk_indices[start_idx:end_idx] = info['topk_indices'].cpu().numpy()
                topk_values[start_idx:end_idx] = info['topk_values'].cpu().numpy()
        
        np.save(topk_indices_path, topk_indices)
        np.save(topk_values_path, topk_values)

        print("Processing complete. Results saved to {}.".format(topk_indices_path))  

        return topk_indices, topk_values

    def generate_matrix(self, mat_path, norms_path): # if it doesn't exist
        activations = matrix.activations(self.topk_indices, self.topk_values, ndir = self.n_dirs, mode = 'onehot')
        mat, norms = matrix.co_occurrence(activations)

        np.save(mat_path, mat)
        np.save(norms_path, norms)

        return activations, mat, norms
    
    def generate_activation_sims(self, actsim_path):
        activations = matrix.activations(self.topk_indices, self.topk_values, ndir = self.n_dirs, mode = 'value')
        mat, norms = matrix.co_occurrence(activations)

        np.save(actsim_path, mat)

        return activations, mat

    def get_feature_activations(self, n, feature_index = None, feature_name = None): # example abstracts
        if feature_index is None:
            if feature_name is None:
                raise ValueError("Either feature_index or feature_name must be provided")
            feature_index = self.clean_labels[feature_name]['index']

        neuron.topk_indices = self.topk_indices
        neuron.topk_values = self.topk_values
        
        return neuron.get_feature_activations(m = n, feature_index = feature_index)

    def get_families(self, n = 3): # iterative family finding in the MST
        G_tree = matrix.make_MST(self.clean_cooc, self.clean_labels)
        subtrees = matrix.subtree_iterate(self.mat, self.norms, G_tree, self.clean_labels, n = n)
        return subtrees
    
    def get_feature_names(self, indices): # get feature names from indices
        # auto results in format similar to feature_analysis_results.json
        k = len(indices)
        feature_list = [""] * k
        for result in self.auto_results:
            if result['index'] in indices:
                feature_list[list(indices).index(result['index'])] = result['label']
        
        return feature_list

abstract_embeddings = np.load("../data/vector_store/abstract_embeddings.npy")
abstract_embeddings = abstract_embeddings.astype(np.float32)
abstract_texts = json.load(open('../data/vector_store/abstract_texts.json'))['abstracts']

dataset = TensorDataset(torch.from_numpy(abstract_embeddings))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
num_abstracts = len(abstract_embeddings)