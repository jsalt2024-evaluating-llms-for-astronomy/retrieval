from tqdm import tqdm
import torch
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, TensorDataset
from topk_sae import FastAutoencoder  # Assuming train.py contains your FastAutoencoder class
import matplotlib.pyplot as plt
import sequencer 

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
    
    # norms_matrix = np.outer(norms + 1, norms + 1)
    # co_occurrence = np.divide(co_occurrence, norms_matrix, where=norms_matrix!=0)

    return co_occurrence, norms


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
    mat_vert = mat/(norms + 1)[:, None]
    mat_horz = mat/(norms + 1)[None, :]
    mat_herm = (mat_vert + mat_horz)/2

    np.save("co_occurrence.npy", mat_herm)

    estimator_list = ['EMD', 'energy', 'KL', 'L2']
    scale_list = [[100], [100], [100], [100]]
    obj_list = np.arange(mat_herm.shape[0])
    seq = sequencer.Sequencer(obj_list, mat_herm, estimator_list, scale_list)

    output_path = "./sequencer"
    final_elongation, final_sequence = seq.execute(output_path, 
                                                to_average_N_best_estimators=True, 
                                                number_of_best_estimators=3)
    
    np.savetxt("final_sequence.txt", final_sequence, fmt='%d')

if __name__ == "__main__":
    main()