import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FastAutoencoder(nn.Module):
    def __init__(self, n_dirs: int, d_model: int, k: int, auxk: int, multik: int, dead_steps_threshold: int = 266):
        super().__init__()
        self.n_dirs = n_dirs
        self.d_model = d_model
        self.k = k
        self.auxk = auxk
        self.multik = multik
        self.dead_steps_threshold = dead_steps_threshold

        self.encoder = nn.Linear(d_model, n_dirs, bias=False)
        self.decoder = nn.Linear(n_dirs, d_model, bias=False)

        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs))

        self.stats_last_nonzero = torch.zeros(n_dirs, dtype=torch.long, device=device)

    def forward(self, x, theta = 0.1):
        x = x - self.pre_bias
        latents_pre_act = self.encoder(x) + self.latent_bias

        # Main top-k selection
        topk_values, topk_indices = torch.topk(latents_pre_act, k=self.k, dim=-1)
        topk_values = F.relu(topk_values)
        multik_values, multik_indices = torch.topk(latents_pre_act, k=4*self.k, dim=-1)
        multik_values = F.relu(multik_values)

        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(-1, topk_indices, topk_values)
        multik_latents = torch.zeros_like(latents_pre_act)
        multik_latents.scatter_(-1, multik_indices, multik_values)

        # Update stats_last_nonzero
        self.stats_last_nonzero += 1
        self.stats_last_nonzero.scatter_(0, topk_indices.unique(), 0)

        recons = self.decoder(latents) + self.pre_bias
        multik_recons = self.decoder(multik_latents) + self.pre_bias

        latents_jump = latents_pre_act.detach().clone()
        latents_jump[latents_jump < theta] = 0

        # AuxK
        if self.auxk is not None:
            # Create dead latents mask
            dead_mask = (self.stats_last_nonzero > self.dead_steps_threshold).float()
            
            # Apply mask to latents_pre_act
            dead_latents_pre_act = latents_pre_act * dead_mask
            
            # Select top-k_aux from dead latents
            auxk_values, auxk_indices = torch.topk(dead_latents_pre_act, k=self.auxk, dim=-1)
            auxk_values = F.relu(auxk_values)

        else:
            auxk_values, auxk_indices = None, None

        return recons, {
            "topk_indices": topk_indices,
            "topk_values": topk_values,
            "multik_indices": multik_indices,
            "multik_values": multik_values,
            "multik_recons": multik_recons,
            "auxk_indices": auxk_indices,
            "auxk_values": auxk_values,
            "latents_pre_act": latents_pre_act,
            "latents_jump": latents_jump,
        }

    def decode_sparse(self, indices, values):
        latents = torch.zeros(self.n_dirs, device=indices.device)
        latents.scatter_(-1, indices, values)
        return self.decoder(latents) + self.pre_bias

    def decode_clamp(self, latents, clamp):
        topk_values, topk_indices = torch.topk(latents, k = 64, dim=-1)
        topk_values = F.relu(topk_values)
        latents = torch.zeros_like(latents)
        latents.scatter_(-1, topk_indices, topk_values)
        # multiply latents by clamp, which is 1D but has has the same size as each latent vector
        latents = latents * clamp
        
        return self.decoder(latents) + self.pre_bias
    
    def decode_at_k(self, latents, k):
        topk_values, topk_indices = torch.topk(latents, k=k, dim=-1)
        topk_values = F.relu(topk_values)
        latents = torch.zeros_like(latents)
        latents.scatter_(-1, topk_indices, topk_values)
        
        return self.decoder(latents) + self.pre_bias

def unit_norm_decoder_(autoencoder: FastAutoencoder) -> None:
    with torch.no_grad():
        autoencoder.decoder.weight.div_(autoencoder.decoder.weight.norm(dim=0, keepdim=True))

def unit_norm_decoder_grad_adjustment_(autoencoder: FastAutoencoder) -> None:
    if autoencoder.decoder.weight.grad is not None:
        with torch.no_grad():
            proj = torch.sum(autoencoder.decoder.weight * autoencoder.decoder.weight.grad, dim=0, keepdim=True)
            autoencoder.decoder.weight.grad.sub_(proj * autoencoder.decoder.weight)

def mse(output, target):
    return F.mse_loss(output, target)

def normalized_mse(recon, xs):
    return mse(recon, xs) / mse(xs.mean(dim=0, keepdim=True).expand_as(xs), xs)

def loss_fn(ae, x, recons, info, auxk_coef, multik_coef):
    recons_loss = normalized_mse(recons, x)
    recons_loss += multik_coef * normalized_mse(info["multik_recons"], x)
    
    if ae.auxk is not None:
        e = x - recons.detach()  # reconstruction error
        auxk_latents = torch.zeros_like(info["latents_pre_act"])
        auxk_latents.scatter_(-1, info["auxk_indices"], info["auxk_values"])
        e_hat = ae.decoder(auxk_latents)  # reconstruction of error using dead latents
        auxk_loss = normalized_mse(e_hat, e)
        total_loss = recons_loss + auxk_coef * auxk_loss
    else:
        auxk_loss = torch.tensor(0.0, device=device)
        total_loss = recons_loss
    
    return total_loss, recons_loss, auxk_loss

def init_from_data_(ae, data_sample):
    # set pre_bias to median of data
    ae.pre_bias.data = torch.median(data_sample, dim=0).values
    nn.init.xavier_uniform_(ae.decoder.weight)

    # decoder is unit norm
    unit_norm_decoder_(ae)

    # encoder as transpose of decoder
    ae.encoder.weight.data = ae.decoder.weight.t().clone()

    nn.init.zeros_(ae.latent_bias)

def train(ae, train_loader, optimizer, epochs, k, auxk_coef, multik_coef, clip_grad=None, save_dir="checkpoints", model_name=""):
    os.makedirs(save_dir, exist_ok=True)
    step = 0
    num_batches = len(train_loader)
    for epoch in range(epochs):
        ae.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            x = batch[0].to(device)
            recons, info = ae(x)
            loss, recons_loss, auxk_loss = loss_fn(ae, x, recons, info, auxk_coef, multik_coef)
            loss.backward()
            step += 1

            # calculate proportion of dead latents (not fired in last num_batches = 1 epoch)
            dead_latents_prop = (ae.stats_last_nonzero > num_batches).float().mean().item()

            wandb.log({
                "total_loss": loss.item(),
                "reconstruction_loss": recons_loss.item(),
                "auxiliary_loss": auxk_loss.item(),
                "dead_latents_proportion": dead_latents_prop,
                "l0_norm": k,
                "step": step
            })
            
            unit_norm_decoder_grad_adjustment_(ae)
            
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(ae.parameters(), clip_grad)
            
            optimizer.step()
            unit_norm_decoder_(ae)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Delete previous model saves for this configuration
        for old_model in glob.glob(os.path.join(save_dir, f"{model_name}_epoch_*.pth")):
            os.remove(old_model)

        # Save new model
        save_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
        torch.save(ae.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def main():
    d_model = 1536
    n_dirs = d_model * 6
    k = 64
    auxk = 128 #256
    multik = 256
    batch_size = 1024
    lr = 1e-4
    epochs = 10
    auxk_coef = 1/32
    clip_grad = 1.0
    multik_coef = 0 # turn it off

    # Create model name
    model_name = f"{k}_{n_dirs}_{auxk}_{multik}_multik"

    wandb.init(project="saerch", name=model_name, config={
        "n_dirs": n_dirs,
        "d_model": d_model,
        "k": k,
        "auxk": auxk,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "auxk_coef": auxk_coef,
        "multik_coef": multik_coef,
        "clip_grad": clip_grad,
        "device": device.type
    })

    data = np.load("../data/vector_store/abstract_embeddings.npy")
    data_tensor = torch.from_numpy(data).float()
    dataset = TensorDataset(data_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ae = FastAutoencoder(n_dirs, d_model, k, auxk, multik).to(device)
    init_from_data_(ae, data_tensor[:10000].to(device))

    optimizer = optim.Adam(ae.parameters(), lr=lr)

    train(ae, train_loader, optimizer, epochs, k, auxk_coef, multik_coef, clip_grad=clip_grad, model_name=model_name)

    wandb.finish()

if __name__ == "__main__":
    main()
