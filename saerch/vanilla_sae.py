import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VanillaSparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, n_dirs: int):
        super().__init__()
        self.d_model = d_model
        self.n_dirs = n_dirs
        
        self.encoder = nn.Linear(d_model, n_dirs)
        self.decoder = nn.Linear(n_dirs, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        
        self.stats_last_nonzero = torch.zeros(n_dirs, dtype=torch.long, device=device)
        
    def forward(self, x):
        x = x - self.pre_bias
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z) + self.pre_bias
        
        # Update stats_last_nonzero
        self.stats_last_nonzero += 1
        self.stats_last_nonzero[z.sum(dim=0) > 0] = 0
        
        return x_hat, z

def mse(output, target):
    return torch.mean((output - target) ** 2)

def normalized_mse(recon, xs):
    return mse(recon, xs) / mse(xs.mean(dim=0, keepdim=True).expand_as(xs), xs)

def loss_fn(x, x_hat, z, l1_lambda):
    reconstruction_loss = normalized_mse(x_hat, x)
    l1_loss = l1_lambda * z.norm(p=1, dim=-1).mean() #torch.mean(torch.abs(z))
    return reconstruction_loss + l1_loss, reconstruction_loss, l1_loss

def train(sae, train_loader, optimizer, epochs, l1_lambda, clip_grad=None, save_dir="checkpoints", model_name=""):
    os.makedirs(save_dir, exist_ok=True)
    step = 0
    num_batches = len(train_loader)
    for epoch in range(epochs):
        sae.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            x = batch[0].to(device)
            x_hat, z = sae(x)
            loss, reconstruction_loss, l1_loss = loss_fn(x, x_hat, z, l1_lambda)
            loss.backward()
            step += 1
            
            # L0 norm (proportion of non-zero elements)
            l0_norm = torch.mean((z != 0).sum(dim=1).float()).item()
            
            # proportion of dead latents (not fired in last num_batches = 1 epoch)
            dead_latents_prop = (sae.stats_last_nonzero > num_batches).float().mean().item()
            
            wandb.log({
                "total_loss": loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "l1_loss": l1_loss.item(),
                "l0_norm": l0_norm,
                "dead_latents_proportion": dead_latents_prop,
                "step": step
            })
            
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), clip_grad)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Delete previous model saves for this configuration
        for old_model in glob.glob(os.path.join(save_dir, f"{model_name}_epoch_*.pth")):
            os.remove(old_model)

        # Save new model
        save_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
        torch.save(sae.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def main():
    d_model = 1536
    n_dirs = d_model * 8
    batch_size = 1024
    lr = 1e-4
    epochs = 10
    l1_lambda = 0.05
    clip_grad = 1.0

    # Create model name
    model_name = f"vanilla_{n_dirs}_{l1_lambda}"

    wandb.init(project="saerch", name=model_name, config={
        "n_dirs": n_dirs,
        "d_model": d_model,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "l1_lambda": l1_lambda,
        "clip_grad": clip_grad,
        "device": device.type
    })

    data = np.load("../data/vector_store/abstract_embeddings.npy")
    data_tensor = torch.from_numpy(data).float()
    dataset = TensorDataset(data_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sae = VanillaSparseAutoencoder(d_model, n_dirs).to(device)
    
    # Initialize pre_bias to median of data
    with torch.no_grad():
        sae.pre_bias.data = torch.median(data_tensor[:10000], dim=0)[0].to(device)

    optimizer = optim.Adam(sae.parameters(), lr=lr)

    train(sae, train_loader, optimizer, epochs, l1_lambda, clip_grad, model_name=model_name)

    wandb.finish()

if __name__ == "__main__":
    main()