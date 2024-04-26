
from src.CamilleDataset import CamillePretrainDataset
# import pytorch dataloader:
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm
import warnings
from typing import Optional
import numpy as np

from src.Model import BYOL
import torch.nn.functional as F
import torch.nn as nn
import torch.multiprocessing as mp
from src.Collate import CollateFn
from torchvision import transforms
from src.Transforms import AddNoise, LocalMinMaxNorm
import socket

@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)

def deactivate_requires_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False

def cosine_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    period: Optional[int] = None,
) -> float:
    if step < 0:
        raise ValueError("Current step number can't be negative")
    if max_steps < 1:
        raise ValueError("Total step number must be >= 1")
    if period is None and step > max_steps:
        warnings.warn(
            f"Current step number {step} exceeds max_steps {max_steps}.",
            category=RuntimeWarning,
        )
    if period is not None and period <= 0:
        raise ValueError("Period must be >= 1")

    decay: float
    if period is not None:  # "cycle" based on period, if provided
        decay = (
            end_value
            - (end_value - start_value) * (np.cos(2 * np.pi * step / period) + 1) / 2
        )
    elif max_steps == 1:
        # Avoid division by zero
        decay = end_value
    elif step == max_steps:
        # Special case for Pytorch Lightning which updates LR scheduler also for epoch
        # after last training epoch.
        decay = end_value
    else:
        decay = (
            end_value
            - (end_value - start_value)
            * (np.cos(np.pi * step / (max_steps - 1)) + 1)
            / 2
        )
    return decay

def byol_loss(p, z):
    # Normalize the prediction and the target embedding
    p_norm = F.normalize(p, dim=1)
    z_norm = F.normalize(z, dim=1)
    # Calculate the mean squared error
    return F.mse_loss(p_norm, z_norm)

def cosine_similarity_loss(p, z):
    # Normalize the prediction and the target embedding
    p_norm = F.normalize(p, dim=1)
    z_norm = F.normalize(z, dim=1)
    # Calculate the cosine similarity
    return 1 - F.cosine_similarity(p_norm, z_norm).mean()

# Function to plot data
def plot_samples(data, title, dataset_type):
    # Number of pairs to plot
    num_pairs = 4
    fig, axes = plt.subplots(nrows=num_pairs, ncols=2, figsize=(10, 10))  # 2 columns for each pair
    fig.suptitle(title, fontsize=16)

    for i in range(num_pairs):
        if i >= len(data):  # In case there are fewer than 4 pairs in the data
            break
        # Unpack the pair (two augmented versions)
        img_pair = data[i]  # This expects each item in 'data' to be a tuple of two images
        for j in range(2):  # Two images per pair
            ax = axes[i, j]
            img = img_pair[j].to('cpu')
            if img.shape[0] == 3:  # RGB images
                ax.imshow(img.permute(1, 2, 0))
            else:  # Single channel images
                ax.imshow(img.squeeze(), cmap='gray')
            ax.axis('off')
    
    # Save the plot to a file
    plt.tight_layout()
    #plt.savefig(f"/staff/tord/Workspace/strasbourg/output/transformed_data_{dataset_type}.png")
    plt.savefig(f"output/transformed_data_{dataset_type}.png")
    
    plt.close()
    
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # Example usage:
    fs = 400  # Original sampling rate
    
    transformations = transforms.Compose([AddNoise(0.5), LocalMinMaxNorm()]) # TODO Find good noise parameter


    dataset_types = ["train", "val", "test"]
    datasets = {}
    for dataset_type in dataset_types:
        datasets[dataset_type] = CamillePretrainDataset(
            data_path="./data/" if socket.gethostname() == "saturn.norsar.no" else "/tf/data",
            dataset_type=dataset_type,
            eb_transform=transformations,
            psd_transform=transformations,
            spec_transform=transformations,
            stalta_transform=transformations,
            
        )
        print(f"Number of {dataset_type} samples: {len(datasets[dataset_type])}")
        
    collate_fn = CollateFn(image_size=(256, 256))
    # creating dataloaders:

    dataloaders = {}
    for dataset_type in dataset_types:
        dataloaders[dataset_type] = DataLoader(datasets[dataset_type],
                                            batch_size=64,
                                            shuffle=True if dataset_type == "train" else False,
                                            num_workers=0,
                                            drop_last = True,
                                            #persistent_workers=True,
                                            collate_fn=collate_fn
                                            )   
        print(f"Number of {dataset_type} batches: {len(dataloaders[dataset_type])}")
        
        



    #Plot some samples from each DataLoader
    for dataset_type, loader in dataloaders.items():
        for batch in loader:
            # Assume batch contains a list of tuples (pair of images)
            plot_samples(list(zip(batch[0][:4], batch[1][:4])), f'First four image pairs from {dataset_type} dataset', dataset_type)
            break  # Only plot the first batch


    epochs = 50
    num_ftrs = 512
    proj_hidden_dim = pred_hidden_dim = 1024
    out_dim = 512
    resnet = torchvision.models.resnet18(pretrained=True).float()
    #resnet = torchvision.models.resnet18()
    
    backbone = nn.Sequential(*list(resnet.children())[:-1]).float()
    model = BYOL(backbone=backbone, num_ftrs=num_ftrs, proj_hidden_dim=proj_hidden_dim, pred_hidden_dim=pred_hidden_dim, out_dim=out_dim)
    model.float()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    LOSS = []
    loss_function = byol_loss
    print("Starting Training")
    LOSS = []

    for epoch in range(epochs):
        total_loss = 0
        # Create a new tqdm progress bar for each epoch
        with tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for x1, x0 in pbar:
                x1 = x1.float().to(device)
                x0 = x0.float().to(device)

                optimizer.zero_grad()
                update_momentum(model.backbone, model.backbone_momentum, m=cosine_schedule(epoch, epochs, 0.996, 1))
                update_momentum(model.projection_head, model.projection_head_momentum, m=cosine_schedule(epoch, epochs, 0.996, 1))

                p0 = model(x0)
                z0 = model.forward_momentum(x0)
                p1 = model(x1)
                z1 = model.forward_momentum(x1)

                loss = 0.5 * (loss_function(p0, z1) + loss_function(p1, z0))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                # Update the progress bar description with current average loss
                pbar.set_description(f"Epoch {epoch+1}/{epochs}")
                pbar.set_postfix(loss=f"{loss.item():.5f}", avg_loss=f"{(total_loss / (pbar.n + 1)):.5f}")

        avg_loss = total_loss / len(dataloaders["train"])
        LOSS.append(f"{avg_loss:.5f}")

        # Optionally save loss and model state at the end of each epoch
        np.save(f'output/BYOL/LOSS_{epoch}.npy', LOSS)
        torch.save(model.state_dict(), f'output/TrainedModel_{epoch}_state_dict.pth')

    # Print final summary
    print(f"Training completed. All epoch losses: {LOSS}")

    # List to store embeddings and filenames
    embeddings = []
    filenames = []

    # Set the model to evaluation mode (no gradient computation)
    model.eval()
    with torch.no_grad():
        for i, (x, _, fnames) in enumerate(dataloaders["val"]):
            # Move images to the gpu
            x = x.to(device)
            # Embed the images with the pre-trained backbone
            y = model.backbone(x).flatten(start_dim=1)
            # Store the embeddings and filenames
            embeddings.append(y)
            filenames = filenames + list(fnames)

    # Concatenate embeddings and convert them to a NumPy array
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()

    # Saving
    np.save(f'outputs/embedding.npy', embeddings)
    np.save(f'outputs/filenames.npy', filenames)