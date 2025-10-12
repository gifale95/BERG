import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import h5py
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

from berg.models.fmri.huze.model import BrainEncodingModel
from berg.models.fmri.huze.config_utils import load_from_yaml


class THINGSfMRIDataset(Dataset):
    def __init__(self, fmri_data, stimuli_names, concepts, images_base_path):
        self.fmri_data = torch.FloatTensor(fmri_data)
        self.stimuli_names = stimuli_names
        self.concepts = concepts
        self.images_base_path = Path(images_base_path)
        
        # ImageNet normalization for DinoV2
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        # Load image
        stim_name = self.stimuli_names[idx]
        concept_name = self.concepts[idx]
        concept = concept_name
        image_path = self.images_base_path / concept / stim_name
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Get fMRI response
        fmri = self.fmri_data[idx]
        
        return image, fmri


def compute_correlation(predictions, targets):
    """Compute mean Pearson correlation across voxels"""
    correlations = []
    for v in range(predictions.shape[1]):
        pred_v = predictions[:, v]
        targ_v = targets[:, v]
        if pred_v.std() > 0 and targ_v.std() > 0:
            corr, _ = pearsonr(pred_v, targ_v)
            correlations.append(corr)
    return np.mean(correlations)


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, fmri in test_loader:
            images = images.to(device)
            predictions = model(images)
            all_predictions.append(predictions.cpu())
            all_targets.append(fmri)
    
    predictions = torch.cat(all_predictions, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    correlation = compute_correlation(predictions, targets)
    return correlation


def train_model(subject_id, data_dir, images_path, config_path, save_path, 
                num_epochs=100, batch_size=16, lr=3e-4):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Trainining on:", device)
    
    # Load data
    print(f"Loading data for {subject_id}...")
    with h5py.File(f"{data_dir}/fmri_{subject_id}_split-train_normalized.h5", 'r') as f:
        fmri_train = f['neural_data'][:]
    
    with h5py.File(f"{data_dir}/fmri_{subject_id}_split-test_averaged_normalized.h5", 'r') as f:
        fmri_test = f['neural_data'][:]
    
    metadata = np.load(f"{data_dir}/fmri_{subject_id}_metadata.npz", allow_pickle=True)
    train_stimuli = metadata['train_stimuli']
    train_concepts = metadata['train_concepts']
    test_stimuli = metadata['test_avg_stimuli']
    test_concepts = metadata['test_concepts']
    
    voxel_coords = torch.FloatTensor(metadata['voxel_coords'])
    n_voxels = len(voxel_coords)
    
    print(f"Training samples: {len(fmri_train)}, Test samples: {len(fmri_test)}, Voxels: {n_voxels}")
    
    # Create datasets
    train_dataset = THINGSfMRIDataset(fmri_train, train_stimuli,train_concepts, images_path)
    test_dataset = THINGSfMRIDataset(fmri_test, test_stimuli, test_concepts, images_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    print("Creating model...")
    cfg = load_from_yaml(config_path)
    model = BrainEncodingModel(cfg, n_voxel_dict={subject_id: n_voxels})
    model.coords = nn.Parameter(voxel_coords, requires_grad=False)
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    criterion = nn.SmoothL1Loss(beta=0.01)
    
    # Training loop
    best_correlation = -1
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for images, fmri in train_loader:
            images = images.to(device)
            fmri = fmri.to(device)
            
            optimizer.zero_grad()
            predictions = model(images, chunk_size=8192)
            loss = criterion(predictions, fmri)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate on test set
        test_corr = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_train_loss:.4f}, Test Corr: {test_corr:.4f}")
        
        # Save best model
        if test_corr > best_correlation:
            best_correlation = test_corr
            torch.save({
                'model_state_dict': model.state_dict(),
                'voxel_coords': voxel_coords,
                'subject_id': subject_id,
                'correlation': best_correlation,
                'epoch': epoch
            }, f"{save_path}/model_{subject_id}_best.pth")
            print(f"  New best model saved (corr: {best_correlation:.4f})")
    
    print(f"Training complete. Best correlation: {best_correlation:.4f}")


if __name__ == "__main__":
    train_model(
        subject_id='sub-01',
        data_dir='/Volumes/Extreme SSD/brain-encoding-response-generator/model_training_datasets/train_dataset-things_fmri/',
        images_path='/Volumes/Extreme SSD/Datasets/THINGS/things_images',
        config_path='/Users/domenicbersch/Documents/Repositories/NEST/berg_creation_code/02_train_encoding_models/train_dataset-things_fmri/config.yaml',
        save_path='/Users/domenicbersch/Documents/Repositories/NEST/berg_creation_code/02_train_encoding_models/train_dataset-things_fmri/saved_models',
        num_epochs=100,
        batch_size=16,
        lr=3e-4
    )