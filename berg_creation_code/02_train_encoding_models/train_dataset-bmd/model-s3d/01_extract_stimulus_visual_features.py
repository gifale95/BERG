"""Extract the BMD stimulus video features using video DNNs, downsample
them to 100 principal components using PCA, and save the PCA weights. 

The stimulus videos can be downloaded at this link:
https://boldmomentsdataset.csail.mit.edu/stimuli_metadata

The video feature are extracted using the S3D model, which is a 3D
convolutional neural network trained on the Kinetics400 video dataset:
https://docs.pytorch.org/vision/main/models/video_s3d.html

Parameters
----------
model_name : str
    Name of video DNN model used to extract the video features.
    Available options are 'mc3_18', 'r3d_18', 'r2plus1d_18', 's3d',
    'mvit_v2_s', 'mvit_v1_b', 'x3d_xs', 'slow_r50'.
n_components : int
    Number of model PCA components retained.
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG
bmd_dir : str
    Directory of the BOLD Moments Dataset (BMD).
    https://openneuro.org/datasets/ds005165

"""

import os
import argparse
import numpy as np
from enum import Enum
import torch
import torchvision
from torchvision.io.video import read_video
from torch.utils.data import Dataset, DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='s3d', type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--n_components', default=100, type=int)
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str) # !!!
parser.add_argument('--bmd_dir', default='/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset', type=str) # !!!
args, unknown = parser.parse_known_args()

print('>>> Extract video features <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# GPU and workers
# =============================================================================
# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Load the video DNN
# =============================================================================
model = torchvision.models.video.s3d(weights='KINETICS400_V1')


# =============================================================================
# Define the layers from which to extract features
# =============================================================================
# For each model, we specify the layers from which to save the features. This
# layer selection is needed because the networks are too large to save all the
# features of all layers. The saved layers are spread across early, middle and
# late layers of the networks.

class Nodes_for_FE(Enum):
    # Define the layers from which to extract features
    VideoResNet_mc3_18 = {
        'stem.2': 'stem',
        'layer1.1.relu': 'layer1',
        'layer2.1.relu': 'layer2',
        'layer3.1.relu': 'layer3',
        'layer4.1.relu': 'layer4',
        'avgpool': 'avgpool'
        }
    VideoResNet_r3d_18 = {
        'stem.2': 'stem',
        'layer1.1.relu': 'layer1',
        'layer2.1.relu': 'layer2',
        'layer3.1.relu': 'layer3',
        'layer4.1.relu': 'layer4',
        'avgpool': 'avgpool'
        }
    VideoResNet_r2plus1d_18 = {
        'stem.5': 'stem',
        'layer1.1.relu': 'layer1',
        'layer2.1.relu': 'layer2',
        'layer3.1.relu': 'layer3',
        'layer4.1.relu': 'layer4',
        'avgpool': 'avgpool'
    }
    s3d = {
        'features.2': 'layer2',
        'features.5.cat': 'layer5',
        'features.7': 'layer7',
        'features.9.cat': 'layer9',
        'features.11.cat': 'layer11',
        'features.13': 'layer13',
        'avgpool': 'avgpool'
        }
    MViT_v2_s = {
        'pos_encoding.cat' : 'stem',
        'blocks.2.mlp': 'early_block',
        'blocks.9.mlp': 'middle_block',
        'blocks.15.mlp': 'late_block'
    }
    MViT_v1_b = {
        'pos_encoding.cat' : 'stem',
        'blocks.2.mlp': 'early_block',
        'blocks.9.mlp': 'middle_block',
        'blocks.15.mlp': 'late_block'
    }
    x3d_xs = {
        'blocks.0.activation': 'stem',
        'blocks.1.res_blocks.2.activation': 'layer1',
        'blocks.2.res_blocks.4.activation': 'layer2',
        'blocks.3.res_blocks.10.activation': 'layer3',
        'blocks.4.res_blocks.6.activation': 'layer4',
        'blocks.5.proj': 'layer5'
    }
    ResNet_3D = {
        'blocks.0.pool': 'stem',
        'blocks.2.res_blocks.3.activation': 'layer2',
        'blocks.4.res_blocks.2.activation': 'layer4',
        'blocks.5.proj': 'projection'
    }


# =============================================================================
# Select the video frame number
# =============================================================================
# Each model has a specific number of frames that need to be selected at once
# from the input video.

class Num_frames(Enum):
    VideoResNet_mc3_18 = 8
    VideoResNet_r3d_18 = 8
    VideoResNet_r2plus1d_18 = 8
    s3d = 14
    MViT_v2_s = 16
    MViT_v1_b = 16
    x3d_xs = 4
    ResNet_3D = 8


# =============================================================================
# Video preprocessing
# =============================================================================
transform = torchvision.models.video.S3D_Weights.KINETICS400_V1.transforms()


# =============================================================================
# Video dataset class
# =============================================================================
class VideoDataset(Dataset):
    def __init__(self, video_dir, num_samples, device, transform=None):
        self.video_dir = video_dir
        self.video_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4'))])
        #self.video_files = videos_first_N
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def sample_frames(self, video_frames, num_samples):
        num_frames = video_frames.shape[0]
        if num_samples > num_frames:
            raise ValueError("The number of samples requested is greater than the number of frames in the video.")
        indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
        sampled_frames = video_frames[indices]
        return sampled_frames

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_frames, _, _ = read_video(video_path, pts_unit='sec',
            output_format='TCHW')
        try:
            sampled_frames = self.sample_frames(video_frames, self.num_samples)
        except ValueError:
            last_frame = video_frames[-1].unsqueeze(0).repeat(
                self.num_samples - video_frames.shape[0], 1, 1, 1)
            sampled_frames = torch.cat([video_frames, last_frame], dim=0)
        if self.transform:
            sampled_frames = self.transform(sampled_frames)
            sampled_frames = sampled_frames.to(device)
        return idx, sampled_frames


# =============================================================================
# Model specific parameters
# =============================================================================
# Define the video frame sample number
if args.model_name == 'mc3_18':
    num_samples = Num_frames.VideoResNet_mc3_18.value
    fe_nodes = Nodes_for_FE.VideoResNet_mc3_18.value
elif args.model_name == 'r3d_18':
    num_samples = Num_frames.VideoResNet_r3d_18.value
    fe_nodes = Nodes_for_FE.VideoResNet_r3d_18.value
elif args.model_name == 'r2plus1d_18':
    num_samples = Num_frames.VideoResNet_r2plus1d_18.value
    fe_nodes = Nodes_for_FE.VideoResNet_r2plus1d_18.value
elif args.model_name == 's3d':
    num_samples = Num_frames.s3d.value
    fe_nodes = Nodes_for_FE.s3d.value
elif args.model_name == 'mvit_v2_s':
    num_samples = Num_frames.MViT_v2_s.value
    fe_nodes = Nodes_for_FE.MViT_v2_s.value
elif args.model_name == 'mvit_v1_b':
    num_samples = Num_frames.MViT_v1_b.value
    fe_nodes = Nodes_for_FE.MViT_v1_b.value
elif args.model_name == 'x3d_xs':
    num_samples = Num_frames.x3d_xs.value
    fe_nodes = Nodes_for_FE.x3d_xs.value
elif args.model_name == 'slow_r50':
    num_samples = Num_frames.ResNet_3D.value
    fe_nodes = Nodes_for_FE.ResNet_3D.value


# =============================================================================
# Create the feature extractor
# =============================================================================
feature_extractor = create_feature_extractor(model, fe_nodes)

# Set the model in evaluation mode, on the current device
feature_extractor.eval()
feature_extractor.to(device)


# =============================================================================
# Extract the train video features
# =============================================================================
video_dir = os.path.join(args.bmd_dir, 'stimulus_set', 'stimuli', 'train')

# Create the dataset
dataset = VideoDataset(video_dir=video_dir, num_samples=num_samples,
    device=device, transform=transform)

# Create a DataLoader without shuffling
dataloader = None
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
    pin_memory=False) # num_workers=num_workers

# Extract the video features
features_train = []
with torch.no_grad():
    for indices, batch in dataloader:
        # Extract the features
        ft = feature_extractor(batch)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        features_train.append(np.squeeze(ft.detach().cpu().numpy()))
        print('batch {} - {} / 1101 done'.format(indices[0].item(), indices[-1].item()))

features_train = np.concatenate(features_train)


# =============================================================================
# Extract the test video features
# =============================================================================
video_dir = os.path.join(args.bmd_dir, 'stimulus_set', 'stimuli', 'test')

# Create the dataset
dataset = VideoDataset(video_dir=video_dir, num_samples=num_samples,
    device=device, transform=transform)

# Create a DataLoader without shuffling
dataloader = None
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
    pin_memory=False) # num_workers=num_workers

# Extract the video features
features_test = []
with torch.no_grad():
    for indices, batch in dataloader:
        # Extract the features
        ft = feature_extractor(batch)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        features_test.append(np.squeeze(ft.detach().cpu().numpy()))
        print('batch {} - {} / 1101 done'.format(indices[0].item(), indices[-1].item()))

features_test = np.concatenate(features_test)


# =============================================================================
# Apply PCA
# =============================================================================
# Z-score the image features
scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

# Downsample the features with PCA
if features_train.shape[1] < args.n_components:
    n_components = features_train.shape[1]
else:
    n_components = args.n_components
pca = TruncatedSVD(n_components=n_components, random_state=20200220)
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)


# =============================================================================
# Save the z-score and PCA weights
# =============================================================================
pca_weights = {
    'scaler_param': {
        'scale_': scaler.scale_,
        'mean_': scaler.mean_,
        'var_': scaler.var_,
        'n_features_in_': scaler.n_features_in_,
        'n_samples_seen_': scaler.n_samples_seen_
        },
    'pca_param': {
        'components_': pca.components_,
        'explained_variance_': pca.explained_variance_,
        'explained_variance_ratio_': pca.explained_variance_ratio_,
        'singular_values_': pca.singular_values_,
        'n_features_in_': pca.n_features_in_
        }
    }

save_dir = os.path.join(args.berg_dir, 'encoding_models', 'modality-fmri',
    'train_dataset-bmd', 'model-'+args.model_name, 'encoding_models_weights')
os.makedirs(save_dir, exist_ok=True)

file_name = 'pca_weights.npy'

np.save(os.path.join(save_dir, file_name), pca_weights)


# =============================================================================
# Save the PCA-transformed model features
# =============================================================================
save_dir = os.path.join(args.berg_dir, 'results', 'stimulus_features',
    'modality-fmri', 'train_dataset-bmd', 'model-'+args.model_name)
os.makedirs(save_dir, exist_ok=True)

file_name_train = 'pca_stimulus_features_train.npy'
file_name_test = 'pca_stimulus_features_test.npy'

np.save(os.path.join(save_dir, file_name_train), features_train)
np.save(os.path.join(save_dir, file_name_test), features_test)