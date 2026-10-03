"""Compute per-image visual complexity with two published CNN models.

ICNet
   Feng et al. (2023), IEEE TPAMI 45(7):8577-8593. Two-branch ResNet18 with
   spatial layout attention, trained on the IC9600 human ratings. Outputs a
   sigmoid score in [0, 1] plus a 64x64 complexity map.
Nagle & Lavie CNN
   Nagle & Lavie (2020), R Soc Open Sci 7:191487. ImageNet-pretrained CNN
   fine-tuned on 75,020 2AFC complexity comparisons over PASCAL VOC. Outputs a
   TrueSkill-derived observer-rating score, roughly on a [0, 50] scale.

Both models are used exactly as released by their authors; this script only
reproduces their published preprocessing and returns one scalar per image.
Weights are not redistributed here and must be fetched separately:
   ICNet   git clone https://github.com/tinglyfeng/IC9600
           + download ck.pth from the Google Drive link in that README
   N&L     git clone https://github.com/fusionlove/image-complexity
           + git lfs pull   (trained_model_inception_v3.h5, ~166 MB)

The two scores live on different scales and are only comparable after
rank-transforming or z-scoring within a stimulus set.

Parameters
----------
images : str
   Path to a .npy file holding the image batch, shape (B, H, W, C), RGB.
input_range : str
   Intensity range of the input array: 'auto', 'uint8' ([0, 255] integer),
   'unit' (float [0, 1]) or 'float255' (float [0, 255]). 'auto' guesses from
   the array max, which is unreliable for very dark float images.
methods : str
   Which models to run: 'icnet' or 'nagle_lavie'.
icnet_repo : str
   Path to a local clone of github.com/tinglyfeng/IC9600 (provides ICNet.py).
icnet_checkpoint : str
   Path to the ICNet checkpoint ck.pth.
nagle_lavie_model : str
   Path to trained_model_inception_v3.h5.
batch_size : int
   Images per forward pass.
berg_dir : str
    Directory of the BERG.
imagenet_dir : str
    Directory of the ImageNet image set.
    https://www.image-net.org/challenges/LSVRC/2012/index.php

"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as trn
from scipy.stats import ttest_ind
import time


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--roi', default='V1', type=str)
parser.add_argument('--time_window_pair', default='0.06-0.10__0.20-0.25', type=str)
parser.add_argument('--imageset', default='imagenet', type=str)
parser.add_argument('--n_images', default=25, type=int)
parser.add_argument('--input_range', type=str, default='auto')
parser.add_argument('--methods', type=str, default='nagle_lavie')
parser.add_argument('--icnet_repo', type=str, default='/home/giffordale95/projects/brain-encoding-response-generator/github/BERG/paper_analyses/04-eeg_fmri_fusion/within_area_dynamics/02_rnc/IC9600')
parser.add_argument('--icnet_checkpoint', type=str, default='/scratch/giffordale95/projects/brain-encoding-response-generator/eeg_fmri_fusion/within_area_dynamics/rnc/IC9600/checkpoint/ck.pth')
parser.add_argument('--nagle_lavie_model', type=str, default='/scratch/giffordale95/projects/brain-encoding-response-generator/eeg_fmri_fusion/within_area_dynamics/rnc/image-complexity/trained_model_inception_v3.h5')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--channels_last', type=str, default='auto')
parser.add_argument('--berg_dir', default='/scratch/giffordale95/projects/brain-encoding-response-generator', type=str)
parser.add_argument('--imagenet_dir', default='/scratch/ccn_datasets/ILSVRC2012', type=str)
args, _ = parser.parse_known_args()

print('>>> Image complexity: ICNet (IC9600) and Nagle & Lavie CNN <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
   print('{:20} {}'.format(key, val))

seed = 20200220
np.random.seed(seed)
torch.manual_seed(seed)

n_threads = args.n_threads if args.n_threads > 0 else (os.cpu_count() or 1)
torch.set_num_threads(n_threads)
print('\nUsing {} CPU threads'.format(n_threads))

icnet_size = 512
nagle_lavie_size = 500
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
caffe_channel_means = [103.939, 116.779, 123.68]
icnet_map_size = icnet_size // 8   # ICNet's upsize; 64 for the asserted 512 input

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Break down the time windows
# =============================================================================
time_window_1_start, time_window_1_end = map(
    float, args.time_window_pair.split('__')[0].split('-'))
time_window_2_start, time_window_2_end = map(
    float, args.time_window_pair.split('__')[1].split('-'))


# =============================================================================
# Load the RNC controlling image IDs
# =============================================================================
data_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'stats', 'cv-0',
    args.time_window_pair, f'imageset-{args.imageset}',
    f'stats_roi-{args.roi}.npy')
data = np.load(data_dir, allow_pickle=True).item()

controlling_images = data['controlling_images']


# =============================================================================
# Load the images
# =============================================================================
# Define the image transform
if args.methods == 'icnet':
   resize_px = icnet_size
if args.methods == 'nagle_lavie':
   resize_px = nagle_lavie_size
transform = trn.Compose([
    trn.Lambda(lambda img: trn.functional.center_crop(img, min(img.size))),
    trn.Resize((resize_px, resize_px))
])

# Access the ILSVRC-2012 validation split
imageset = torchvision.datasets.ImageNet(root=args.imagenet_dir, split='val',
    transform=transform)

# Load the controlling images
images_all = {}
for key, val in tqdm(controlling_images.items()):
    for i, img_id in enumerate(val):
        img, _ = imageset.__getitem__(img_id)
        if i == 0:
            images_all[key] = np.expand_dims(img, 0)
        else:
            images_all[key] = np.append(images_all[key], np.expand_dims(img, 0), 0)


# =============================================================================
# Input canonicalisation
# =============================================================================
def to_uint8_rgb(images, input_range='auto'):
   """Canonicalise an image batch to uint8 RGB.
 
   Both models rescale intensities internally, so the incoming range must be
   resolved once, up front, rather than being guessed twice downstream.
 
   Parameters
   ----------
   images : ndarray
      Image batch of shape (B, H, W, C).
   input_range : str
      One of 'auto', 'uint8', 'unit', 'float255'.
 
   Returns
   -------
   images_uint8 : ndarray
      Image batch of shape (B, H, W, 3), dtype uint8.
   """
   images = np.asarray(images)
   if images.ndim != 4:
      raise ValueError('Expected (B, H, W, C), got {}'.format(images.shape))
 
   if input_range == 'auto':
      if images.dtype == np.uint8:
         input_range = 'uint8'
      else:
         # Fragile for uniformly dark float images: pass input_range explicitly
         # whenever the batch might not contain a near-saturated pixel.
         input_range = 'unit' if float(np.nanmax(images)) <= 1. else 'float255'
         print('\nAuto-detected input_range: {}'.format(input_range))
 
   if input_range == 'uint8':
      images = images.astype(np.uint8)
   elif input_range == 'unit':
      images = np.clip(images.astype(np.float32) * 255., 0, 255)
      images = images.round().astype(np.uint8)
   elif input_range == 'float255':
      images = np.clip(images.astype(np.float32), 0, 255).round().astype(np.uint8)
   else:
      raise ValueError('Unknown input_range: {}'.format(input_range))
 
   n_channels = images.shape[-1]
   if n_channels == 4:
      images = images[..., :3]
   elif n_channels == 1:
      images = np.repeat(images, 3, axis=-1)
   elif n_channels != 3:
      raise ValueError('Expected 1, 3 or 4 channels, got {}'.format(n_channels))
 
   min_side = min(images.shape[1], images.shape[2])
   if min_side < 200:
      print('\nWarning: smallest image side is {} px. Nagle & Lavie (2020) '
         'excluded images under 200 px from their training set, so scores for '
         'these images are extrapolations.'.format(min_side))
 
   return images   # Shape: (n_images, height, width, 3)


# =============================================================================
# ICNet (Feng et al., 2023)
# =============================================================================
def complexity_icnet(images, icnet_repo, checkpoint_path, device='cuda',
   batch_size=16, return_maps=False):
   """Predict IC9600-calibrated complexity scores with ICNet.

   Parameters
   ----------
   images : ndarray
      uint8 RGB batch of shape (B, H, W, 3).
   icnet_repo : str
      Local clone of github.com/tinglyfeng/IC9600, providing ICNet.py.
   checkpoint_path : str
      Path to ck.pth.
   device : str
      Torch device string.
   batch_size : int
      Images per forward pass.
   return_maps : bool
      Also return the 64x64 complexity maps.

   Returns
   -------
   scores : ndarray
      Complexity scores of shape (B,), sigmoid output in [0, 1].
   maps : ndarray or None
      Complexity maps of shape (B, 64, 64), or None if return_maps is False.
   """
   if icnet_repo not in sys.path:
      sys.path.insert(0, icnet_repo)
   from ICNet import ICNet

   device = torch.device(device)
   # is_pretrain=False skips the ImageNet ResNet18 download; ck.pth overwrites
   # those weights anyway.
   model = ICNet(is_pretrain=False)
   model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
   model.eval().to(device)

   mean = torch.tensor(imagenet_mean, device=device).view(1, 3, 1, 1)
   std = torch.tensor(imagenet_std, device=device).view(1, 3, 1, 1)

   n_images = len(images)
   scores = np.zeros(n_images, dtype=np.float32)   # Shape: (n_images)
   maps = np.zeros((n_images, icnet_map_size, icnet_map_size),
      dtype=np.float32) if return_maps else None

   for start in tqdm(range(0, n_images, batch_size), desc='ICNet'):
      batch = images[start:start+batch_size]
      stop = start + len(batch)
      x = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
      x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.)
      # ICNet asserts a square 512x512 input. antialias=True reproduces the
      # antialiased PIL bilinear resize used in the authors' gene.py; aspect
      # ratio is deliberately not preserved, as in the original.
      x = F.interpolate(x, size=(icnet_size, icnet_size), mode='bilinear',
         align_corners=False, antialias=True)
      x = (x - mean) / std   # Shape: (batch, 3, 512, 512)
      with torch.no_grad():
         score, complexity_map = model(x)
      # atleast_1d: ICNet calls .squeeze() on its output, which collapses to a
      # 0-d tensor whenever batch_size == 1.
      scores[start:stop] = torch.atleast_1d(score).float().cpu().numpy()
      if return_maps:
         maps[start:stop] = complexity_map.reshape(-1, icnet_map_size,
            icnet_map_size).cpu().numpy()

   del model
   if device.type == 'cuda':
      torch.cuda.empty_cache()

   return scores, maps


# =============================================================================
# Nagle & Lavie preprocessing
# =============================================================================
def preprocess_nagle_lavie(images):
   """Reproduce prep_x_resize_vgg / prep_image_vgg from the authors' release.
 
   Verified bit-identical to github.com/fusionlove/image-complexity.
 
   Parameters
   ----------
   images : ndarray
      uint8 RGB batch of shape (B, H, W, 3).
 
   Returns
   -------
   features : ndarray
      Model input of shape (B, 3, 500, 500), float32, channels_first.
   """
   features = np.zeros((len(images), 3, nagle_lavie_size, nagle_lavie_size),
      dtype=np.float32)   # Shape: (batch, 3, 500, 500)
 
   for i in range(len(images)):
      # skimage.resize on uint8 returns anti-aliased float in [0, 1]
      image = resize(images[i], (nagle_lavie_size, nagle_lavie_size))
      image = np.float32(image)
      image *= 255.
      # Caffe/VGG channel means applied in RGB order, with no BGR swap, exactly
      # as in the authors' prep_image_vgg. Reproduced verbatim: the model was
      # trained this way, so "correcting" the channel order would push inputs
      # off-distribution.
      image[:, :, 0] -= caffe_channel_means[0]
      image[:, :, 1] -= caffe_channel_means[1]
      image[:, :, 2] -= caffe_channel_means[2]
      features[i] = image.transpose((2, 0, 1))   # channels_first, as released
 
   return features
 
 
# =============================================================================
# Layout conversion for TensorFlow builds without oneDNN
# =============================================================================
def to_channels_last(model):
   """Rebuild a channels_first Keras model in channels_last layout.
 
   Conv and Dense kernels are stored layout-independently in Keras, so no
   weight transposition is needed: only the layer configs change. Verified
   numerically equivalent on a channels_first InceptionV3 (max |delta| 1.4e-9).
 
   Parameters
   ----------
   model : keras.Model
      Functional model whose layers are configured as channels_first.
 
   Returns
   -------
   model_nhwc : keras.Model
      Same architecture and weights, expecting (B, H, W, C) input.
   """
   config = model.get_config()
   n_rewritten = [0, 0, 0]
 
   for layer in config['layers']:
      layer_config = layer.get('config', {})
 
      # Keras 3 caches each layer's built input_shape in NCHW form; drop it so
      # the layer rebuilds from the rewritten graph. Keras 2 has no such key.
      layer.pop('build_config', None)
 
      if layer_config.get('data_format') == 'channels_first':
         layer_config['data_format'] = 'channels_last'
         n_rewritten[0] += 1
 
      # Channel-axis arguments: 1 (or -3) becomes -1
      if layer['class_name'] in ('BatchNormalization', 'Concatenate',
         'LayerNormalization', 'GroupNormalization'):
         axis = layer_config.get('axis')
         if isinstance(axis, (list, tuple)):
            new_axis = [-1 if a in (1, -3) else a for a in axis]
            if list(new_axis) != list(axis):
               layer_config['axis'] = type(axis)(new_axis)
               n_rewritten[1] += 1
         elif axis in (1, -3):
            layer_config['axis'] = -1
            n_rewritten[1] += 1
 
      # Input shape: (B, C, H, W) becomes (B, H, W, C)
      for key in ('batch_input_shape', 'batch_shape'):
         shape = layer_config.get(key)
         if shape is not None and len(shape) == 4:
            layer_config[key] = type(shape)(
               [shape[0], shape[2], shape[3], shape[1]])
            n_rewritten[2] += 1
 
   print('Rewrote {} data_format, {} channel-axis and {} input-shape entries'
      .format(*n_rewritten))
 
   model_nhwc = model.__class__.from_config(config)
   model_nhwc.set_weights(model.get_weights())
 
   return model_nhwc
 
 
# =============================================================================
# Nagle & Lavie CNN (2020)
# =============================================================================
def complexity_nagle_lavie(images, model_path, batch_size=8,
   channels_last='auto', n_threads=0):
   """Predict observer-scaled complexity ratings with the Nagle & Lavie CNN.
 
   Preprocessing reproduces prep_x_resize_vgg / prep_image_vgg from
   github.com/fusionlove/image-complexity verbatim.
 
   Parameters
   ----------
   images : ndarray
      uint8 RGB batch of shape (B, H, W, 3).
   model_path : str
      Path to trained_model_inception_v3.h5.
   batch_size : int
      Images per forward pass.
   channels_last : str
      'auto' probes the model and converts only if channels_first fails,
      'always' converts unconditionally, 'never' keeps channels_first.
   n_threads : int
      TensorFlow intra-op threads. 0 leaves TensorFlow's own default.
 
   Returns
   -------
   scores : ndarray
      Complexity scores of shape (B,), on the observer-rating scale (~0-50).
   """
   # Imported lazily: TensorFlow and PyTorch compete for threads and pin
   # incompatible builds, so the TF import stays out of module scope.
   os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
   os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')   # Mute the cpu_feature_guard INFO line
   # channels_first Conv2D/MaxPool have CPU kernels only through oneDNN. Do NOT
   # set TF_ENABLE_ONEDNN_OPTS=0: that reinstates the NHWC-only CPU kernels.
   # These env vars only bite if TF has not already been imported in this
   # process; in a notebook, restart the kernel for them to take effect.
   os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '1')
   import tensorflow as tf
   try:
      from tf_keras.models import load_model   # Keras 2 API; the .h5 predates Keras 3
   except ImportError:
      from keras.models import load_model
 
   # CPU-only execution. Both calls raise once the TF context is initialised,
   # so failures here are benign and just mean the settings are already fixed.
   try:
      tf.config.set_visible_devices([], 'GPU')
   except RuntimeError:
      pass
   if n_threads > 0:
      try:
         tf.config.threading.set_intra_op_parallelism_threads(n_threads)
         tf.config.threading.set_inter_op_parallelism_threads(2)
      except RuntimeError:
         pass
   from tensorflow.python.framework import test_util
   print('\noneDNN enabled: {}'.format(test_util.IsMklEnabled()))
 
   model = load_model(model_path, compile=False)
 
   # Decide the layout on one image, before committing to the full loop.
   probe = preprocess_nagle_lavie(images[:1])   # Shape: (1, 3, 500, 500)
   use_channels_last = channels_last == 'always'
   if use_channels_last:
      model = to_channels_last(model)
   elif channels_last == 'auto':
      try:
         model.predict(probe, verbose=0)
      except (tf.errors.InvalidArgumentError, tf.errors.UnimplementedError) as err:
         print('\nchannels_first kernels unavailable on this device ({}); '
            'converting the model to channels_last'.format(type(err).__name__))
         model = to_channels_last(model)
         use_channels_last = True
   del probe
 
   n_images = len(images)
   scores = np.zeros(n_images, dtype=np.float32)   # Shape: (n_images)
   start_time = time.time()
 
   # Preprocessing is streamed per batch rather than materialised for the whole
   # set: at 500 x 500 x 3 float32 each image costs 3 MB, so a 26k-image
   # stimulus set would otherwise need ~78 GB of RAM.
   for start in tqdm(range(0, n_images, batch_size), desc='Nagle & Lavie'):
      stop = min(start + batch_size, n_images)
      batch = preprocess_nagle_lavie(images[start:stop])   # Shape: (batch, 3, 500, 500)
      if use_channels_last:
         batch = np.ascontiguousarray(batch.transpose(0, 2, 3, 1))
      pred = model.predict(batch, verbose=0)
      scores[start:stop] = np.asarray(pred, dtype=np.float32).reshape(-1)
      del batch
 
   elapsed = time.time() - start_time
   print('\n{} images in {:.1f} s ({:.2f} s/image)'.format(n_images, elapsed,
      elapsed / n_images))
 
   del model
 
   return scores


# =============================================================================
# Compute the complexity scores
# =============================================================================
img_complexity = {}
for key, val in tqdm(images_all.items(), desc='Computing complexity'):

    val = to_uint8_rgb(val, args.input_range)   # Shape: (n_images, height, width, 3)

    if args.methods == 'icnet':
        icnet_scores, icnet_maps = complexity_icnet(val, args.icnet_repo,
            args.icnet_checkpoint, device, args.batch_size,
            return_maps=True)
        img_complexity[key] = icnet_scores   # Shape: (n_images)

    if args.methods == 'nagle_lavie':
        img_complexity[key] = complexity_nagle_lavie(val,
            args.nagle_lavie_model, args.batch_size, args.channels_last,
            n_threads)   # Shape: (n_images)


# =============================================================================
# Test for significant differences in image complexity between the two sets of
# controlling images
# =============================================================================
p_val = ttest_ind(img_complexity['high_1_low_2'],
    img_complexity['low_1_high_2'], alternative='less').pvalue


# =============================================================================
# Save the results
# =============================================================================
stats = {
    'img_complexity': img_complexity,
    'p_val': p_val
    }

save_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
    'within_area_dynamics', 'rnc', 'stats', 'cv-0', args.time_window_pair,
    f'imageset-{args.imageset}')
os.makedirs(save_dir, exist_ok=True)

file_name = f'stats_images_roi-{args.roi}_method-{args.methods}.npy'

np.save(os.path.join(save_dir, file_name), stats)