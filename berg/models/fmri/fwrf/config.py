# files
import sys
import os
import struct
import time
import numpy as np
import h5py
from scipy.io import loadmat
from scipy.stats import pearsonr
from tqdm import tqdm
import pickle
import math
import matplotlib.pyplot as plt
import seaborn as sns

fpX = np.float32

root_dir   = os.getcwd() + '/'
net_dir    = root_dir + "net/" 
video_dir  = root_dir+"video/"

stim_dir = root_dir+'../../data/nsd/stims/'
mask_dir = root_dir+'../../data/nsd/masks/'    
voxel_dir = root_dir+'../../data/nsd/voxels/'
    
exp_design_file = root_dir+"../../data/nsd/nsd_expdesign.mat"   
exp_design = loadmat(exp_design_file)
ordering = exp_design['masterordering'].flatten() - 1 # zero-indexed ordering of indices (matlab-like to python-like)

if not os.path.exists(video_dir):
    os.makedirs(video_dir)
if not os.path.exists(net_dir):
    os.makedirs(net_dir)
    
    
# graphics
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Patch

saveext = ".png"
savearg = {'format':'png', 'dpi': 120, 'facecolor': 'None'}

subj_cmap = [cm.get_cmap('rainbow', 9)(k) for k in range(9)]
roi_cmap = [cm.get_cmap('rainbow', 5)(0), cm.get_cmap('rainbow', 5)(1), cm.get_cmap('rainbow', 5)(3), cm.get_cmap('rainbow', 5)(4)]

sns.axes_style()
sns.set_style("whitegrid", {"axes.facecolor": '.95'})
sns.set_context("notebook", rc={'axes.labelsize': 18.0, 'axes.titlesize': 24.0, 'legend.fontsize': 18.0, 'xtick.labelsize': 18.0, 'ytick.labelsize': 18.0})
sns.set_palette("deep")