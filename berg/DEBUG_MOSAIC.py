import h5py
import matplotlib
from matplotlib import pyplot as plt
from berg import BERG
import numpy as np
import os
from PIL import Image
import torchvision
from torchvision import transforms as trn
from tqdm import tqdm
from IPython.display import display, JSON

berg_dir = "/home/ale/aaa_stuff/science/projects/brain-encoding-response-generator" 

images = np.random.randint(0, 255, (1, 3, 224, 224))

# Print the images dimensions
print('\n\nImages shape:')
print(images.shape)
print('(Batch size × 3 RGB Channels x Width x Height)')

# Initialize the BERG object with the path to the toolkit directory
berg = BERG(berg_dir)


# =============================================================================
# Test model (CNN8_multihead_subNSD_verticesAll)
# =============================================================================
model_id = 'fmri-mosaic-CNN8_multihead_subNSD_verticesAll'
subject = 1

model_full = berg.get_encoding_model(model_id, 
                                     subject='all', 
                                     device="auto")

pred_full = berg.encode(model_full, images)

metadata = berg.get_model_metadata(model_id, subject='all')
print(metadata.keys())

for dataset, subjects in pred_full.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)


# =============================================================================
# Testing single ROI
# =============================================================================
model_v1 = berg.get_encoding_model(model_id, 
                                   subject=subject, 
                                   selection={"roi": ["L_V1"]}, 
                                   device="auto")
pred_v1 = berg.encode(model_v1, images)
for dataset, subjects in pred_v1.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)
print("metadata: ", metadata["fmri"]["roi"]["L_V1"].shape)


# =============================================================================
# Does the slicing work correctly?
# =============================================================================
subject = 1

# Model Full
model_full = berg.get_encoding_model(
    model_id, 
    subject=subject, 
    device="auto"
)
pred_full = berg.encode(model_full, images)
metadata = berg.get_model_metadata(model_id, subject=subject)


# Model ROI
roi_to_test = "L_V1"
model_roi = berg.get_encoding_model(
    model_id, 
    subject=subject, 
    selection={"roi": [roi_to_test]}, 
    device="auto"
)
pred_roi_direct = berg.encode(model_roi, images)


# Print our data shapes
pred_full_array = pred_full['NaturalScenesDataset'][f'sub-{subject:02d}']  # (57051,)
print(f"Full predictions shape: {pred_full_array.shape}")

pred_roi_direct_array = pred_roi_direct['NaturalScenesDataset'][f'sub-{subject:02d}']  # (816,)
print(f"Roi predictions shape: {pred_roi_direct_array.shape}")  

vertex_mapping = metadata["encoding_models"]["vertex_mapping_all"]  # (57051,)
print(f"Vertex mapping shape: {vertex_mapping.shape}")

roi_indices = metadata["fmri"]["roi"][roi_to_test]  # (816,)
print(f"ROI '{roi_to_test}' has {len(roi_indices)} vertices")


# Expand predictions to 91k space
pred_91k = np.full((pred_full_array.shape[0], 91282), np.nan)
pred_91k[:, vertex_mapping] = pred_full_array

# Slice with ROI indices
pred_roi_manual = pred_91k[:, roi_indices]

# Extract direct ROI result
pred_roi_direct_array = pred_roi_direct['NaturalScenesDataset'][f'sub-{subject:02d}']

# Compare
print(f"\nDirect shape: {pred_roi_direct_array.shape}")
print(f"Manual shape: {pred_roi_manual.shape}")
print(f"Match: {np.allclose(pred_roi_direct_array, pred_roi_manual, rtol=1e-5)}")


# =============================================================================
# Testing two ROIs
# =============================================================================
model_v1v2 = berg.get_encoding_model(model_id, 
                                     subject=subject, 
                                     selection={"roi": ['L_V1', 'R_V1']}, 
                                     device="auto")
pred_v1v2 = berg.encode(model_v1v2, images)


for dataset, subjects in pred_v1v2.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)

print("metadata: ", len(metadata["fmri"]["roi"]["L_V1"]) + len(metadata["fmri"]["roi"]["R_V1"]))


# =============================================================================
# Testing last 10 voxels
# =============================================================================
voxel_mask = np.zeros(57051, dtype=int)

voxel_mask[-10:] = 1
model_voxel = berg.get_encoding_model(model_id, 
                                      subject=subject, 
                                      selection={"voxel_index": voxel_mask}, 
                                      device="auto")
pred_voxel = berg.encode(model_voxel, images)



for dataset, subjects in pred_voxel.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)


# =============================================================================
# Testing combination
# =============================================================================
model_combined = berg.get_encoding_model(model_id, 
                                         subject=subject, 
                                         selection={"roi": ["L_V1"], "voxel_index": voxel_mask}, 
                                         device="auto")
pred_combined = berg.encode(model_combined, images)

for dataset, subjects in pred_combined.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)
        
        
print("metadata: ", len(metadata["fmri"]["roi"]["L_V1"]) + 10)


# =============================================================================
# Testing list of subjects
# =============================================================================
subject = [1,2]

model_combined = berg.get_encoding_model(model_id, 
                                         subject=subject, 
                                         selection={"roi": ["L_V1"], "voxel_index": voxel_mask}, 
                                         device="auto")
pred_combined_multi_sub = berg.encode(model_combined, images)

for dataset, subjects in pred_combined_multi_sub.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)


# =============================================================================
# Testing all subjects
# =============================================================================
subject = "all"

model_combined = berg.get_encoding_model(model_id, 
                                         subject=subject, 
                                         selection={"roi": ["L_V1"], "voxel_index": voxel_mask}, 
                                         device="auto")
pred_all = berg.encode(model_combined, images)

for dataset, subjects in pred_all.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)


















# =============================================================================
# Test model (CNN8_multihead_subNSD_verticesAll)
# =============================================================================
model_id = 'fmri-mosaic-CNN8_multihead_subAll_verticesVisual'
subject = 'NSD-03'

model_full = berg.get_encoding_model(model_id, 
                                     subject=subject, 
                                     device="auto")
pred_full = berg.encode(model_full, images)


metadata = berg.get_model_metadata(model_id, subject=subject)

print(metadata.keys())

for dataset, subjects in pred_full.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)


# =============================================================================
# Does slicing work correclty?
# =============================================================================
metadata['NaturalScenesDataset']['sub-03'].keys()

subject = 'NSD-03'

# Model Full
model_full = berg.get_encoding_model(
    model_id, 
    subject=subject, 
    device="auto"
)
pred_full = berg.encode(model_full, images)
metadata = berg.get_model_metadata(model_id, subject=subject)


# Model ROI
roi_to_test = "L_V1"
model_roi = berg.get_encoding_model(
    model_id, 
    subject=subject, 
    selection={"roi": [roi_to_test]}, 
    device="auto"
)
pred_roi_direct = berg.encode(model_roi, images)


# Print our data shapes
pred_full_array = pred_full['NaturalScenesDataset'][f'sub-03']  # (7831,)
print(f"Full predictions shape: {pred_full_array.shape}")

pred_roi_direct_array = pred_roi_direct['NaturalScenesDataset'][f'sub-03']  # (816,)
print(f"Roi predictions shape: {pred_roi_direct_array.shape}")  

vertex_mapping = metadata['NaturalScenesDataset']['sub-03']["encoding_models"]["vertex_mapping_visual"]  # (7831,)
print(f"Vertex mapping shape: {vertex_mapping.shape}")

roi_indices = metadata['NaturalScenesDataset']['sub-03']["fmri"]["roi"][roi_to_test]  # (816,)
print(f"ROI '{roi_to_test}' has {len(roi_indices)} vertices")


# Expand predictions to 91k space
pred_91k = np.full((pred_full_array.shape[0], 91282), np.nan)
pred_91k[:, vertex_mapping] = pred_full_array

# Slice with ROI indices
pred_roi_manual = pred_91k[:, roi_indices]

# Extract direct ROI result
pred_roi_direct_array = pred_roi_direct['NaturalScenesDataset'][f'sub-03']

# Compare
print(f"\nDirect shape: {pred_roi_direct_array.shape}")
print(f"Manual shape: {pred_roi_manual.shape}")
print(f"Match: {np.allclose(pred_roi_direct_array, pred_roi_manual, rtol=1e-5)}")


# =============================================================================
# Testing single ROIs
# =============================================================================
model_v1 = berg.get_encoding_model(model_id, 
                                   subject=subject, 
                                   selection={"roi": ["L_V1"]}, 
                                   device="auto")

pred_v1 = berg.encode(model_v1, images)

for dataset, subjects in pred_v1.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)
        
        
print("metadata: ", metadata['NaturalScenesDataset']['sub-03']["fmri"]["roi"]["L_V1"].shape)


# =============================================================================
# Testing two ROIs
# =============================================================================
model_v1v2 = berg.get_encoding_model(model_id, 
                                     subject=subject, 
                                     selection={"roi": ['L_V1', 'R_V1']}, 
                                     device="auto")
pred_v1v2 = berg.encode(model_v1v2, images)


for dataset, subjects in pred_v1v2.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)

print("metadata: ", len(metadata['NaturalScenesDataset']['sub-03']["fmri"]["roi"]["L_V1"]) + len(metadata['NaturalScenesDataset']['sub-03']["fmri"]["roi"]["R_V1"]))


# =============================================================================
# Testing last 10 voxels
# =============================================================================
voxel_mask = np.zeros(7831, dtype=int)

voxel_mask[-10:] = 1
model_voxel = berg.get_encoding_model(model_id, 
                                      subject=subject, 
                                      selection={"voxel_index": voxel_mask}, 
                                      device="auto")
pred_voxel = berg.encode(model_voxel, images)



for dataset, subjects in pred_voxel.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)


# =============================================================================
# Testing combination
# =============================================================================
model_combined = berg.get_encoding_model(model_id, 
                                         subject=subject, 
                                         selection={"roi": ["L_V1"], "voxel_index": voxel_mask}, 
                                         device="auto")
pred_combined = berg.encode(model_combined, images)

for dataset, subjects in pred_combined.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)
        
        
print("metadata: ", len(metadata['NaturalScenesDataset']['sub-03']["fmri"]["roi"]["L_V1"]) + 10)


# =============================================================================
# Testing list of subjects
# =============================================================================
subject= ['THINGS-02', 'NSD-03']

model_combined = berg.get_encoding_model(model_id, 
                                         subject=subject, 
                                         selection={"roi": ["L_V1"], "voxel_index": voxel_mask}, 
                                         device="auto")
pred_combined_multi_sub = berg.encode(model_combined, images)

for dataset, subjects in pred_combined_multi_sub.items():
    for subject_data, data in subjects.items():
        print(dataset, "-", subject_data, data.shape)
