# Pycortex ROI labels

For the encoding models trained on NSD, we plotted the results of some analyses on flattened cortical surfaces using [pycortex' fsaverage subject](https://figshare.com/articles/dataset/fsaverage_subject_for_pycortex/9916166).

However, since the ROI surface labels of pycortex' fsaverage subject are based on templates, we replaced these labels with the [subject-defined ROI lebels provided in NSD](https://cvnlab.slite.page/p/X_7BBMgghj/ROIs) (using the Python scripts found in this GitHub directory, and Inkscape). 

To use these subject-specific ROI labels, copy the `./fsaverage_nsd_sub-0*/overlays.svg` files from this GitHub directory onto your local pycortex surface folder of the subject of interest (within an Anaconda environment, you should find this folder at: `../anaconda3/envs/env_name/share/pycortex/db/fsaverage_nsd_sub-0*/overlays.svg`).

Once you copied the `overlays.svg` files, you can plot fMRI surfaces with the NSD subject-specific ROI labels using the following Python commands:

```python
import numpy as np
import cortex

# Create dummy data in fsaverage space
data = np.random.randn(163842*2)

# Choose the NSD subject you to plot
subject = 'fsaverage_nsd_sub-01'

# Create the flat brain surface
vertex_data = cortex.Vertex(
    data,
    subject=subject,
    cmap='coolwarm',
    vmin=None,
    vmax=None,
    with_colorbar=True
    )

# Plot the flat brain surface
fig = cortex.quickshow(
    vertex_data,
    height=2000, # Increase resolution of map and ROI contours
    with_curvature=True,
    with_rois=True,
    roi_list=['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'FFA-1',
        'FFA-2', 'EBA', 'PPA', 'Early', 'Intermediate', 'Ventral', 'Lateral',
        'Dorsal'],
    linewidth=3,
    linecolor=(1, 1, 1),
    with_labels=True,
    labelsize=25,
    curvature_brightness=0.5,
    with_colorbar=True
    )
```