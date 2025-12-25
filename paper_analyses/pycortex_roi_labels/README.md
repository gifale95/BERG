# Pycortex ROI labels

For the encoding models trained on NSD, we plotted the results of some analyses on flattened cortical surfaces using [pycortex' `fsaverage` subject](https://figshare.com/articles/dataset/fsaverage_subject_for_pycortex/9916166).

However, since the ROI surface labels of pycortex' fsaverage subject are based on templates, we replaced these labels with the [subject-specific ROI lebels provided in NSD](https://cvnlab.slite.page/p/X_7BBMgghj/ROIs) (using the Python scripts found in this GitHub directory, and Inkscape).

To use these subject-specific ROI labels, follow these steps:

1. Download the pycortex' `fsaverage` subject (within an Anaconda environment, you should find the downloaded `fsaverage` subject at: `../anaconda3/envs/env_name/share/pycortex/db/fsaverage/`).

2. Make one copy of the `fsaverage` subject for each of the 8 NSD subjects:

```bash
cd ../anaconda3/envs/env_name/share/pycortex/db/
cp fsaverage -r fsaverage_nsd_sub-0*
```

3. Replace the overlay files in your your pycortex surface folder of the subject of interest (`../anaconda3/envs/env_name/share/pycortex/db/fsaverage_nsd_sub-0*/overlays.svg`) files with the overlay files from this GitHub directory (`./fsaverage_nsd_sub-0*/overlays.svg`).

4. You can now plot fMRI surfaces with the NSD subject-specific ROI labels using the following Python commands:

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