# Model: `fmri_nsd_fwrf`

**Version**: 1.0.0  
**Modality**: fmri  
**Dataset**: NSD  
**Repeats**: x  
**Features**: x  
**Subject-level**: x  

## Supported Parameters

### `subject`
- **Type**: int
- **Required**: True
- **Valid Values**: 1, 2, 3, 4, 5, 6, 7, 8
- **Example**: `1`
- **Description**: Subject ID from the NSD dataset

### `roi`
- **Type**: str
- **Required**: True
- **Valid Values**: V1, V2, V3, hV4, EBA, FBA-2, OFA, FFA-1, FFA-2, PPA, RSC, OPA, OWFA, VWFA-1, VWFA-2, mfs-words, early, midventral, midlateral, midparietal, parietal, lateral, ventral
- **Example**: `V1`
- **Description**: Region of Interest (ROI) for voxel prediction

### `nest_dir`
- **Type**: str
- **Required**: False
- **Example**: `./`
- **Description**: Root directory of the NEST repository (optional if default paths are set)

