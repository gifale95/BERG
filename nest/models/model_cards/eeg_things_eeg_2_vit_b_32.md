# Model: `eeg_things_eeg_2_vit_b_32`

**Version**: 1.0.0  
**Modality**: eeg  
**Dataset**: things_eeg_2  
**Repeats**: multi  
**Features**: vision_transformer  
**Subject-level**: x  

## Supported Parameters

### `subject`
- **Type**: int
- **Required**: True
- **Valid Values**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- **Example**: `1`
- **Description**: Subject ID from the things_eeg_2 dataset

### `nest_dir`
- **Type**: str
- **Required**: False
- **Example**: `./`
- **Description**: Root directory of the NEST repository (optional if default paths are set)

