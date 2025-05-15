# Neural Encoding Simulation Toolkit (NEST)

**Neural Encoding Simulation Toolkit (NEST)** is a resource consisting of of multiple pre-trained encoding models of the brain and an accompanying Python package to generate accurate in silico neural responses to arbitrary stimuli with just a few lines of code.

In silico neural responses from encoding models increasingly resemble in vivo responses recorded from real brains, enabling the novel research paradigm of in silico neuroscience. In silico neural responses are quick and cheap to generate, allowing researchers to explore and test scientific hypotheses across vastly larger solution spaces than possible in vivo. Novel findings from large-scale in silico experimentation are then validated through targeted small-scale in vivo data collection, in this way optimizing research resources. Thus, in silico neuroscience scales beyond what is possible with in vivo data, and democratizes research across groups with diverse data collection infrastructure and resources. To catalyze this emerging research paradigm, we introduce the Neural Encoding Simulation Toolkit (NEST), a resource consisting of multiple pre-trained encoding models of the brain and an accompanying Python package to generate accurate in silico neural responses to arbitrary stimuli with just a few lines of code. NEST includes a growing, well documented library of encoding models trained on different neural data acquisition modalities, datasets, subjects, stimulation types, and brain areas, offering broad versatility for addressing a wide range of research questions through in silico neuroscience.

For additional information on the Neural Encoding Dataset you can check out our [documentation][nest_website].



## 🤝 Contribute to Expanding NEST

We warmly welcome contributions to improve and expand NEST, including:
- Encoding models with higher prediction accuracies.
- Encoding models for new neural data recording modalities (e.g., MEG/ECoG/animal).
- Encoiding models from new neural dataset.
- Encoding models of neural responses for new stimulus types (e.g., videos, audio, language, multimodal).
- Suggestions to improve NEST.

For more information on how to contribute, please refer to [our documentation][nest_contribute]. If you have questions or would like to discuss your contribution before submitting, please contact us at brain.nest.contact@gmail.com. All feedback and help is strongly appreciated!



## ⚙️ Installation

To install `NEST` run the following command on your terminal:

```shell
pip install -U git+https://github.com/gifale95/NEST.git
```

You will additionally need to install the Python dependencies found in [requirements.txt][requirements].



## 🕹️ How to use

### 🧰 Download the Neural Encoding Simulation Toolkit encoding models


NEST is hosted as a public AWS S3 bucket via the AWS Open Data Program. You do **not need an AWS account** to [browse](https://neural-encoding-simulation-toolkit.s3.us-west-2.amazonaws.com/index.html) or download the data.

To download the full dataset into a local folder named `neural-encoding-simulation-toolkit`, use the AWS CLI:

```bash
aws s3 sync --no-sign-request s3://neural-encoding-simulation-toolkit ./neural-encoding-simulation-toolkit
```

You can also download specific subfolders, for example:

```bash
aws s3 sync --no-sign-request s3://neural-encoding-simulation-toolkit/encoding_models/modality-fmri ./modality-fmri
```

For detailed instructions and folder structure, see the [full documentation](https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/data_storage.html#).


### 🧠 Available encoding models

The following table shows the encoding models currently available in NEST. For more details on these models, please refer to the [documentation][model_cards].

| Model ID | Training dataset | Species | Stimuli |
|----------|------------------|---------|---------|
| [fmri-nsd_fsaverage-vit_b_32][fmri-nsd_fsaverage-vit_b_32] | [NSD (surface space)][allen] | Human | Images |
| [fmri-nsd-fwrf][fmri-nsd-fwrf] | [NSD (volume space)][allen] | Human | Images |
| [eeg-things_eeg_2-vit_b_32][eeg-things_eeg_2-vit_b_32] | [THINGS EEG2][THINGS EEG2] | Human | Images |
 


### ✨ NEST functions

#### 🔹 Initialize the NEST object

To use `NEST`'s functions, you first need to import `NEST` and create a `nest_object`.

```python
from nest import NEST

# Initialize NEST with the path to the toolkit
nest = NEST(nest_dir="path/to/neural_encoding_simulation_toolkit")
```
#### 🔹 Generate in silico neural responses to stimuli

Step 1: Load an encoding model of your choice using the `get_encoding_model` function.

```python
# Load an example fMRI encoding model
fmri_model = nest.get_encoding_model("fmri-nsd-fwrf", 
                                     subject=1,
                                     selection={"roi": "V1"},
                                     device="cpu")

# Load an example EEG encoding model
eeg_model = nest.get_encoding_model("eeg-things_eeg_2-vit_b_32",
                                    subject=1,
                                    device="auto")

```

Step 2: Generate in silico neural responses to stimuli using the `encode` function.

```python
# Encode fMRI responses to images with metadata
insilico_fmri, insilico_fmri_metadata = nest.encode(fmri_model,
                                                   images,
                                                   return_metadata=True)  # if needed

# Encode EEG responses to images without metadata
insilico_eeg = nest.encode(eeg_model,
                          images)
```

For more detailed information on how to use these functions and which parameters are available, please refer to the **Tutorials** below ⬇️.


### 💻 Tutorials

We provide several tutorials to help you get started with NEST:

**Using NEST:**
- [Quickstart Tutorial](https://colab.research.google.com/drive/1JS4um1eS4Ml983lUNQgEw4544_Lc5Qn0) - Quick Guide on how to generate in silico neural responses
- [fMRI Tutorial](https://colab.research.google.com/drive/1w4opmM9h8Oe1NWlwIDuLuDIGuIXj9UaV) - Learn how to generate in silico fMRI responses.
- [EEG Tutorial](https://colab.research.google.com/drive/1uF5nr1pyg0_my3gULj3w5y0nuq5gZjhL) - Explore how to generate EEG responses.
- [Adding New Models](https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/tutorials/Adding_New_Models_to_Nest.html) - Guide on how to implement and contribute your own encoding models to NEST.

**Example Application - Relational Neural Control (RNC):**

We used NEST to develop [RNC](https://github.com/gifale95/RNC), a neural control algorithm to move from an atomistic understanding of visual cortical areas (i.e., What does each area represent?) to a network-level understanding (i.e., What is the relationship between representations in different areas?). Through RNC we discovered controlling images that align or disentangle responses across areas, thus indicating their shared or unique representational content. Closing the empirical cycle, we validated the in silico discoveries on in vivo fMRI responses from independent subjects. Following are RNC tutorials based on in silico fMRI responses generated through NEST:

- [Univariate RNC Tutorial](https://colab.research.google.com/drive/1QpMSlvKZMLrDNeESdch6AlQ3qKsM1isO?usp=sharing) 
- [Multivariate RNC Tutorial](https://colab.research.google.com/drive/1bEKCzkjNfM-jzxRj-JX2zxB17XBouw23?usp=sharing) 



## 📦 Neural Encoding Simulation Toolkit creation code

The folder [`../NEST/nest_creation_code/`][nest_creation_code] contains the code used to create the Neural Encoding Simulation Toolkit, divided in the following sub-folders:

* **[`../01_prepare_data/`][prepare_data]:** prepare the neural responses in the right format for encoding model training.
* **[`../02_train_encoding_models/`][train_encoding]:** train the encoding models, and save their weights.
* **[`../03_test_encoding_models/`][test_encoding]:** test the encoding models (i.e., compute and plot their encoding accuracy).



## ❗ Issues

If you come across problems with this Python package, please submit an issue!



## 📜 Citation

If you use the Neural Encoding Simulation Toolkit, please cite:

> *Gifford AT, Bersch D, Roig G, Cichy RM. 2025. The Neural Encoding Simulation Toolkit. In preparation. https://github.com/gifale95/NEST*


[nest_website]: https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/
[nest_structure]: https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/data_storage.html#
[model_cards]: https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/models/overview.html
[nest_contribute]: https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/contribution.html
[imagenet]: https://www.image-net.org/challenges/LSVRC/2012/index.php
[russakovsky]: https://link.springer.com/article/10.1007/s11263-015-0816-y
[things]: https://things-initiative.org/
[hebart]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792
[nsd]: https://naturalscenesdataset.org/
[allen]: https://www.nature.com/articles/s41593-021-00962-x
[requirements]: https://github.com/gifale95/NEST/blob/main/requirements.txt
[rclone]: https://rclone.org/
[guide]: https://noisyneuron.github.io/nyu-hpc/transfer.html
[nest_data]: https://forms.gle/ZKxEcjBmdYL6zdrg9
[data_manual]: https://docs.google.com/document/d/1DeQwjq96pTkPEnqv7V6q9g_NTHCjc6aYr6y3wPlwgDE/edit?usp=drive_link


[get_encoding_model]: https://github.com/gifale95/NEST/blob/main/nest/nest.py#L207
[encode]: https://github.com/gifale95/NEST/blob/main/nest/nest.py#L321
[load_insilico_neural_responses]: https://github.com/gifale95/NEST/blob/main/nest/nest.py#L551


[fmri-nsd_fsaverage-vit_b_32]: https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/models/model_cards/fmri-nsd_fsaverage-vit_b_32.html
[fmri-nsd-fwrf]: https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/models/model_cards/fmri-nsd-fwrf.html
[eeg-things_eeg_2-vit_b_32]: https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/models/model_cards/eeg-things_eeg_2-vit_b_32.html
[THINGS EEG2]: https://doi.org/10.1016/j.neuroimage.2022.119754


[fmri_tutorial_colab]: https://colab.research.google.com/drive/1W9Sroz2Y0eTYfyhVrAJwe50GGHHAGBdE?usp=drive_link
[eeg_tutorial_colab]: https://colab.research.google.com/drive/10NSRBrJ390vuaPyRWq5fDBIA4NNAUlTk?usp=drive_link
[fmri_tutorial_jupyter]: https://github.com/gifale95/NEST/blob/main/tutorials/nest_fmri_tutorial.ipynb
[eeg_tutorial_jupyter]: https://github.com/gifale95/NEST/blob/main/tutorials/nest_eeg_tutorial.ipynb
[uni_rnc_colab]: https://colab.research.google.com/drive/1QpMSlvKZMLrDNeESdch6AlQ3qKsM1isO?usp=sharing
[multi_rnc_colab]: https://colab.research.google.com/drive/1bEKCzkjNfM-jzxRj-JX2zxB17XBouw23?usp=sharing
[uni_rnc_jupyter]: https://github.com/gifale95/RNC/blob/main/tutorials/univariate_rnc_tutorial.ipynb
[multi_rnc_jupyter]: https://github.com/gifale95/RNC/blob/main/tutorials/multivariate_rnc_tutorial.ipynb
[nest_creation_code]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/
[prepare_data]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/01_prepare_data
[train_encoding]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/02_train_encoding_models
[test_encoding]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/03_test_encoding_models
[metadata]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/03_create_metadata
[synthesize]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/04_synthesize_neural_responses

