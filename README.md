# Neural Encoding Simulation Toolkit (NEST)

**Neural Encoding Simulation Toolkit (NEST)** is a Python package that provides trained encoding models of the brain that you can use to generate accurate *in silico* neural responses to arbitrary stimuli with just a few lines of code.

*In silico* neural responses from encoding models increasingly resemble *in vivo* responses recorded from real brains, enabling the novel research paradigm of *in silico* neuroscience. These simulated responses are fast and cost-effective to generate, allowing researchers to explore and test scientific hypotheses across vastly larger solution spaces than possible with traditional *in vivo* methods.

Novel findings from large-scale *in silico* experimentation can be validated through targeted small-scale *in vivo* data collection, optimizing research resources. This approach scales beyond what is possible with *in vivo* data alone and democratizes research across groups with diverse data collection infrastructure and resources.

NEST includes a growing, well-documented library of encoding models trained on different neural data acquisition modalities (including fMRI and EEG), datasets, subjects, stimulation types, and brain areas, offering broad versatility for addressing a wide range of research questions through *in silico* neuroscience.


For additional information on the Neural Encoding Dataset you can check out our [documentation][nest_website].




## 🤝 Contribute to Expanding NEST

We welcome contributions to expand NEST! If you have:
- Encoding models with higher prediction accuracies
- Models for new neural datasets, data modalities (e.g., MEG/ECoG/animal), or stimulus types (e.g., videos, language)
- Suggestions for improving NEST

For more information on how to contribute please refer to [How to Contribute][nest_contribute]. 

Please contact us at brain.nest.contact@gmail.com. All feedback and help is strongly appreciated!




## ⚙️ Installation

To install `NEST` run the following command on your terminal:

```shell
pip install -U git+https://github.com/gifale95/NEST.git
```

You will additionally need to install the Python dependencies found in [requirements.txt][requirements].



## 🕹️ How to use

### 🧰 Download the Neural Encoding Simulation Toolkit encoding models


To use `NEST`, you first need to download the trained encoding models from [here][nest_data]. Depending on your needs, you might download all or only parts of the toolkit. Please refer to the [documentation][nest_structure] to understand how NEST is structured.
We recommend downloading the folder directly from Google Drive via terminal using [Rclone][rclone]. [Here][guide] is a step-by-step guide for using Rclone. Before downloading, add a shortcut of the `neural_encoding_simulation_toolkit` folder to your Google Drive by right-clicking on the folder and selecting `Organise` → `Add shortcut`.




### 🧠 Available encoding models

The following table shows the encoding models currently available in NEST. For more details on these models feel free to check out the [model cards][model_cards] in our documentation page.

| modality | train_dataset | model | subject | roi |
|-------------|-----------------------|----------| ----------| ----|
| fmri | nsd | fwrf | 1, 2, 3, 4, 5, 6, 7, 8 | V1, V2, V3, hV4, EBA, FBA-2, OFA, FFA-1, FFA-2, PPA, RSC, OPA, OWFA, VWFA-1, VWFA-2, mfs-words, early, midventral, midlateral, midparietal, parietal, lateral, ventral|
| eeg | things_eeg_2 | vit_b_32 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10| – |
 


### ✨ NEST functions

#### 🔹 Initialize the NEST object

To use `NEST`'s functions you need to import `NEST` and create a `nest_object`.

```python
from nest import NEST

# Initialize NEST with the path to the toolkit
nest = NEST(nest_dir="path/to/neural_encoding_simulation_toolkit")
```
#### 🔹 Generate in silico neural responses to the stimuli of your choice

Step 1: Load an encoding model

```python
# Load an example fMRI encoding model
fmri_model = nest.get_encoding_model("fmri-nsd-fwrf", 
                                     subject=1,
                                     roi="V1",
                                     device="cpu")

# Load an example EEG encoding model
eeg_model = nest.get_encoding_model("eeg-things_eeg_2-vit_b_32",
                                    subject=1,
                                    device="auto")

```

Step 2: Generate responses for your images

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
- [fMRI Tutorial](https://colab.research.google.com/drive/1W9Sroz2Y0eTYfyhVrAJwe50GGHHAGBdE?usp=drive_link) - Learn how to generate in silico fMRI responses 
- [EEG Tutorial](https://colab.research.google.com/drive/10NSRBrJ390vuaPyRWq5fDBIA4NNAUlTk?usp=drive_link) - Explore how to generate time-resolved EEG responses 
- [Adding New Models](https://neural-encoding-simulation-toolkit.readthedocs.io/en/latest/tutorials/Adding_New_Models_to_Nest.html) - Guide on how to implement and contribute your own encoding models to NEST

**Example Application - Relational Neural Control (RNC):**

[RNC](https://github.com/gifale95/RNC) is an example application using NEST to find images that selectively activate specific brain regions:

- [Univariate RNC Tutorial](https://colab.research.google.com/drive/1QpMSlvKZMLrDNeESdch6AlQ3qKsM1isO?usp=sharing) 
- [Multivariate RNC Tutorial](https://colab.research.google.com/drive/1bEKCzkjNfM-jzxRj-JX2zxB17XBouw23?usp=sharing) 



## 📦 Neural Encoding Simulation Toolkit creation code

The folder [`../NEST/nest_creation_code/`][nest_creation_code] contains the code used to create the Neural Encoding Simulation Toolkit, divided in the following sub-folders:

* **[`../00_prepare_data/`][prepare_data]:** prepare the data (i.e., images and corresponding neural responses) used to train the encoding models.
* **[`../01_train_encoding_models/`][train_encoding]:** train the encoding models, and save their weights.
* **[`../02_test_encoding_models/`][test_encoding]:** test the encoding models (i.e., compute and plot their encoding accuracy).
* **[`../03_create_metadata/`][metadata]:** create metadata files relative to the encoding models and their in silico neural responses.



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



[fmri_tutorial_colab]: https://colab.research.google.com/drive/1W9Sroz2Y0eTYfyhVrAJwe50GGHHAGBdE?usp=drive_link
[eeg_tutorial_colab]: https://colab.research.google.com/drive/10NSRBrJ390vuaPyRWq5fDBIA4NNAUlTk?usp=drive_link
[fmri_tutorial_jupyter]: https://github.com/gifale95/NEST/blob/main/tutorials/nest_fmri_tutorial.ipynb
[eeg_tutorial_jupyter]: https://github.com/gifale95/NEST/blob/main/tutorials/nest_eeg_tutorial.ipynb
[uni_rnc_colab]: https://colab.research.google.com/drive/1QpMSlvKZMLrDNeESdch6AlQ3qKsM1isO?usp=sharing
[multi_rnc_colab]: https://colab.research.google.com/drive/1bEKCzkjNfM-jzxRj-JX2zxB17XBouw23?usp=sharing
[uni_rnc_jupyter]: https://github.com/gifale95/RNC/blob/main/tutorials/univariate_rnc_tutorial.ipynb
[multi_rnc_jupyter]: https://github.com/gifale95/RNC/blob/main/tutorials/multivariate_rnc_tutorial.ipynb
[nest_creation_code]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/
[prepare_data]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/00_prepare_data
[train_encoding]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/01_train_encoding_models
[test_encoding]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/02_test_encoding_models
[metadata]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/03_create_metadata
[synthesize]: https://github.com/gifale95/NEST/tree/main/nest_creation_code/04_synthesize_neural_responses

