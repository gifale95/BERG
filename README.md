# Brain Encoding Response Generator (BERG)

The **Brain Encoding Response Generator (BERG)** is a resource consisting of of multiple pre-trained encoding models of the brain and an accompanying Python package to generate accurate in silico neural responses to arbitrary stimuli with just a few lines of code.

In silico neural responses from encoding models increasingly resemble in vivo responses recorded from real brains, enabling the novel research paradigm of in silico neuroscience. In silico neural responses are quick and cheap to generate, allowing researchers to explore and test scientific hypotheses across vastly larger solution spaces than possible in vivo. Novel findings from large-scale in silico experimentation are then validated through targeted small-scale in vivo data collection, in this way optimizing research resources. Thus, in silico neuroscience scales beyond what is possible with in vivo data, and democratizes research across groups with diverse data collection infrastructure and resources. To catalyze this emerging research paradigm, we introduce the Brain Encoding Response Generator (BERG), a resource consisting of multiple pre-trained encoding models of the brain and an accompanying Python package to generate accurate in silico neural responses to arbitrary stimuli with just a few lines of code. BERG includes a growing, well documented library of encoding models trained on different neural data acquisition modalities, datasets, subjects, stimulation types, and brain areas, offering broad versatility for addressing a wide range of research questions through in silico neuroscience.

For additional information on BERG, you can check out our [documentation][documentation].



## 🤝 Contribute to Expanding BERG

We warmly welcome contributions to improve and expand BERG, including:
- Encoding models with higher prediction accuracies.
- Encoding models for new neural data recording modalities (e.g., MEG/ECoG/animal).
- Encoiding models from new neural dataset.
- Encoding models of neural responses for new stimulus types (e.g., videos, audio, language, multimodal).
- Suggestions to improve BERG.

For more information on how to contribute, please refer to [our documentation][berg_contribute]. If you have questions or would like to discuss your contribution before submitting, please contact us at brain.berg.info@gmail.com. All feedback and help is strongly appreciated!



## ⚙️ Installation

To install `BERG` run the following command on your terminal:

```shell
pip install -U git+https://github.com/gifale95/BERG.git
```

You will additionally need to install the Python dependencies found in [requirements.txt][requirements].



## 🕹️ How to use

### 🧰 Download the Brain Encoding Response Generator


BERG is hosted as a public [AWS S3 bucket](https://brain-encoding-response-generator.s3.us-west-2.amazonaws.com/index.html) via the AWS Open Data Program. You do **not need an AWS account** to browse or download the data.

<font color='red'><b>IMPORTANT:</b></font> By downloading the data you agree to BERG's [Terms and Conditions](https://brain-encoding-response-generator.readthedocs.io/en/latest/about/terms_and_conditions.html).

To download the full BERG dataset into a local folder named `brain-encoding-response-generator`, use the AWS CLI:

```bash
aws s3 sync --no-sign-request s3://brain-encoding-response-generator ./brain-encoding-response-generator
```

You can also download specific subfolders, for example:

```bash
aws s3 sync --no-sign-request s3://brain-encoding-response-generator/encoding_models/modality-fmri ./modality-fmri
```

Or, you can also downlaod specific files:

```bash
aws s3 cp --no-sign-request s3://brain-encoding-response-generator/encoding_models/../model_weights.npy ./modality-fmri
```

For detailed instructions and folder structure, see the [documentation](https://brain-encoding-response-generator.readthedocs.io/en/latest/data_storage.html#).


### 🧠 Available encoding models

The following table shows the encoding models currently available in BERG. For more details on these models, please refer to the [documentation][model_cards].

| Model ID | Training dataset | Species | Stimuli |
|----------|------------------|---------|---------|
| [fmri-nsd_fsaverage-vit_b_32][fmri-nsd_fsaverage-vit_b_32] | [NSD (surface space)][allen] | Human | Images |
| [fmri-nsd-fwrf][fmri-nsd-fwrf] | [NSD (volume space)][allen] | Human | Images |
| [eeg-things_eeg_2-vit_b_32][eeg-things_eeg_2-vit_b_32] | [THINGS EEG2][THINGS EEG2] | Human | Images |
 


### ✨ BERG functions

#### 🔹 Initialize the BERG object

To use `BERG`'s functions, you first need to import `BERG` and create a `berg_object`.

```python
from berg import BERG

# Initialize BERG with the path to the toolkit
berg = BERG(berg_dir="path/to/brain-encoding-response-generator")
```
#### 🔹 Generate in silico neural responses to stimuli

Step 1: Load an encoding model of your choice using the `get_encoding_model` function.

```python
# Load an example fMRI encoding model
fmri_model = berg.get_encoding_model("fmri-nsd-fwrf", 
                                     subject=1,
                                     selection={"roi": "V1"},
                                     device="cpu")

# Load an example EEG encoding model
eeg_model = berg.get_encoding_model("eeg-things_eeg_2-vit_b_32",
                                    subject=1,
                                    device="auto")

```

Step 2: Generate in silico neural responses to stimuli using the `encode` function.

```python
# Encode fMRI responses to images with metadata
insilico_fmri, insilico_fmri_metadata = berg.encode(fmri_model,
                                                   images,
                                                   return_metadata=True)  # if needed

# Encode EEG responses to images without metadata
insilico_eeg = berg.encode(eeg_model,
                          images)
```

For more detailed information on how to use these functions and which parameters are available, please refer to the **Tutorials** below ⬇️.


### 💻 Tutorials

We provide several tutorials to help you get started with BERG:

**Using BERG:**
- [Quickstart Tutorial](https://drive.google.com/file/d/1JS4um1eS4Ml983lUNQgEw4544_Lc5Qn0/view?usp=drive_link) - Quick Guide on how to generate in silico neural responses
- [fMRI Tutorial](https://drive.google.com/file/d/1w4opmM9h8Oe1NWlwIDuLuDIGuIXj9UaV/view?usp=drive_link) - Learn how to generate in silico fMRI responses.
- [EEG Tutorial](https://drive.google.com/file/d/1uF5nr1pyg0_my3gULj3w5y0nuq5gZjhL/view?usp=drive_link) - Learn how to generate in silico EEG responses.
- [Adding New Models](https://drive.google.com/file/d/1nBxEiJATzJdWwfzRPmyai2G76HkeBhAU/view?usp=drive_link) - Guide on how to implement and contribute your own encoding models to BERG.

**Example Application - Relational Neural Control (RNC):**

We used BERG to develop [Relational Neural Control](https://github.com/gifale95/RNC), a neural control algorithm to move from an atomistic understanding of visual cortical areas (i.e., What does each area represent?) to a network-level understanding (i.e., What is the relationship between representations in different areas?). Through RNC we discovered controlling images that align or disentangle responses across areas, thus indicating their shared or unique representational content. Closing the empirical cycle, we validated the in silico discoveries on in vivo fMRI responses from independent subjects. Following are RNC tutorials based on in silico fMRI responses generated through BERG:

- [Univariate RNC Tutorial](https://colab.research.google.com/drive/1QpMSlvKZMLrDNeESdch6AlQ3qKsM1isO?usp=sharing) 
- [Multivariate RNC Tutorial](https://colab.research.google.com/drive/1bEKCzkjNfM-jzxRj-JX2zxB17XBouw23?usp=sharing) 



## 📦 BERG creation code

The folder [`../BERG/berg_creation_code/`][berg_creation_code] contains the code used to train the encoding models in BERG, divided in the following sub-folders:

* **[`../01_prepare_data/`][prepare_data]:** prepare the neural responses in the right format for encoding model training.
* **[`../02_train_encoding_models/`][train_encoding]:** train the encoding models, and save their weights.
* **[`../03_test_encoding_models/`][test_encoding]:** test the encoding models (i.e., compute and plot their encoding accuracy).



## ❗ Issues

If you come across problems with this Python package, please submit an issue!



## 📜 Citation

If you use the Brain Encoding Response Generator, please cite:

> *Gifford AT, Bersch D, Roig G, Cichy RM. 2025. The Brain Encoding Response Generator. In preparation. https://github.com/gifale95/BERG*


[documentation]: https://brain-encoding-response-generator.readthedocs.io/en/latest/
[berg_structure]: https://brain-encoding-response-generator.readthedocs.io/en/latest/data_storage.html#
[model_cards]: https://brain-encoding-response-generator.readthedocs.io/en/latest/models/overview.html
[berg_contribute]: https://brain-encoding-response-generator.readthedocs.io/en/latest/contribution.html
[nsd]: https://naturalscenesdataset.org/
[allen]: https://www.nature.com/articles/s41593-021-00962-x
[requirements]: https://github.com/gifale95/BERG/blob/main/requirements.txt
[rclone]: https://rclone.org/
[guide]: https://noisyneuron.github.io/nyu-hpc/transfer.html


[get_encoding_model]: https://github.com/gifale95/BERG/blob/main/berg/berg.py#L207
[encode]: https://github.com/gifale95/BERG/blob/main/berg/berg.py#L321
[load_insilico_neural_responses]: https://github.com/gifale95/BERG/blob/main/berg/berg.py#L551


[fmri-nsd_fsaverage-vit_b_32]: https://brain-encoding-response-generator.readthedocs.io/en/latest/models/model_cards/fmri-nsd_fsaverage-vit_b_32.html
[fmri-nsd-fwrf]: https://brain-encoding-response-generator.readthedocs.io/en/latest/models/model_cards/fmri-nsd-fwrf.html
[eeg-things_eeg_2-vit_b_32]: https://brain-encoding-response-generator.readthedocs.io/en/latest/models/model_cards/eeg-things_eeg_2-vit_b_32.html
[THINGS EEG2]: https://doi.org/10.1016/j.neuroimage.2022.119754


[fmri_tutorial_colab]: https://colab.research.google.com/drive/1W9Sroz2Y0eTYfyhVrAJwe50GGHHAGBdE?usp=drive_link
[eeg_tutorial_colab]: https://colab.research.google.com/drive/10NSRBrJ390vuaPyRWq5fDBIA4NNAUlTk?usp=drive_link
[fmri_tutorial_jupyter]: https://github.com/gifale95/BERG/blob/main/tutorials/berg_fmri_tutorial.ipynb
[eeg_tutorial_jupyter]: https://github.com/gifale95/BERG/blob/main/tutorials/berg_eeg_tutorial.ipynb
[uni_rnc_colab]: https://colab.research.google.com/drive/1QpMSlvKZMLrDNeESdch6AlQ3qKsM1isO?usp=sharing
[multi_rnc_colab]: https://colab.research.google.com/drive/1bEKCzkjNfM-jzxRj-JX2zxB17XBouw23?usp=sharing
[uni_rnc_jupyter]: https://github.com/gifale95/RNC/blob/main/tutorials/univariate_rnc_tutorial.ipynb
[multi_rnc_jupyter]: https://github.com/gifale95/RNC/blob/main/tutorials/multivariate_rnc_tutorial.ipynb
[berg_creation_code]: https://github.com/gifale95/BERG/tree/main/berg_creation_code/
[prepare_data]: https://github.com/gifale95/BERG/tree/main/berg_creation_code/01_prepare_data
[train_encoding]: https://github.com/gifale95/BERG/tree/main/berg_creation_code/02_train_encoding_models
[test_encoding]: https://github.com/gifale95/BERG/tree/main/berg_creation_code/03_test_encoding_models
[metadata]: https://github.com/gifale95/BERG/tree/main/berg_creation_code/03_create_metadata
[synthesize]: https://github.com/gifale95/BERG/tree/main/berg_creation_code/04_synthesize_neural_responses
