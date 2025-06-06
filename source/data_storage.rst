====================
How to Access BERG
====================

BERG is stored in a public `Amazon S3 bucket <https://brain-encoding-response-generator.s3.us-west-2.amazonaws.com/index.html>`_ made available through the **AWS Open Data Program**. You do **not need an AWS account** to browse or download the data. By downloading the data you agree to BERG's :doc:`Terms and Conditions </about/terms_and_conditions>`.

To access the bucket, use the following information:

- **Bucket name:** brain-encoding-response-generator
- **AWS region:** us-west-2
- **ARN:** arn:aws:s3:::brain-encoding-response-generator

.. note::
   The *brain-encoding-response-generator* bucket contains many GBs of data. Depending on your needs, you may choose to download only specific folders. This documentation provides a detailed description of BERG's content to help you decide what to download.

Recommended Download Method
---------------------------

The most efficient way to download BERG is using the **AWS Command Line Interface (CLI)**.

**Step 1: Install AWS CLI**

You can follow the installation instructions here: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html

**Step 2: Browse or Download Data**

To list all folders in the bucket:
::

    aws s3 ls --no-sign-request s3://brain-encoding-response-generator/

To download the entire dataset into a local folder named `brain-encoding-response-generator`:
::

    aws s3 sync --no-sign-request s3://brain-encoding-response-generator ./brain-encoding-response-generator

To download only a specific subfolder (e.g., models trained on fMRI data):
::

    aws s3 sync --no-sign-request s3://brain-encoding-response-generator/encoding_models/modality-fmri ./modality-fmri

You can also use `--dryrun` to preview what would be downloaded:
::

    aws s3 sync --no-sign-request --dryrun s3://brain-encoding-response-generator ./brain-encoding-response-generator

Finally, you can also download individual files with:
::

    aws s3 cp --no-sign-request s3://brain-encoding-response-generator/encoding_models/../model_weights.npy ./modality-fmri

**Optional Web Access**

You can access files directly from your browser using:
::

    https://brain-encoding-response-generator.s3.us-west-2.amazonaws.com/index.html


============================
BERG Dataset Structure
============================

Overview of BERG Folder Organization
------------------------------------

BERG is organized in a structured hierarchy designed to make it easy to locate specific encoding models by their neural data recording modality, training dataset, and model type.

The main folder structure follows this pattern:

.. code-block:: text

    brain-encoding-response-generator/
    ├── encoding_models/
    │   ├── modality-{modality}/
    │   │   ├── train_dataset-{dataset}/
    │   │   │   └── model-{model}/
    │   │   │       ├── encoding_models_accuracy/
    │   │   │       ├── encoding_models_weights/
    │   │   │       └── metadata/
    └── berg_tutorials/

Detailed Structure of Encoding Models
-------------------------------------

The ``encoding_models`` directory contains all trained models organized hierarchically by:

1. **modality:** The neural recording recording modality on which the encoding model was trained (e.g., ``fmri``, ``eeg``).
2. **train_dataset:** The neural dataset on which the encoding model was trained (e.g., ``nsd``, ``things_eeg_2``).
3. **model:** The type of encoding model used (e.g., ``fwrf``, ``vit_b_32``).



Contents of Model Directories
----------------------------

Each model directory contains three subdirectories:

**encoding_models_accuracy/**

* This directory contains plots of the encoding models' prediction accuracy.

**encoding_models_weights/**

* This directory contains the trained model weights used to generate the in silico neural responses.

**metadata/**

* This directory contains important metadata relative to the encoding models and to the neural data used to train them.

The BERG Python package automatically handles access to these files based on your requested parameters, making it easy to use without managing these paths directly.
