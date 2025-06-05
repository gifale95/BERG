=========================
Contribute to BERG
=========================

Whether you have developed better encoding models, have models for different neural datasets, want to add support for new modalities, or have suggestions for improvement, we'd love to hear from you!

Ways to Contribute
-----------------

You can contribute to BERG in several ways:

* **New encoding models** with higher prediction accuracies than the currently available.
* **Models trained on different neural datasets** or brain regions.
* **Models trained on new neural recording modalities** (e.g., MEG, ECoG, animal recordings).
* **Models tained for different stimulus types** (e.g., videos, language, audio, multimodal).
* **Improvements to the codebase** or documentation.

Add Your Model to BERG
------------------------

NEST provides a standardized framework for integrating new models. To add your model:

1. **Implement your model class** following NEST's interface.
2. **Create a YAML configuration file** describing your model's parameters and behavior.

For detailed instructions, see the :doc:`Tutorial </tutorials/Adding_New_Models_to_Berg>` on this website!


Code Quality Guidelines
----------------------

When contributing, please follow these quality guidelines:

* Include clear **docstrings** for all public methods.
* Add **type hints** to improve code readability.
* Implement **robust error handling** with informative messages.
* Follow existing **NEST naming conventions**.
* Be thorough with your **YAML configuration**.
* If available, include **performance details**.

How to Submit Your Contribution
------------------------------

To submit your contribution:

1. **Fork** the BERG repository.
2. **Create a branch** from the ``development`` branch.
3. **Add your model** following the tutorial guidelines.
4. **Submit a pull request** with:

  * A clear description of your model.
  * Example code showing how to use your model.
  * Any relevant citations or references.

Contact
-------

If you have questions or would like to discuss your contribution before submitting, please contact us at brain.berg.info@gmail.com. All feedback and help is strongly appreciated!

We look forward to your contributions and are excited to see how the community expands BERG!
