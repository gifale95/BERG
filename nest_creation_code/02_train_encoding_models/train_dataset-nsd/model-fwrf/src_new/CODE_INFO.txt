ORIGINAL CODE

This code is an adapted version of the code from the paper:
Allen, E.J., St-Yves, G., Wu, Y., Breedlove, J.L., Prince, J.S., Dowdle, L.T., Nau, M., Caron, B., Pestilli, F., Charest, I. and Hutchinson, J.B., 2022. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature neuroscience, 25(1), pp.116-126.


The original code, written by Ghislain St-Yves, can be found here:
https://github.com/styvesg/nsd_gnet8x



EDITS OF ORIGINAL CODE

1. The classes in "torch_gnet.py" are modified such that the Encoder returns the pre-filter feature maps too, so that the FWRF models can use these features too to predict brain data.

2. TensorBoard logs are added to the training loop in the "torch_joint_training_unpacked_sequences.learn_params_" function.

3. Change the "load_nsd.ordering_split", "torch_joint_training_unpacked_sequences.learn_params_" and "torch_joint_training_unpacked_sequences.validation_" functions so that the models are trained on the subject-unique images, tested on the 515 shared images with 3 repeats across all subjects, and validated on the remaining shared images.
