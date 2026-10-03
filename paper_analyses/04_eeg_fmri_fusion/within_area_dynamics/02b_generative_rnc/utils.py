def load_encoding_models(args, idx_v, t_min, t_max, n_times, device):
    """Load the encoding EEG and t-fMRI encoding models for the time window
    of interest.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    idx_v : dict
        Dictionary containing the indices of the vertices.
    t_min : int
        The starting time point.
    t_max : int
        The ending time point.
    n_times : int
        The number of time points.
    device : str
        If 'cpu' keep the model on CPU, if 'cuda' send the model to GPU.

    Returns
    -------
    model_eeg : list
        The trained EEG encoding models.
    model_tfmri : list
        The trained t-fMRI encoding models.

    """

    import os
    import numpy as np
    from berg import BERG
    from copy import deepcopy
    from sklearn.linear_model import LinearRegression

    ### Load the EEG encoding models ###
    timepoints = np.zeros(n_times, dtype=np.int32)
    timepoints[t_min:t_max+1] = 1
    berg_object = BERG(args.berg_dir)
    model_eeg = []
    for esub in args.eeg_subjects:
        model_eeg.append(deepcopy(berg_object.get_encoding_model(
            model_id='eeg-things_eeg_2-vit_b_32',
            subject=esub,
            selection={'timepoints': timepoints},
            device=device
            )))

    ### Load the t-fMRI encoding models ###
    model_tfmri = []
    for fsub in args.fmri_subjects:
        model_tfmri_sub = []
        for t, tid in enumerate(range(t_min, t_max+1)):
            for h, hemi in enumerate(args.hemispheres):
                file_name = (f'weights_fmri_sub-{fsub:02d}_'
                    f'hemi-{hemi}_eeg_train_trials-all_eeg_time-{tid:03d}.npy')
                reg_param = np.load(os.path.join(args.berg_dir,
                    'eeg_fmri_fusion', 'encoding_fusion_weights', file_name),
                    allow_pickle=True).item()
                if h == 0:
                    coef_ = reg_param['coef_'][idx_v[(fsub,hemi)]]
                    intercept_ = reg_param['intercept_'][idx_v[(fsub,hemi)]]
                    n_features_in_ = reg_param['n_features_in_']
                else:
                    coef_ = np.append(coef_,
                        reg_param['coef_'][idx_v[(fsub,hemi)]], 0)
                    intercept_ = np.append(intercept_,
                        reg_param['intercept_'][idx_v[(fsub,hemi)]])
                del reg_param
            reg = LinearRegression()
            reg.coef_ = coef_
            reg.intercept_ = intercept_
            reg.n_features_in_ = n_features_in_
            model_tfmri_sub.append(deepcopy(reg))
            del reg, coef_, intercept_
        model_tfmri.append(deepcopy(model_tfmri_sub))
        del model_tfmri_sub

    ### Output ###
    return model_eeg, model_tfmri


def load_image_generator(args, device):
    """Load the image generator.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    device : str
        If 'cpu' keep the model on CPU, if 'cuda' send the model to GPU.

    Returns
    -------
    image_generator : object
        Image generator model.

    """

    import os
    import torch
    import torch_nets # https://github.com/willwx/XDream/tree/master/xdream/net_utils/torch_nets
    # from diffusers import ConsistencyModelPipeline

    ### DeePSiM (GAN) ###
    if args.image_generator_name == 'DeePSiM':
        # (Dosovitskiy & Bronx, 2016) https://doi.org/10.48550/arXiv.1602.02644
        # (Xiao & Kreiman, 2020) https://doi.org/10.1371/journal.pcbi.1007973
        # (Ponce et al., 2019) https://doi.org/10.1016/j.cell.2019.04.005
        # ------------------------------
        # When using the 'fc6' version of DeePSiM we sometimes get a patch with
        # stereotypical shape in the center right of the synthesized images.
        # This is a known problem, that has been pointed out in
        # (Ponce et al., 2019):
        # """Some images synthesized by the network contained a patch with a
        # stereotypical shape that occurred in the center right of the image
        # (e.g., second and third image in Figure 4B, and "late synthetic" in
        # Figure 5C and 5D). This was identified as an artifact of the network
        # commonly known as "mode collapse" and it appeared in the same position
        # in a variety of contexts, including a subset of simulated evolutions.
        # This artifact was easily identifiable and it did not affect our
        # interpretations. In the future, more modern GNN training methods
        # should avoid this problem (personal correspondence with Alexey
        # Dosovitskiy)."""
        # ------------------------------
        # Instantiate the 'DeePSiM' image generator model
        image_generator = torch_nets.load_net('deepsim-fc7')
        # Load the generator model weights
        # The generator trained weights can be downloaded from:
        # https://drive.google.com/drive/folders/1bDWEe1Awu-8HaIGzAOaglqxxgzpBBvle
        weight_dir = os.path.join(args.berg_dir, 'eeg_fmri_fusion',
            'within_area_dynamics', 'generative_univariate_rnc',
            'generator_weights', 'deepsim', 'fc7.pt')
        generator_weights = torch.load(weight_dir, map_location='cpu')
        # Load the trained weights into the image generator
        image_generator.load_state_dict(generator_weights)
        # Set "requires_grad" to False, and turn on evaluation mode
        for param in image_generator.parameters():
            param.requires_grad = False
        image_generator.eval()

    ### cd_imagenet64_l2 (diffusion model) ###
    # elif args.image_generator_name == 'cd_imagenet64_l2':
    #     # https://huggingface.co/openai/diffusers-cd_imagenet64_l2
    #     model = 'openai/diffusers-cd_imagenet64_l2'
    #     #image_generator = ConsistencyModelPipeline.from_pretrained(
    #     #	model).to(device)
    #     # Speed up inference:
    #     # https://huggingface.co/docs/diffusers/v0.32.2/optimization/fp16
    #     image_generator = ConsistencyModelPipeline.from_pretrained(model,
    #         torch_dtype=torch.float16, use_safetensors=True).to(device)
    #     # Speed up inference:
    #     # https://huggingface.co/docs/diffusers/v0.32.2/optimization/fp16
    #     # https://huggingface.co/docs/diffusers/v0.32.2/optimization/xformers
    #     #image_generator.enable_xformers_memory_efficient_attention()
    #     # Reduce memory usage
    #     image_generator.enable_sequential_cpu_offload()
    #     image_generator.enable_xformers_memory_efficient_attention()

    ### Output ###
    return image_generator


def generate_tfmri(args, model_eeg, model_tfmri, images, berg):
    """Use the EEG encoding models to generate in silico fMRI responses for the
    synthesized images, and then use the t-fMRI encoding models to generate
    t-fMRI responses from the in silico EEG. The t-fMRI responses are then
    averaged across vertices and time points belonging to the same time window.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    model_eeg : list
        The trained EEG encoding models.
    model_tfmri : list
        The trained t-fMRI encoding models.
    images : int
        Synthesized images. Must be a 4-D numpy array of shape
        (Batch size x 3 RGB Channels x Width x Height) consisting of integer
        values in the range [0, 255]. Furthermore, the images must be of square
        size (i.e., equal width and height).
    berg : BERG
        The BERG object.

    Returns
    -------
    tfmri : float
        t-fMRI responses for the generated images.

    """

    import numpy as np
    from berg import BERG
    from sklearn.linear_model import LinearRegression

    ### Generate the in silico EEG responses to images ###
    # Loop across EEG subjects
    for es, esub in enumerate(args.eeg_subjects):
        # Predict the in silico EEG responses, average them across repeats, and
        # append them across subjects across the channels dimension
        if es == 0:
            eeg = np.mean(berg.encode(
                model_eeg[es], images, show_progress=False), 1)
        else:
            eeg = np.append(eeg, np.mean(berg.encode(
                model_eeg[es], images, show_progress=False), 1), 1)

    ### Generate the t-fMRI responses ###
    # Loop across fMRI subjects
    tfmri = []
    for fs, fsub in enumerate(args.fmri_subjects):
        # Loop across time points
        for t in range(len(model_tfmri[fs])):
            # Generate the t-fMRI responses
            if t == 0:
                tfmri_times = model_tfmri[fs][t].predict(eeg[:,:,t])
            else:
                tfmri_times = np.append(tfmri_times,
                    model_tfmri[fs][t].predict(eeg[:,:,t]), 1)
        # Average the t-fMRI responses across vertices and time points from the
        # same time window, and store the responses
        tfmri.append(np.mean(tfmri_times, 1))
        del tfmri_times
    tfmri = np.array(tfmri)

    ### Output ###
    return tfmri


def score_select(args, tfmri_tw_1, tfmri_tw_2, image_codes, images,
    baseline_tw_1, baseline_tw_2, margin_tw_1, margin_tw_2):
    """Score and rank the generated images based on their neural control
    magnitude on the t-fMRI responses and their complexity.

    Parameters
    ----------
    args : Namespace
        Input arguments.
    tfmri_tw_1 : float
        t-fMRI responses for the generated images (first time window).
    tfmri_tw_2 : float
        t-fMRI responses for the generated images (second time window).
    image_codes : float
        Image codes for the generated images.
    images : int
        Synthesized images. Must be a 4-D numpy array of shape
        (Batch size x 3 RGB Channels x Width x Height) consisting of integer
        values in the range [0, 255]. Furthermore, the images must be of square
        size (i.e., equal width and height).
    baseline_tw_1 : float
        t-fMRI baseline score (first time window).
    baseline_tw_2 : float
        t-fMRI baseline score (second time window).
    margin_tw_1 : float
        Baseline margin for the first time window.
    margin_tw_2 : float
        Baseline margin for the second time window.

    Returns
    -------
    scores_train : float
        Generated image scores for the train subjects.
    scores_test : float
        Generated image scores for the test subject.
    neural_control_scores_train : float
        Neural control scores for the train subjects.
    neural_control_scores_test : float
        Neural control scores for the test subject.
    baseline_penalty_train : float
        Baseline penalty scores for the train subjects.
    baseline_penalty_test : float
        Baseline penalty scores for the test subject.
    images_complexity : int
        Image complexity.
    image_codes : float
        Selected image codes for the best generated images.
    tfmri_tw_1 : float
        t-fMRI responses for the generated images (first time window).
    tfmri_tw_2 : float
        t-fMRI responses for the generated images (second time window).
    images : PyTorch tensor
        Generated images.

    """

    import numpy as np
    import io
    from PIL import Image
    from copy import copy

    ### Define the train/test partitions ###
    if args.cv == 0:
        tfmri_tw_1_train = np.mean(tfmri_tw_1, 0)
        tfmri_tw_1_test = np.mean(tfmri_tw_1, 0)
        tfmri_tw_2_train = np.mean(tfmri_tw_2, 0)
        tfmri_tw_2_test = np.mean(tfmri_tw_2, 0)
    elif args.cv == 1:
        tfmri_tw_1_train = np.mean(np.delete(
            tfmri_tw_1, args.cv_subject-1, 0), 0)
        tfmri_tw_1_test = tfmri_tw_1[args.cv_subject-1]
        tfmri_tw_2_train = np.mean(np.delete(
            tfmri_tw_2, args.cv_subject-1, 0), 0)
        tfmri_tw_2_test = tfmri_tw_2[args.cv_subject-1]

    ### Compute the neural control scores ###
    # [high_1_high_2] --> High time window 1 - High time window 2
    # [high_1_low_2] --> High time window 1 - Low time window 2
    # [low_1_high_2] --> Low time window 1 - High time window 2
    # [low_1_low_2] --> Low time window 1 - Low time window 2
    if args.control_type == 'high_1_high_2':
        neural_control_scores_train = tfmri_tw_1_train + tfmri_tw_2_train
        neural_control_scores_test = tfmri_tw_1_test + tfmri_tw_2_test
    elif args.control_type == 'high_1_low_2':
        neural_control_scores_train = tfmri_tw_1_train - tfmri_tw_2_train
        neural_control_scores_test = tfmri_tw_1_test - tfmri_tw_2_test
    elif args.control_type == 'low_1_high_2':
        neural_control_scores_train = tfmri_tw_2_train - tfmri_tw_1_train
        neural_control_scores_test = tfmri_tw_2_test - tfmri_tw_1_test
    elif args.control_type == 'low_1_low_2':
        neural_control_scores_train = tfmri_tw_1_train + tfmri_tw_2_train
        neural_control_scores_test = tfmri_tw_1_test + tfmri_tw_2_test

    ### Compute a penalty based on the fMRI baseline ###
    baseline_penalty_train = np.zeros((args.n_image_codes), dtype=int)
    baseline_penalty_test = np.zeros((args.n_image_codes), dtype=int)
    if args.control_type == 'high_1_high_2':
        idx_bad_train_tw_1 = tfmri_tw_1_train < (baseline_tw_1 + margin_tw_1)
        idx_bad_train_tw_2 = tfmri_tw_2_train < (baseline_tw_2 + margin_tw_2)
        idx_bad_test_tw_1 = tfmri_tw_1_test < (baseline_tw_1 + margin_tw_1)
        idx_bad_test_tw_2 = tfmri_tw_2_test < (baseline_tw_2 + margin_tw_2)
    elif args.control_type == 'high_1_low_2':
        idx_bad_train_tw_1 = tfmri_tw_1_train < (baseline_tw_1 + margin_tw_1)
        idx_bad_train_tw_2 = tfmri_tw_2_train > (baseline_tw_2 - margin_tw_2)
        idx_bad_test_tw_1 = tfmri_tw_1_test < (baseline_tw_1 + margin_tw_1)
        idx_bad_test_tw_2 = tfmri_tw_2_test > (baseline_tw_2 - margin_tw_2)
    elif args.control_type == 'low_1_high_2':
        idx_bad_train_tw_1 = tfmri_tw_1_train > (baseline_tw_1 - margin_tw_1)
        idx_bad_train_tw_2 = tfmri_tw_2_train < (baseline_tw_2 + margin_tw_2)
        idx_bad_test_tw_1 = tfmri_tw_1_test > (baseline_tw_1 - margin_tw_1)
        idx_bad_test_tw_2 = tfmri_tw_2_test < (baseline_tw_2 + margin_tw_2)
    elif args.control_type == 'low_1_low_2':
        idx_bad_train_tw_1 = tfmri_tw_1_train > (baseline_tw_1 - margin_tw_1)
        idx_bad_train_tw_2 = tfmri_tw_2_train > (baseline_tw_2 - margin_tw_2)
        idx_bad_test_tw_1 = tfmri_tw_1_test > (baseline_tw_1 - margin_tw_1)
        idx_bad_test_tw_2 = tfmri_tw_2_test > (baseline_tw_2 - margin_tw_2)
    idx_bad_train = np.where(idx_bad_train_tw_1 + idx_bad_train_tw_2)[0]
    baseline_penalty_train[idx_bad_train] = 1e+10
    idx_bad_test = np.where(idx_bad_test_tw_1 + idx_bad_test_tw_2)[0]
    baseline_penalty_test[idx_bad_test] = 1e+10

    ### Compute the images complexity ###
    images_complexity = np.zeros((args.n_image_codes), dtype=np.int32)
    # Image complexity is estimated through compression: given that all
    # images have equal pixel sizes, the more an image can be compressed
    # (i.e., the lower the bytes of the compressed image), the less complex
    # it is (i.e., the less information it contains).
    for i, img in enumerate(images):
        img = np.swapaxes(np.swapaxes(copy(img), 0, 1), 1, 2)
        img_pil = Image.fromarray(img, 'RGB')
        output = io.BytesIO()
        if args.img_complexity_measure == 'png':
            img_pil.save(output, format='PNG')
        elif args.img_complexity_measure == 'jpg':
            img_pil.save(output, format='JPEG')
        images_complexity[i] = output.tell()

    ### Aggregate neural control and baseline penalty scores ###
    if args.control_type == 'low_1_low_2':
        neural_scores_train = neural_control_scores_train + \
            baseline_penalty_train
        neural_scores_test = neural_control_scores_test + baseline_penalty_test
    else:
        neural_scores_train = neural_control_scores_train - \
            baseline_penalty_train
        neural_scores_test = neural_control_scores_test - baseline_penalty_test
    # Only take neural control scores into account when they are below the
    # baseline + margin threshold: in this way, once the images well control
    # neural responses, their successive optimizations are only based on images
    # complexity.
    neural_scores_train[baseline_penalty_train==0] = 0
    neural_scores_test[baseline_penalty_test==0] = 0

    ### Image complexity scores ###
    image_scores = - copy(images_complexity) # we want to reduce complexity
    # Once the neural control scores pass the threshold, only use the images
    # complexity to rank the images. However, these image scores are ignored
    # prior to passing the threshold.
    image_scores[baseline_penalty_train!=0] = 0

    ### Aggregate neural control and complexity scores ###
    # The loss will only care about the neural control scores until these pass
    # the threshold (baseline + margin). After this threshold is reached, the
    # loss will only care about images complexity. This is because we
    # necessarily want images which well control neural responses but, once
    # we have such images, we want them to be as simple (to isolate controlling
    # visual features) as possible.
    if args.control_type == 'low_1_low_2':
        scores_train = neural_scores_train - image_scores
        scores_test = neural_scores_test - image_scores
    else:
        scores_train = neural_scores_train + image_scores
        scores_test = neural_scores_test + image_scores

    ### Rank the scores from best to worst ###
    if args.control_type == 'low_1_low_2':
        rank = np.argsort(scores_train)
    else:
        rank = np.argsort(scores_train)[::-1]

    ### Output ###
    return scores_train[rank], scores_test[rank], \
        neural_control_scores_train[rank], neural_control_scores_test[rank], \
        baseline_penalty_train[rank], baseline_penalty_test[rank], \
        images_complexity[rank], image_codes[rank], tfmri_tw_1[:,rank], \
        tfmri_tw_2[:,rank], images[rank]


def optimize_image_codes(args, image_codes, scores, random_generator):
    """Genetic algorithm optimization from (Ponce et al., 2019):

    "The algorithm began with an initial population of 40 image codes
    ('individuals'), each consisting of a 4096-dimensional vector ('genes') and
    associated with a synthesized image. Images were presented to the subject,
    and the corresponding spiking response was used to calculate the 'fitness'
    of the image codes by transforming the firing rate into a Z-score within the
    generation, scaling it by a selectiveness factor of 0.5, and passing it
    through a softmax function to become a probability. The 10 highest-fitness
    individuals (25%) were passed on to the next generation without
    recombination or mutation. Another 30 children image codes (75%) were
    produced from recombinations between two parent image codes from the current
    generation, with the probability for each image code to be a parent being
    its fitness. The two parents contributed unevenly (75%:25%) to any one
    child. Individual children genes had a 0.25 probability of being mutated,
    with mutations drawn from a 0-centered Gaussian with standard deviation
    0.75."

    Parameters
    ----------
    args : Namespace
        Input arguments.
    image_codes : float
        Selected image codes for the best synthesized images.
    scores : float
        Generated images scores.
    random_generator : object
        Random numbers generator.

    Returns
    -------
    image_codes_new : float
        Optimized image codes.

    """

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from scipy.special import softmax
    from copy import copy

    ### Standardize the scores, scale them by 0.5, and apply softmax ###
    # https://towardsdatascience.com/transformer-networks-a-mathematical-explanation-why-scaling-the-dot-products-leads-to-more-stable-414f87391500
    scaler = StandardScaler()
    scores = scaler.fit_transform(
        np.reshape(scores, (len(scores), -1))).squeeze() * .5
    if args.control_type == 'low_1_low_2':
        scores = - scores
    prob_scores = softmax(scores)

    ### Keep X% of the best image codes untouched ###
    n_kept_img_codes = int(len(image_codes) * args.frac_kept_image_codes)
    n_new_image_codes = len(image_codes) - n_kept_img_codes

    ### Vectorize the image codes ###
    new_image_codes_shape = list(image_codes.shape)
    new_image_codes_shape[0] = n_new_image_codes
    new_image_codes_shape = tuple(new_image_codes_shape)
    image_codes = np.reshape(image_codes, (len(image_codes),-1))

    ### Recombine and mutate the remaining image codes ###
    image_codes_new = []
    for i in range(n_new_image_codes):
        # Select the two parent image codes based on their probability scores
        parents = random_generator.choice(len(image_codes), size=2,
            replace=False, p=prob_scores)
        # Create the child image code
        random_idx = random_generator.choice(image_codes.shape[1],
            size=int(image_codes.shape[1]*args.heritability),
            replace=False)
        child_image_code = copy(image_codes[parents[0]])
        child_image_code[random_idx] = copy(image_codes[parents[1],random_idx])
        # Mutate the child image code
        mutations_idx = np.where(random_generator.choice(np.asarray((0,1)),
            size=len(child_image_code),
            p=np.asarray((1-args.mutation_prob,args.mutation_prob))))[0]
        child_image_code[mutations_idx] += random_generator.normal(loc=0,
            scale=.75, size=len(mutations_idx))
        image_codes_new.append(copy(child_image_code))
    image_codes_new = np.asarray(image_codes_new)

    ### Reshape the image codes to original format ###
    image_codes_new = np.reshape(image_codes_new, new_image_codes_shape)

    ### Output ###
    return image_codes_new
