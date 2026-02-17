from berg import BERG
import numpy as np

berg = BERG(berg_dir="/scratch/giffordale95/projects/brain-encoding-response-generator")


metadata = berg.get_model_metadata("fmri-bmd-s3d", subject=1)

model = berg.get_encoding_model("fmri-bmd-s3d", subject=1, device="cpu")

videos = np.random.randint(0, 255, (1, 60, 3, 256, 256))

in_silico_fmri = berg.encode(model, videos)