import torch
from transformers import ASTModel, ASTFeatureExtractor
from decord import AudioReader, cpu
from tqdm import tqdm
from contextlib import contextmanager

from berg.models.fmri.vibe_utils.config import VIBEConfig


AST_CHUNK_LENGTH = 10.24


class AudioFeatureExtractor:
    def __init__(self, config: VIBEConfig, device: str = "cpu", low_mem_usage: bool = True):
        super().__init__()
        self.config = config
        self.model = None
        self.device = device
        self.processor = None
        self.model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.low_mem_usage = low_mem_usage

    def load_model(self):
        self.processor = ASTFeatureExtractor.from_pretrained(self.model_name)
        self.model = ASTModel.from_pretrained(self.model_name,
                                              attn_implementation="sdpa").to(self.device).eval()

    def cleanup(self):
        if self.model is not None:
            self.model.cpu()
            self.model = None
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def _model_session(self):
        if self.model is None:
            self.load_model()
        try:
            yield
        finally:
            if self.low_mem_usage:
                self.cleanup()

    def extract_features(self, video_path):
        with self._model_session():
            full_waveform = extract_audio_for_ast(video_path)
            all_features = []  # feature for each TR
            duration = full_waveform.shape[0] / 16000.0
            chunks, pooling_meta = split_movie_into_chunks(
                duration, self.config.tr)

            all_features = []

            batch_waveforms = []
            batch_meta = []

            for i, (start_sec, end_sec) in tqdm(enumerate(chunks), total=len(chunks), desc=f"Extracting Audio Features", leave=False):
                s_idx = int(start_sec * 16000)
                e_idx = int(end_sec * 16000)
                clip = full_waveform[s_idx:e_idx]

                batch_waveforms.append(clip.numpy())
                batch_meta.append(pooling_meta[i])

                if len(batch_waveforms) == self.config.audio_extractor_batch_size or i == len(chunks) - 1:
                    feats = self.run_batch_inference(batch_waveforms, batch_meta)
                    all_features.extend(feats)

                    # Reset
                    batch_waveforms = []
                    batch_meta = []

            return torch.stack(all_features)

    def run_batch_inference(self, waveforms, pooling_meta):
        inputs = self.processor(
            waveforms,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length"  # Ensures everything is 10.24s
        ).to(self.device)

        with torch.no_grad():
            features_seq = self.model(
                **inputs, output_hidden_states=True).hidden_states[-self.config.audio_extractor_last_layer_index]

        features_content = features_seq[:, 2:, :]

        config = self.model.config
        num_mel = config.num_mel_bins      # 128
        f_stride = config.frequency_stride  # 10
        p_size = config.patch_size         # 16

        # Height (Frequency)
        n_freq_patches = (num_mel - p_size) // f_stride + 1  # 12
        # Width (Time)
        n_tokens = features_content.shape[1]
        n_time_patches = n_tokens // n_freq_patches          # ~101

        # Reshape as (Frequency, Time) first
        features_grid = features_content.view(
            features_content.shape[0],
            n_freq_patches,   # Height first
            n_time_patches,   # Width second
            -1
        )

        # Transpose to (Time, Frequency, Dim)
        # Now we swap axes so Time is the primary sequence dimension
        features_grid = features_grid.transpose(1, 2)

        # Pool across Frequency (dim 2)
        # Now we have one vector per time step, averaged over all frequencies
        time_features = features_grid.mean(dim=2)  # (Batch, Time_Patches, Dim)

        batch_pooled = []

        for i, (rel_start, rel_end, valid_len) in enumerate(pooling_meta):
            # Map time to Time Patch Indices
            start_idx = int((rel_start / AST_CHUNK_LENGTH) * n_time_patches)
            end_idx = int((rel_end / AST_CHUNK_LENGTH) * n_time_patches)

            start_idx = max(0, start_idx)
            end_idx = min(n_time_patches, end_idx)

            if end_idx <= start_idx:
                end_idx = start_idx + 1
            clip_feats = time_features[i, start_idx:end_idx, :]
            pooled = clip_feats.mean(dim=0)
            pooled_std = clip_feats.std(dim=0, unbiased=False)

            concatenated = torch.cat((pooled, pooled_std))
            batch_pooled.append(concatenated)

        return batch_pooled


def extract_audio_for_ast(video_path):
    ar = AudioReader(video_path, ctx=cpu(0), sample_rate=16000, mono=True)
    audio_data = ar[:].asnumpy()
    waveform = torch.from_numpy(audio_data).squeeze()
    return waveform


def split_movie_into_chunks(video_duration, tr):
    """
    Generates time windows. 
    - chunk_length is fixed to 10.24s (AST requirement).
    - stride is fixed to tr.
    - Windows are causal (ending at current TR).
    """
    chunks = []
    pooling_ranges = []
    current_time = tr

    while current_time <= video_duration + tr:  # Ensure we cover the last partial TR
        w_end = min(current_time, video_duration)
        w_start = max(0, w_end - AST_CHUNK_LENGTH)

        chunks.append((w_start, w_end))
        valid_audio_len = w_end - w_start
        rel_end_sec = valid_audio_len
        rel_start_sec = max(0, valid_audio_len - tr)

        pooling_ranges.append((rel_start_sec, rel_end_sec, valid_audio_len))

        current_time += tr

        if w_end >= video_duration:
            break

    return chunks, pooling_ranges
