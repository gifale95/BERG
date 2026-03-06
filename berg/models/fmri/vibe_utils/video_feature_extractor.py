import torch
from transformers import AutoModel
from decord import VideoReader, cpu
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from berg.models.fmri.vibe_utils.config import VIBEConfig

__all__ = ["VideoFeatureExtractor"]


class VideoLoader:
    """
    Drop-in replacement for your class:
      - Same __init__(video_path, chunks, chunk_of_interests)
      - Same __len__()
      - Same get_batch(start_idx, batch_size) -> (batch_tensor, meta, actual_bs)

    Functionality matches:
      - Output tensor: (B, 64, C, H, W) on GPU
      - dtype: bfloat16
      - values: (uint8/255) then (x - MEAN)/STD
      - padding: if a row has padding_mask True, those frames are filled with 0.0 (exactly like your code)
    """

    def __init__(self, 
                 video_path, 
                 chunks, 
                 chunk_of_interests,
                 config: VIBEConfig,
                 device = "cuda",
                 decode_ctx=cpu(0)):
        
        width=256
        height=256
        samples_per_clip=64
        self.config = config
        self.device = device
        self.chunks = chunks
        self.coi = chunk_of_interests
        self.samples_per_clip = int(samples_per_clip)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        # Keep the VideoReader alive; do NOT load full video anywhere.
        self.vr = VideoReader(video_path, ctx=decode_ctx, width=width, height=height)
        self.fps = float(self.vr.get_avg_fps())
        self.total_frames = int(len(self.vr))

        # Precompute ALL indices + padding masks ON CPU (small)
        all_indices = np.empty((len(chunks), self.samples_per_clip), dtype=np.int64)
        padding_masks = np.empty((len(chunks), self.samples_per_clip), dtype=np.bool_)

        for i, (c_start, c_end) in enumerate(chunks):
            frame_start = int(c_start * self.fps)
            frame_end = int(c_end * self.fps)
            frame_end = min(self.total_frames, frame_end)

            indices = np.arange(frame_start, frame_end, dtype=np.int64)

            if indices.size == 0:
                all_indices[i] = 0
                padding_masks[i] = True
                continue

            # exactly like your approach: pick 64 evenly spaced from [0..len-1]
            sel = np.linspace(0, indices.size - 1, self.samples_per_clip).astype(np.int64)
            final_frame_indices = indices[sel]

            # Your old code computed "is_padding = final_frame_indices < 0" (never happens here),
            # but we keep the exact same logic structure anyway.
            is_padding = final_frame_indices < 0
            clamped = np.maximum(0, final_frame_indices)

            all_indices[i] = clamped
            padding_masks[i] = is_padding

        self.index_matrix_cpu = all_indices          # (N, 64) int64
        self.padding_mask_cpu = padding_masks        # (N, 64) bool

    def __len__(self):
        return len(self.chunks)

    @torch.no_grad()
    def get_batch(self, start_idx, batch_size):
        end_idx = min(start_idx + batch_size, len(self.chunks))
        bs = end_idx - start_idx

        # ---- indices/masks for this batch (CPU) ----
        batch_indices_np = self.index_matrix_cpu[start_idx:end_idx]   # (B,64)
        batch_padding_np = self.padding_mask_cpu[start_idx:end_idx]   # (B,64)

        # ---- Deduplicate frame indices within the batch to reduce decode+transfer ----
        flat = batch_indices_np.reshape(-1)  # (B*64,)
        uniq, inv = np.unique(flat, return_inverse=True)  # uniq: (U,), inv: (B*64,)
        inv = inv.reshape(bs, self.samples_per_clip)      # (B,64)

        # ---- Decode only unique frames ----
        # decord returns NDArray (U,H,W,3) uint8-like
        frames_u = self.vr.get_batch(uniq).asnumpy()  # numpy uint8, (U,H,W,3)

        # ---- Move to GPU and normalize (same as your output semantics) ----
        # Keep it efficient: move uint8 -> GPU once, then convert/normalize on GPU.
        frames_u_t = torch.from_numpy(frames_u).to(self.device, non_blocking=False)  # (U,H,W,3), uint8
        frames_u_t = frames_u_t.permute(0, 3, 1, 2).contiguous()                # (U,3,H,W)

        # Match your dtype/normalization path
        frames_u_t = frames_u_t.to(torch.bfloat16).div_(255.0)
        frames_u_t.sub_(self.mean).div_(self.std)

        # ---- Reconstruct (B,64,C,H,W) by indexing unique-frame bank ----
        inv_t = torch.from_numpy(inv).to(device=self.device, dtype=torch.long)  # (B,64)
        batch_tensor = frames_u_t[inv_t]  # (B,64,3,H,W)

        # ---- Apply padding exactly like your code ----
        if batch_padding_np.any():
            pad_t = torch.from_numpy(batch_padding_np).to(device=self.device, dtype=torch.bool)  # (B,64)
            batch_tensor.masked_fill_(pad_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0)

        # ---- Meta (CPU, light) ----
        current_coi = self.coi[start_idx:end_idx]
        current_chunks = self.chunks[start_idx:end_idx]
        meta = {
            "rel_start": torch.tensor([c[0] for c in current_coi]),
            "rel_end": torch.tensor([c[1] for c in current_coi]),
            "clip_duration": torch.tensor([k[1] - k[0] for k in current_chunks]),
        }

        return batch_tensor, meta, bs

class VideoFeatureExtractor(torch.nn.Module):
    def __init__(self, config: VIBEConfig, device: str = "cpu", low_mem_usage: bool = True):
        super().__init__()
        self.config = config
        self.model = None
        self.device = device
        self.low_mem_usage = low_mem_usage

    def load_model(self):
        self.model = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256",
                                               output_hidden_states=True,
                                               attn_implementation="sdpa",
                                               torch_dtype=self.config.video_extractor_dtype,
                                               ).eval().to(self.device)

    def cleanup(self):
        if self.model is not None:
            self.model.cpu()
            self.model = None
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
            video_duration = get_video_info(video_path)[2]
            seconds_before_chunk = self.config.video_extractor_chunk_length_seconds - self.config.tr

            chunks, chunk_of_interests = split_movie_into_chunks(
                video_duration,
                self.config.tr,
                self.config.video_extractor_chunk_length_seconds,
                seconds_before_chunk
            )

            loader = VideoLoader(video_path, chunks,
                                 chunk_of_interests, self.config, self.device)
            all_features = []

            num_chunks = len(chunks)
            batch_size = self.config.video_extractor_batch_size

            pbar = tqdm(total=num_chunks,
                        desc=f"Extracting Video Features", leave=False)
            for i in range(0, num_chunks, batch_size):
                # Fetch batch directly (Instant GPU Slice)
                batch_tensors, batch_meta, actual_bs = loader.get_batch(
                    i, batch_size)

                feats = self.run_batch_inference(
                    batch_tensors, batch_meta, actual_bs)
                all_features.extend(feats)
                pbar.update(actual_bs)

            pbar.close()

            return torch.stack(all_features)

    def run_batch_inference(self, batch_tensors, batch_meta, actual_batch_size):
        with torch.inference_mode():
            batch_tensors = batch_tensors.to(self.device)
            B = batch_tensors.shape[0]
            outputs = self.model(
                pixel_values_videos=batch_tensors).hidden_states

            selected_layers = outputs[-self.config.video_extractor_num_last_hidden_states:]
            avg_features = torch.stack(selected_layers, dim=0).mean(dim=0)

            # Spatial Pooling
            avg_features = avg_features.reshape(
                B, -1, 16, 16, avg_features.shape[-1])
            B, T, H, W, D = avg_features.shape
            # Note: B might be smaller than requested batch_size at the end of video

            flat_feats = avg_features.permute(
                0, 1, 4, 2, 3).reshape(B * T, D, H, W)
            pooled_feats = F.adaptive_avg_pool2d(
                flat_feats, (self.config.video_extractor_pool_size, self.config.video_extractor_pool_size))
            time_feats = pooled_feats.reshape(B, T, -1)

        batch_results = []
        rel_starts = batch_meta['rel_start']
        rel_ends = batch_meta['rel_end']
        clip_durations = batch_meta['clip_duration']

        for i in range(actual_batch_size):
            t_feat = time_feats[i]
            rel_start = rel_starts[i].item()
            rel_end = rel_ends[i].item()
            clip_dur = clip_durations[i].item()

            tok0 = min(T - 1, int(round(rel_start / clip_dur * T)))
            tok1 = min(T, max(tok0 + 1, int(round(rel_end / clip_dur * T))))

            final_feat = t_feat[tok0:tok1].mean(dim=0)
            batch_results.append(final_feat)

        return batch_results


def split_movie_into_chunks(video_duration, tr, chunk_length, seconds_before_chunk):
    chunks, chunk_of_interests = [], []
    start_time = 0.0
    while start_time < video_duration:
        chunk_start = start_time - seconds_before_chunk

        chunk_end = min(chunk_start + chunk_length, video_duration)

        rel_start = start_time - chunk_start
        rel_end = min(rel_start + tr, chunk_end - chunk_start)

        chunks.append((chunk_start, chunk_end))
        chunk_of_interests.append((rel_start, rel_end))
        start_time += tr
    return chunks, chunk_of_interests


def get_video_info(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    video_duration = total_frames / fps
    del vr
    return fps, total_frames, video_duration
