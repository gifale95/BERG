import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from berg.models.fmri.text2fmri_utils.config import Text2fMRIConfig


class FeatureExtractor:
    """
    A wrapper around a Hugging Face Causal LLM to extract temporal features aligned with fMRI TRs.

    This class handles:
    1. Loading the LLM (with memory optimizations).
    2. aligning text transcripts to fMRI Time Repetition (TR) windows using character offsets.
    3. Extracting and pooling hidden states to create a feature vector per TR.

    Attributes:
        config (Text2fMRIConfig): Configuration object containing model parameters.
        device (str): The target device ('cuda', 'cpu', or 'mps').
        model (AutoModelForCausalLM): The loaded Hugging Face model.
        tokenizer (AutoTokenizer): The loaded tokenizer.
        berg_dir (str): Path to the BERG cache directory.
    """

    def __init__(self, config: Text2fMRIConfig = Text2fMRIConfig(), device: str = "cpu", berg_dir: str = None):
        """
        Initialize the FeatureExtractor.

        Args:
            config (Text2fMRIConfig): Configuration dataclass.
            device (str): Device to load the model on.
            berg_dir (str, optional): Path to the BERG cache directory.
        """
        self.config: Text2fMRIConfig = config
        self.model = None
        self.tokenizer = None
        self.device = device
        self.berg_dir = berg_dir

    def load_model(self):
        """
        Loads the Tokenizer and LLM from the Hugging Face Hub.

        Applies optimizations:
        - `trust_remote_code=True` for custom architectures (e.g., Qwen).
        - `low_cpu_mem_usage=True` to prevent RAM spikes during weight loading.
        - Casts to the dtype specified in config (usually float16) to save VRAM.
        """
        # Construct cache directory path
        cache_dir = None
        if self.berg_dir is not None:
            cache_dir = os.path.join(
                self.berg_dir,
                "encoding_models",
                "modality-fmri",
                "train_dataset-cneuromod",
                "model-text2fmri",
                "encoding_models_weights"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.extractor_LLM, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                self.config.extractor_LLM,
                trust_remote_code=True,
                dtype=self.config.extractor_LLM_dtype,
                low_cpu_mem_usage=True,        # helps avoid CPU RAM spikes during load
                cache_dir=cache_dir
            )
            .to(self.device)
            .eval()
        )

    def cleanup(self):
        if self.model is not None:
            self.model.cpu()
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_prompt(self, transcripts: list[str]) -> tuple[str, list[tuple[int, int]]]:
        """
        Concatenates TR transcripts into a single string and records character spans.

        This ensures the model sees the previous TRs as context for the current TR.

        Args:
            transcripts (list[str]): A list of strings, where the i-th string corresponds 
                to the text spoken/heard during the i-th fMRI TR.

        Returns:
            tuple[str, list[tuple[int, int]]]:
                - The full concatenated string (with newlines separating TRs).
                - A list of (start_index, end_index) tuples indicating the character 
                  span of each TR in the full string.
        """
        char_spans = []
        parts = []
        pos = 0

        for line in transcripts:
            start = pos
            parts.append(line + "\n")   # newline marks TR boundary
            pos += len(line) + 1        # +1 for '\n'
            char_spans.append((start, pos))

        return "".join(parts), char_spans

    def tokens_by_tr(self, offsets: torch.Tensor, char_spans: list[tuple[int, int]]):
        """
        Maps token indices to TR indices based on character overlap.

        Args:
            offsets (torch.Tensor): Tensor of shape (seq_len, 2) containing 
                (char_start, char_end) for each token.
            char_spans (list[tuple[int, int]]): List of (start, end) character 
                boundaries for each TR.

        Returns:
            list[list[int]]: A list of length `num_TRs`. Each entry contains the 
                indices of tokens that fall within that TR's time window.
        """
        per_tr = [[] for _ in char_spans]

        for tok_idx, (tok_start, tok_end) in enumerate(offsets.tolist()):
            # Ignore special tokens with empty or non-overlapping spans
            # typically catches specials at the very beginning
            if tok_end <= char_spans[0][0]:
                continue

            # Assign the token to the first TR it overlaps.
            # (Overlap condition: token_start < TR_end and token_end > TR_start)
            for tr_idx, (start, end) in enumerate(char_spans):
                if tok_start < end and tok_end > start:
                    per_tr[tr_idx].append(tok_idx)
                    break

        return per_tr

    @torch.no_grad()
    def extract_features(
        self,
        transcripts: list[str],
    ):
        """
        Generates fMRI-aligned text features from a list of per-TR transcripts.

        The pipeline processes the text as a single continuous stream to maintain context,
        extracts hidden states from the LLM, and averages them based on time alignment.

        Process:
            1. Concatenate all transcripts.
            2. Run the LLM forward pass (inference mode).
            3. Extract the last N hidden layers.
            4. Average the hidden layers to get one vector per token.
            5. Average the token vectors within each TR's time window.

        Args:
            transcripts (list[str]): List of strings, one per TR.

        Returns:
            torch.Tensor: A Float32 Tensor of shape [Num_TRs, Feature_Dim].
                If a TR has no text, its vector will be all zeros.
        """
        if self.model is None:
            self.load_model()

        num_tr = len(transcripts)

        # Build a single text blob + TR character spans.
        full_text, char_spans = self.build_prompt(transcripts)

        # Tokenize with character offsets
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding=False,
            truncation=False
        )

        input_ids = encoded["input_ids"].to(
            self.model.device)    # (1, seq_len)
        offsets = encoded["offset_mapping"][0]               # (seq_len, 2)

        # --- Safety: extremely long inputs ---
        # If the tokenized length exceeds the model's context window, we truncate with a warning.
        max_len = getattr(getattr(self.model, "config", None),
                          "max_position_embeddings", None)
        if isinstance(max_len, int) and input_ids.size(1) > max_len:
            logging.warning(
                f"tokenized length {input_ids.size(1)} > model max {max_len}. Truncating to first {max_len} tokens."
            )
            input_ids = input_ids[:, :max_len]
            offsets = offsets[:max_len]

        # Forward pass to get hidden states; inference_mode is slightly faster than no_grad
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids,
                                 output_hidden_states=True)

        # hidden_states is a tuple (n_layers+1) of tensors with shape (1, seq_len, hidden_size).
        hidden_last4 = torch.stack(
            # (seq_len, hidden_size)
            outputs.hidden_states[-self.config.extractor_LLM_num_last_hidden_states:], dim=0).mean(dim=0)[0]

        # Token -> TR assignment, then per-TR mean pooling.
        per_tr_tokens = self.tokens_by_tr(
            offsets, char_spans)  # list[list[int]]
        features = torch.zeros(
            (num_tr, hidden_last4.size(1)), dtype=torch.float32).to(self.model.device)

        for i, token_idxs in enumerate(per_tr_tokens):
            if token_idxs:  # non-empty TR
                features[i] = hidden_last4[token_idxs].mean(dim=0).float()
            # else: silent TR -> remain zeros

        return features
