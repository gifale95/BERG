from __future__ import annotations
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager

from berg.models.fmri.vibe_utils.config import VIBEConfig

EMOTIONAL_PRIMING_PHRASE = (
    "[Instruction]: audience empathy expert, thoroughly examine the emotional state "
    "of the characters and the overall emotional atmosphere of the following scene. "
    "consider and encode the predominant emotions this particular moment is crafted to elicit  "
    "e.g., joy, sadness, tension, surprise, fear, anger, relief, boredom, wonder, etc.). "
    "Pay attention to emotional transitions and intensity. "
)


class TextFeatureExtractor:
    """Extract per-TR text embeddings for VIBE using a causal language model."""

    def __init__(self,
                 config: VIBEConfig,
                 device: str = "cpu",
                 low_mem_usage: bool = True,
                 emotional_priming_phrase: str | None = None):
        """
        Initialize the text feature extractor.

        Args:
            config (VIBEConfig): VIBE configuration.
            device (str): Device used for model inference.
            low_mem_usage (bool): If True, unload model after each extraction call.
            emotional_priming_phrase (str | None): Optional phrase prepended to
                metadata prompt text.
        """
        self.config: VIBEConfig = config
        self.model = None
        self.tokenizer = None
        self.device = device
        self.low_mem_usage = low_mem_usage
        self.emotional_priming_phrase = emotional_priming_phrase or EMOTIONAL_PRIMING_PHRASE

    def load_model(self):
        """
        Loads the Tokenizer and LLM from the Hugging Face Hub.

        Applies optimizations:
        - `trust_remote_code=True` for custom architectures (e.g., Qwen).
        - `low_cpu_mem_usage=True` to prevent RAM spikes during weight loading.
        - Casts to the dtype specified in config (usually float16) to save VRAM.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.text_extractor, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                self.config.text_extractor,
                trust_remote_code=True,
                dtype=self.config.text_extractor_dtype,
                low_cpu_mem_usage=True        # helps avoid CPU RAM spikes during load
            )
            .to(self.device)
            .eval()
        )

    def cleanup(self):
        """Release tokenizer/model references and clear CUDA cache when available."""
        if self.model is not None:
            self.model.cpu()
            self.model = None
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def _model_session(self):
        """Context manager that lazily loads and conditionally unloads the model."""
        if self.model is None:
            self.load_model()
        try:
            yield
        finally:
            if self.low_mem_usage:
                self.cleanup()

    def build_prompt(self,
                     transcripts: list[str],
                     metadata: str = "") -> tuple[str, list[tuple[int, int]]]:
        """
        Build a single prompt string and record character spans per TR.

        This lets each TR use preceding transcript context while keeping a mapping
        back from token offsets to TR indices.

        Args:
            transcripts (list[str]): A list of strings, where the i-th string corresponds 
                to the text spoken/heard during the i-th fMRI TR.

        Returns:
            tuple[str, list[tuple[int, int]]]:
                - The full concatenated string (with newlines separating TRs).
                - A list of (start_index, end_index) tuples indicating the character 
                  span of each TR in the full string.
        """
        metadata = self.emotional_priming_phrase + metadata
        char_spans = []
        parts = [f"[Metadata]: {metadata}\n"]
        pos = len(parts[0])

        for line in transcripts:
            start = pos
            parts.append(line + "\n")   # newline marks TR boundary
            pos += len(line) + 1        # +1 for '\n'
            char_spans.append((start, pos))

        return "".join(parts), char_spans

    def tokens_by_tr(self, offsets: torch.Tensor, char_spans: list[tuple[int, int]]) -> list[list[int]]:
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
        Generate per-TR text features from transcript strings.

        The pipeline processes the text as a single continuous stream to maintain context,
        extracts hidden states from the LLM, and averages them based on time alignment.

        Args:
            transcripts (list[str]): List of strings, one per TR.

        Returns:
            torch.Tensor: Float32 tensor of shape [num_trs, text_extractor_feature_size].
            TRs without mapped tokens remain zero vectors.
        """
        with self._model_session():
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
                outputs.hidden_states[-self.config.text_extractor_num_last_hidden_states:], dim=0).mean(dim=0)[0]

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
