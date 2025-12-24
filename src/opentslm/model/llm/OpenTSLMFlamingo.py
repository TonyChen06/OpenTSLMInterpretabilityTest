# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from collections import defaultdict
import numpy as np
from opentslm.model.encoder.CNNTokenizer import CNNTokenizer
from opentslm.model.llm.TimeSeriesFlamingoWithTrainableEncoder import (
    TimeSeriesFlamingoWithTrainableEncoder,
)
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
import torch
import torch._dynamo
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.llm.TimeSeriesLLM import TimeSeriesLLM
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


class SignalContributionTracker:
    """
    Tracks signal contribution metrics during model forward passes.

    Measures:
    - residual_stream: |x| magnitude of input to cross-attention blocks
    - total_contribution: |x_new - x| total contribution from block (attn + ff)
    - ecg_signal_contribution: |raw_attn * tanh(attn_gate)| - the TRUE ECG signal contribution
    - ff_contribution: |raw_ff * tanh(ff_gate)| - feedforward contribution (not signal-related)
    """

    def __init__(self):
        self.measurements = defaultdict(list)
        self.hooks = []
        self._enabled = False
        # Temporary storage for raw attention output (captured by inner hook)
        self._pending_raw_attn = {}
        self._pending_raw_ff = {}

    def _make_attn_hook(self, layer_idx: int):
        """Hook for the inner attention module to capture raw output BEFORE gating."""
        def hook(module, inputs, output):
            if not self._enabled:
                return
            with torch.no_grad():
                self._pending_raw_attn[layer_idx] = output.norm(dim=-1).mean().item()
        return hook

    def _make_ff_hook(self, layer_idx: int):
        """Hook for the inner feedforward module to capture raw output BEFORE gating."""
        def hook(module, inputs, output):
            if not self._enabled:
                return
            with torch.no_grad():
                self._pending_raw_ff[layer_idx] = output.norm(dim=-1).mean().item()
        return hook

    def _make_block_hook(self, layer_idx: int, layer):
        """Create a forward hook for the outer GatedCrossAttentionBlock."""
        def hook(module, inputs, output):
            if not self._enabled:
                return

            x = inputs[0]  # residual stream input
            x_new = output  # output after cross-attn block

            with torch.no_grad():
                # |x| - residual stream magnitude
                x_magnitude = x.norm(dim=-1).mean().item()

                # Total contribution (attn + ff)
                total_contribution = (x_new - x).norm(dim=-1).mean().item()

                # Get gate values
                attn_gate_tanh = torch.tanh(layer.attn_gate).item()
                ff_gate_tanh = torch.tanh(layer.ff_gate).item()

                # Get raw outputs captured by inner hooks
                raw_attn = self._pending_raw_attn.get(layer_idx, 0.0)
                raw_ff = self._pending_raw_ff.get(layer_idx, 0.0)

                # TRUE ECG signal contribution = raw_attn * |tanh(attn_gate)|
                ecg_contribution = raw_attn * abs(attn_gate_tanh)

                # FF contribution (not from ECG signal)
                ff_contribution = raw_ff * abs(ff_gate_tanh)

                # Percentages
                if x_magnitude > 1e-8:
                    total_pct = (total_contribution / x_magnitude) * 100
                    ecg_pct = (ecg_contribution / x_magnitude) * 100
                    ff_pct = (ff_contribution / x_magnitude) * 100
                else:
                    total_pct = ecg_pct = ff_pct = 0.0

                self.measurements[f'layer_{layer_idx}'].append({
                    'residual_stream': x_magnitude,
                    'total_contribution': total_contribution,
                    'total_contribution_pct': total_pct,
                    'raw_attn_output': raw_attn,
                    'attn_gate_tanh': attn_gate_tanh,
                    'ecg_signal_contribution': ecg_contribution,
                    'ecg_signal_contribution_pct': ecg_pct,
                    'raw_ff_output': raw_ff,
                    'ff_gate_tanh': ff_gate_tanh,
                    'ff_contribution': ff_contribution,
                    'ff_contribution_pct': ff_pct,
                })

                # Clear pending values
                self._pending_raw_attn.pop(layer_idx, None)
                self._pending_raw_ff.pop(layer_idx, None)

        return hook

    def register_hooks(self, model):
        """Register hooks on all gated cross-attention blocks and their inner modules."""
        self.remove_hooks()  # Clear any existing hooks

        # Find the gated cross-attention layers
        lang_encoder = model.lang_encoder

        if hasattr(lang_encoder, 'gated_cross_attn_layers'):
            for idx, layer in enumerate(lang_encoder.gated_cross_attn_layers):
                if layer is not None:
                    # Hook the inner attention module (captures raw output before gating)
                    h1 = layer.attn.register_forward_hook(self._make_attn_hook(idx))
                    self.hooks.append(h1)

                    # Hook the inner feedforward module
                    h2 = layer.ff.register_forward_hook(self._make_ff_hook(idx))
                    self.hooks.append(h2)

                    # Hook the outer block (calculates final metrics)
                    h3 = layer.register_forward_hook(self._make_block_hook(idx, layer))
                    self.hooks.append(h3)

            n_layers = len([l for l in lang_encoder.gated_cross_attn_layers if l is not None])
            print(f"[SignalContributionTracker] Registered hooks on {n_layers} layers ({len(self.hooks)} total hooks)")
        else:
            print("[SignalContributionTracker] Warning: Could not find gated_cross_attn_layers")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def enable(self):
        """Enable measurement collection."""
        self._enabled = True

    def disable(self):
        """Disable measurement collection."""
        self._enabled = False

    def clear(self):
        """Clear all measurements."""
        self.measurements.clear()

    def get_summary(self) -> Dict:
        """Get summary statistics across all samples."""
        if not self.measurements:
            return {'error': 'No measurements collected'}

        summary = {}

        # Per-layer statistics
        for layer_name, measurements in self.measurements.items():
            summary[layer_name] = {
                'residual_stream_mean': float(np.mean([m['residual_stream'] for m in measurements])),
                'total_contribution_mean': float(np.mean([m['total_contribution'] for m in measurements])),
                'total_contribution_pct_mean': float(np.mean([m['total_contribution_pct'] for m in measurements])),
                'raw_attn_output_mean': float(np.mean([m['raw_attn_output'] for m in measurements])),
                'attn_gate_tanh_mean': float(np.mean([m['attn_gate_tanh'] for m in measurements])),
                'ecg_signal_contribution_mean': float(np.mean([m['ecg_signal_contribution'] for m in measurements])),
                'ecg_signal_contribution_pct_mean': float(np.mean([m['ecg_signal_contribution_pct'] for m in measurements])),
                'raw_ff_output_mean': float(np.mean([m['raw_ff_output'] for m in measurements])),
                'ff_gate_tanh_mean': float(np.mean([m['ff_gate_tanh'] for m in measurements])),
                'ff_contribution_mean': float(np.mean([m['ff_contribution'] for m in measurements])),
                'ff_contribution_pct_mean': float(np.mean([m['ff_contribution_pct'] for m in measurements])),
                'n_samples': len(measurements)
            }

        # Overall average across all layers
        all_measurements = []
        for measurements in self.measurements.values():
            all_measurements.extend(measurements)

        summary['overall'] = {
            'residual_stream_mean': float(np.mean([m['residual_stream'] for m in all_measurements])),
            'total_contribution_mean': float(np.mean([m['total_contribution'] for m in all_measurements])),
            'total_contribution_pct_mean': float(np.mean([m['total_contribution_pct'] for m in all_measurements])),
            'raw_attn_output_mean': float(np.mean([m['raw_attn_output'] for m in all_measurements])),
            'attn_gate_tanh_mean': float(np.mean([m['attn_gate_tanh'] for m in all_measurements])),
            'ecg_signal_contribution_mean': float(np.mean([m['ecg_signal_contribution'] for m in all_measurements])),
            'ecg_signal_contribution_pct_mean': float(np.mean([m['ecg_signal_contribution_pct'] for m in all_measurements])),
            'raw_ff_output_mean': float(np.mean([m['raw_ff_output'] for m in all_measurements])),
            'ff_gate_tanh_mean': float(np.mean([m['ff_gate_tanh'] for m in all_measurements])),
            'ff_contribution_mean': float(np.mean([m['ff_contribution'] for m in all_measurements])),
            'ff_contribution_pct_mean': float(np.mean([m['ff_contribution_pct'] for m in all_measurements])),
            'n_measurements': len(all_measurements)
        }

        return summary

# Monkey-patch FlamingoLayer to add attention_type property for compatibility with newer transformers
from open_flamingo.src.flamingo_lm import FlamingoLayer


def _attention_type_property(self):
    """Proxy the attention_type attribute from the underlying decoder layer."""
    return getattr(self.decoder_layer, "attention_type", None)


# Add the attention_type property to FlamingoLayer
FlamingoLayer.attention_type = property(_attention_type_property) # type: ignore


class OpenTSLMFlamingo(TimeSeriesLLM):
    def __init__(
        self,
        device: str,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        cross_attn_every_n_layers: int = 1,
        decoder_layers_attr_name: str = None,
        freeze_lm_embeddings: bool = False,
        **flamingo_kwargs,
    ):
        super().__init__(device)
        print(f"Flamingo Using device: {self.device}")
        time_series_encoder = CNNTokenizer().to(device)

        text_tokenizer = AutoTokenizer.from_pretrained(
            llm_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
        )

        lang_encoder = AutoModelForCausalLM.from_pretrained(
            llm_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
            device_map={"": device},
            attn_implementation="eager",
        )

        # add Flamingo special tokens to the tokenizer
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            text_tokenizer.pad_token = "<PAD>"

        # convert LM to FlamingoLM
        extend_instance(lang_encoder, FlamingoLMMixin)

        def _infer_decoder_layers_attr_name(model):
            __KNOWN_DECODER_LAYERS_ATTR_NAMES = {
                "opt": "model.decoder.layers",
                "gptj": "transformer.h",
                "gpt-j": "transformer.h",
                "pythia": "gpt_neox.layers",
                "llama": "model.layers",
                "gptneoxforcausallm": "gpt_neox.layers",
                "mpt": "transformer.blocks",
                "mosaicgpt": "transformer.blocks",
                "gemma": "model.layers",
                "gemma2": "model.layers",
                "gemma3": "model.layers",
                "medgemma": "model.layers",
            }

            # Special handling for Gemma3 models with different architectures
            model_class_name = model.__class__.__name__
            if "gemma3" in model_class_name.lower():
                if "ConditionalGeneration" in model_class_name:
                    # Gemma3ForConditionalGeneration (multimodal 4B model) - layers are at language_model.layers
                    return "language_model.layers"
                else:
                    # Gemma3ForCausalLM (text-only 1B model) - layers are at standard model.layers
                    return "model.layers"

            # Original logic for non-Gemma3 models
            for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
                if k.lower() in model.__class__.__name__.lower():
                    return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

            raise ValueError(
                f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
            )

        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        lang_encoder.resize_token_embeddings(len(text_tokenizer))

        # Fix compatibility for Gemma3Config which has hidden_size in text_config
        if hasattr(lang_encoder.config, "text_config") and hasattr(
            lang_encoder.config.text_config, "hidden_size"
        ):
            if not hasattr(lang_encoder.config, "hidden_size"):
                lang_encoder.config.hidden_size = (
                    lang_encoder.config.text_config.hidden_size
                )

        model = TimeSeriesFlamingoWithTrainableEncoder(
            SimpleNamespace(visual=time_series_encoder),
            lang_encoder,
            text_tokenizer.encode("<|endofchunk|>")[-1],
            text_tokenizer.encode("<image>")[-1],
            vis_dim=ENCODER_OUTPUT_DIM,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            **flamingo_kwargs,
        )

        # Freeze all parameters
        model.requires_grad_(False)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

        # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
        model.perceiver.requires_grad_(True)
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        if not freeze_lm_embeddings:
            model.lang_encoder.get_input_embeddings().requires_grad_(True)
            # TODO: investigate also training the output embeddings when untied

        # additonally unfreeze encoder
        # Note: Flamingo.__init__ extracts vision_encoder.visual, so vision_encoder IS the CNNTokenizer
        # However, in some cases vision_encoder may be a SimpleNamespace wrapper, so check first
        if hasattr(model.vision_encoder, 'requires_grad_'):
            model.vision_encoder.requires_grad_(True)
        elif hasattr(model.vision_encoder, 'visual') and hasattr(model.vision_encoder.visual, 'requires_grad_'):
            model.vision_encoder.visual.requires_grad_(True)

        self.model = model
        self.llm = model
        self.text_tokenizer = text_tokenizer

        # Initialize signal contribution tracker (disabled by default)
        self._signal_tracker = SignalContributionTracker()
        self._signal_tracker.register_hooks(model)

    def enable_signal_tracking(self):
        """Enable signal contribution tracking during forward passes."""
        self._signal_tracker.enable()
        print("[OpenTSLMFlamingo] Signal contribution tracking ENABLED")

    def disable_signal_tracking(self):
        """Disable signal contribution tracking."""
        self._signal_tracker.disable()
        print("[OpenTSLMFlamingo] Signal contribution tracking DISABLED")

    def clear_signal_measurements(self):
        """Clear all collected signal contribution measurements."""
        self._signal_tracker.clear()

    def get_signal_contribution_summary(self) -> Dict:
        """
        Get summary of signal contribution metrics.

        Returns dict with:
        - Per-layer: residual_stream, raw_attn_output, ecg_signal_contribution, raw_ff_output, ff_contribution
        - Overall: ecg_signal_contribution_pct (TRUE ECG influence), ff_contribution_pct, total_contribution_pct
        """
        return self._signal_tracker.get_summary()

    def print_signal_contribution_summary(self):
        """Print a formatted summary of signal contribution metrics."""
        summary = self.get_signal_contribution_summary()

        if 'error' in summary:
            print(f"[Signal Contribution] {summary['error']}")
            return

        print("\n" + "=" * 70)
        print("SIGNAL CONTRIBUTION SUMMARY")
        print("=" * 70)

        overall = summary['overall']
        print(f"\nOverall (across all {overall['n_measurements']} measurements):")
        print(f"  Residual stream magnitude:      {overall['residual_stream_mean']:.4f}")
        print(f"\n  *** RAW MODULE OUTPUTS (before gating) ***")
        print(f"  Raw attention output:           {overall['raw_attn_output_mean']:.4f}")
        print(f"  Raw FF output:                  {overall['raw_ff_output_mean']:.4f}")
        print(f"\n  *** GATE VALUES ***")
        print(f"  tanh(attn_gate):                {overall['attn_gate_tanh_mean']:.6f}")
        print(f"  tanh(ff_gate):                  {overall['ff_gate_tanh_mean']:.6f}")
        print(f"\n  *** TRUE ECG SIGNAL CONTRIBUTION ***")
        print(f"  ECG contribution (raw*gate):    {overall['ecg_signal_contribution_mean']:.6f}")
        print(f"  ECG contribution %:             {overall['ecg_signal_contribution_pct_mean']:.4f}%")
        print(f"\n  *** FEEDFORWARD CONTRIBUTION ***")
        print(f"  FF contribution (raw*gate):     {overall['ff_contribution_mean']:.4f}")
        print(f"  FF contribution %:              {overall['ff_contribution_pct_mean']:.4f}%")
        print(f"\n  *** TOTAL (ECG + FF) ***")
        print(f"  Total contribution:             {overall['total_contribution_mean']:.4f}")
        print(f"  Total contribution %:           {overall['total_contribution_pct_mean']:.4f}%")

        print(f"\nPer-layer breakdown:")
        for layer_name in sorted(summary.keys(), key=lambda x: int(x.split('_')[1]) if x != 'overall' else -1):
            if layer_name == 'overall':
                continue
            s = summary[layer_name]
            print(f"  {layer_name}: ECG={s['ecg_signal_contribution_pct_mean']:.4f}%, FF={s['ff_contribution_pct_mean']:.4f}%, Total={s['total_contribution_pct_mean']:.4f}%")

        print("=" * 70)

    @property
    def tokenizer(self):
        """Alias for text_tokenizer for compatibility."""
        return self.text_tokenizer

    def pad_and_apply_batch(
        self, batch: List[Dict[str, any]], include_labels: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def pad_time_series(batch, max_length=None):
            """Pad time series to the same length (either max in batch or specified max)"""
            time_series = [item["time_series"] for item in batch]

            # Determine target length (either specified or max in batch)
            if max_length is None:
                max_length = max(ts.shape[1] for ts in time_series)

            padded_series = []
            for ts in time_series:
                current_length = ts.shape[1]
                if current_length < max_length:
                    # Pad with zeros to reach max_length
                    # Ensure padding has the same number of dimensions as the time series
                    padding_shape = list(ts.shape)
                    padding_shape[1] = max_length - current_length
                    padding = torch.zeros(
                        padding_shape, device=ts.device, dtype=ts.dtype
                    )
                    padded = torch.cat([ts, padding], dim=1)
                else:
                    # If already at or exceeding max_length, truncate
                    padded = ts[:, :max_length]

                padded_series.append(padded)

            return torch.stack(padded_series)

        cast_dtype = None
        tokenizer = self.text_tokenizer
        media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
            "input_ids"
        ][-1]

        # Process time series data
        images = pad_time_series(batch).to(
            self.device, dtype=cast_dtype, non_blocking=True
        )
        images = images.unsqueeze(1)  # Add time dimension

        # Process text inputs WITH answers
        text_inputs = []
        # Process text inputs WITH answers
        text_inputs = []
        prompt_lengths = []

        for item in batch:
            # Build the prompt text without answer
            prompt_text = item["pre_prompt"]
            for ts_text in item["time_series_text"]:
                prompt_text += f" {tokenizer.decode([media_token_id])} {ts_text} {tokenizer.decode([endofchunk_token_id])}"
            if item["post_prompt"]:
                prompt_text += f" {item['post_prompt']}"

            if include_labels:
                text_inputs.append(prompt_text)
                continue

            # Store the prompt length in tokens
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prompt_tokens))

            # Add the answer to create full text
            full_text = prompt_text + f" {item['answer']}"
            text_inputs.append(full_text)

        # Tokenize full text (prompt + answer)
        tokenized = tokenizer(text_inputs, padding="longest", return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device, non_blocking=True)
        attention_mask = tokenized.attention_mask.to(self.device, non_blocking=True)

        if include_labels:
            return input_ids, images, attention_mask, None

        # Create labels matrix (-100 for masked tokens)
        labels = torch.full_like(input_ids, -100)

        # Set labels for answer tokens using the stored prompt lengths
        for i, prompt_length in enumerate(prompt_lengths):
            non_padding_indices = torch.where(input_ids[i] != tokenizer.pad_token_id)[0]
            answer_indices = non_padding_indices[non_padding_indices >= prompt_length]

            if len(answer_indices) > 0:
                labels[i, answer_indices] = input_ids[i, answer_indices]

        return input_ids, images, attention_mask, labels

    def generate(
        self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        # Temporarily disable compilation to avoid data-dependent operation issues
        original_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True

        try:
            with torch.inference_mode():
                input_ids, images, attention_mask, _ = self.pad_and_apply_batch(
                    batch, include_labels=True
                )

                gen_ids = self.llm.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.text_tokenizer.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    **generate_kwargs,
                )

                # Remove input ids from generation
                answer_only_ids = gen_ids[:, input_ids.shape[1] :]

                return self.text_tokenizer.batch_decode(
                    answer_only_ids, skip_special_tokens=True
                )
        finally:
            # Restore original compilation setting
            torch._dynamo.config.disable = original_disable

    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        batch: same format as generate()
        answers: List[str] of length B
        """
        input_ids, images, attention_mask, labels = self.pad_and_apply_batch(
            batch, include_labels=False
        )

        output = self.model(
            vision_x=images,
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output[0]

    def get_eos_token(self) -> str:
        return self.text_tokenizer.eos_token

    def store_to_file(self, path: str = "best_model.pt"):
        # The cross attention layers are part of the LLM state dict since we extended
        # the LLM with FlamingoLMMixin using extend_instance. This adds the
        # cross attention layers directly to the LLM model, so they are automatically
        # included when we save llm.state_dict()
        state_dict = {
            "llm": self.llm.state_dict(),  # Includes cross attention layers
        }
        torch.save(state_dict, path)
        print(f"Model saved to {path}")

    def load_from_file(self, path: str = "best_model.pt"):
        """
        Load model parameters with non-strict loading to handle Flamingo-specific layers.
        """
        checkpoint = torch.load(path, map_location=self.device)

        if "llm" in checkpoint:
            model_state = checkpoint["llm"]
        elif "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
        else:
            raise RuntimeError("No recognized model state key in checkpoint.")

        # Handle DDP (DistributedDataParallel) if needed
        if hasattr(self, "module"):
            model_state = {f"module.{k}": v for k, v in model_state.items()}

        # Remove 'model.' prefix if present in checkpoint keys
        if all(k.startswith("model.") for k in model_state.keys()):
            model_state = {
                k.replace("model.", "", 1): v for k, v in model_state.items()
            }

        # Load state dict with strict=False to handle missing/unexpected keys
        missing_keys, unexpected_keys = self.load_state_dict(model_state, strict=False)
        if missing_keys:
            print(f"⚠️  Warning: Missing keys when loading checkpoint:")
            for key in missing_keys[:10]:
                print(f"   - {key}")
            if len(missing_keys) > 10:
                print(f"   ... and {len(missing_keys) - 10} more keys")
        if unexpected_keys:
            print(f"⚠️  Warning: Unexpected keys when loading checkpoint:")
            for key in unexpected_keys[:10]:
                print(f"   - {key}")
            if len(unexpected_keys) > 10:
                print(f"   ... and {len(unexpected_keys) - 10} more keys")
        self.to(self.device)

    def eval_prompt(
        self, prompt: FullPrompt, max_new_tokens: int = 1000, normalize: bool = False
    ) -> str:
        """
        Evaluate a prompt and return the generated text.
        """
        # Temporarily disable compilation to avoid data-dependent operation issues
        original_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        try:
            batch = [prompt.to_dict()]
            self.eval()
            batch = extend_time_series_to_match_patch_size_and_aggregate(
                batch, normalize=normalize
            )
            print("Generating")
            output = self.generate(batch, max_new_tokens=max_new_tokens)
            print(f"Generated output: {output[0]}")
            return output[0]
        finally:
            # Restore original compilation setting
            torch._dynamo.config.disable = original_disable
