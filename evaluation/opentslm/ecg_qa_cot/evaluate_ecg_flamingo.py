#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Evaluate OpenTSLMFlamingo on ECG-QA CoT dataset with signal contribution tracking.

This script runs inference on the ECG-QA test set and measures:
1. Accuracy/F1 metrics
2. Signal contribution metrics (residual_stream, gated_cross_attn_output, signal_contribution_pct)

Usage:
    python evaluate_ecg_flamingo.py --checkpoint path/to/best_model.pt [--max_samples 100] [--use_noise]

Output:
    - Accuracy and F1 metrics per template
    - Signal contribution summary showing how much the ECG signal influences model output
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path for imports
script_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(script_dir / 'src'))
sys.path.insert(0, str(script_dir / 'src' / 'open_flamingo'))

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE


def setup_device():
    """Setup the device for model inference."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device


def load_model(checkpoint_path: str, device: str, llm_id: str = "meta-llama/Llama-3.2-1B"):
    """Load the trained OpenTSLMFlamingo model."""
    print(f"Loading model from {checkpoint_path}...")

    model = OpenTSLMFlamingo(
        device=device,
        llm_id=llm_id,
        cross_attn_every_n_layers=1,
    )

    model.load_from_file(checkpoint_path)
    model.eval()
    print("Model loaded successfully")
    return model


def extract_answer(text: str) -> str:
    """Extract the final answer from model text."""
    if text is None:
        return ""
    if "Answer: " not in text:
        return text.strip()
    answer = text.split("Answer: ")[-1].strip()
    answer = re.sub(r"<\|.*?\|>|<eos>$", "", answer).strip()
    answer = re.sub(r"\.$", "", answer).strip()
    return answer


def normalize_label(label: str) -> str:
    """Lowercase, strip, and remove trailing punctuation."""
    if label is None:
        return ""
    return label.lower().strip().rstrip(".,!?;:")


def evaluate_sample(ground_truth: str, prediction: str, template_id: int) -> Dict[str, Any]:
    """Evaluate a single sample."""
    pred_raw = extract_answer(prediction)
    gt_raw = extract_answer(ground_truth)

    pred_norm = normalize_label(pred_raw)
    gt_norm = normalize_label(gt_raw)

    possible_answers = ECGQACoTQADataset.get_possible_answers_for_template(template_id)
    possible_answers_lower = [a.lower().strip() for a in possible_answers]

    pred_supported = pred_norm in possible_answers_lower
    gt_supported = gt_norm in possible_answers_lower

    is_correct = int(pred_norm == gt_norm)

    return {
        "accuracy": is_correct,
        "f1_score": float(is_correct),
        "prediction_normalized": pred_norm,
        "ground_truth_normalized": gt_norm,
        "prediction_supported": pred_supported,
        "ground_truth_supported": gt_supported,
        "template_id": template_id,
        "possible_answers": possible_answers,
    }


def run_evaluation(
    model: OpenTSLMFlamingo,
    dataset: ECGQACoTQADataset,
    max_samples: int = None,
    max_new_tokens: int = 400,
) -> Dict[str, Any]:
    """Run evaluation on the dataset with signal tracking."""

    # Enable signal tracking
    model.enable_signal_tracking()
    model.clear_signal_measurements()

    # Create dataloader with proper collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        )
    )

    results = []
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    print(f"\nRunning inference on {num_samples} samples...")
    print("=" * 70)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, total=num_samples, desc="Evaluating")):
            if idx >= num_samples:
                break

            try:
                # batch is a list with one item (batch_size=1)
                sample = batch[0]
                template_id = sample.get("template_id") or sample.get("cot_template_id")

                # Generate prediction - model.generate expects a batch (list of samples)
                predictions = model.generate(batch, max_new_tokens=max_new_tokens)
                prediction = predictions[0] if predictions else ""

                # Get ground truth
                ground_truth = sample.get("answer", "")

                # Evaluate
                metrics = evaluate_sample(ground_truth, prediction, template_id)

                result = {
                    "sample_idx": idx,
                    "template_id": template_id,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "metrics": metrics,
                }
                results.append(result)

                # Print first few samples
                if idx < 3:
                    print(f"\nSample {idx + 1}:")
                    print(f"  Template: {template_id}")
                    print(f"  Ground truth: {metrics['ground_truth_normalized']}")
                    print(f"  Prediction: {metrics['prediction_normalized']}")
                    print(f"  Correct: {metrics['accuracy']}")

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Disable tracking
    model.disable_signal_tracking()

    return {
        "results": results,
        "signal_contribution": model.get_signal_contribution_summary(),
    }


def calculate_aggregate_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate aggregate metrics including Macro-F1 from results."""
    if not results:
        return {}

    # Group by template
    template_groups = defaultdict(list)
    for r in results:
        template_id = r["metrics"]["template_id"]
        template_groups[template_id].append(r["metrics"])

    # Per-template stats with F1
    template_stats = {}
    total_correct = 0
    total_samples = 0
    all_template_macro_f1s = []

    for template_id, metrics_list in template_groups.items():
        n_samples = len(metrics_list)
        n_correct = sum(m["accuracy"] for m in metrics_list)
        accuracy = n_correct / n_samples if n_samples > 0 else 0

        # Calculate per-class F1 for this template
        # Group by ground truth class
        class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for m in metrics_list:
            gt = m["ground_truth_normalized"]
            pred = m["prediction_normalized"]
            if pred == gt:
                class_stats[gt]["tp"] += 1
            else:
                class_stats[gt]["fn"] += 1
                class_stats[pred]["fp"] += 1

        # Calculate F1 per class
        class_f1_scores = {}
        template_f1_sum = 0
        valid_classes = 0
        for class_name, stats in class_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            class_f1_scores[class_name] = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "support": tp + fn,
            }
            if tp + fn > 0:  # Only count classes that appear in ground truth
                template_f1_sum += f1
                valid_classes += 1

        macro_f1 = template_f1_sum / valid_classes if valid_classes > 0 else 0

        template_stats[template_id] = {
            "num_samples": n_samples,
            "accuracy": accuracy,
            "correct": n_correct,
            "macro_f1": macro_f1,
            "num_classes": valid_classes,
        }

        total_correct += n_correct
        total_samples += n_samples
        all_template_macro_f1s.append(macro_f1)

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    # Overall Macro-F1: average of per-template Macro-F1s (unweighted)
    overall_macro_f1 = sum(all_template_macro_f1s) / len(all_template_macro_f1s) if all_template_macro_f1s else 0

    return {
        "overall": {
            "total_samples": total_samples,
            "total_correct": total_correct,
            "accuracy": overall_accuracy,
            "macro_f1": overall_macro_f1,
        },
        "per_template": template_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenTSLMFlamingo on ECG-QA with signal tracking")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (None for all)")
    parser.add_argument("--max_new_tokens", type=int, default=400, help="Max tokens to generate")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B", help="LLM ID")
    parser.add_argument("--use_noise", action="store_true", help="Replace ECG signals with noise")
    parser.add_argument("--noise_seed", type=int, default=67, help="Seed for noise generation")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    # Setup
    device = setup_device()

    # Configure noise mode
    if args.use_noise:
        print(f"[NOISE MODE] ECG signals will be replaced with Gaussian noise (seed={args.noise_seed})")
        ECGQACoTQADataset.set_noise_mode(use_noise=True, noise_type="gaussian", noise_seed=args.noise_seed)
    else:
        ECGQACoTQADataset.set_noise_mode(use_noise=False)

    # Load model
    model = load_model(args.checkpoint, device, args.llm_id)

    # Load dataset (eval_only=True for faster loading - only loads test split)
    print("Loading ECG-QA CoT dataset (test split only)...")
    dataset = ECGQACoTQADataset(
        split="test",
        EOS_TOKEN=model.text_tokenizer.eos_token,
        max_samples=args.max_samples,
        preload_processed_data=True,
        eval_only=True,
    )
    print(f"Loaded {len(dataset)} samples")

    # Run evaluation
    eval_results = run_evaluation(
        model,
        dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )

    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(eval_results["results"])

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    overall = aggregate_metrics.get("overall", {})
    print(f"\nOverall Accuracy: {overall.get('accuracy', 0):.4f} ({overall.get('total_correct', 0)}/{overall.get('total_samples', 0)})")
    print(f"Overall Macro-F1: {overall.get('macro_f1', 0):.4f}")

    print(f"\nPer-Template Metrics:")
    for template_id, stats in sorted(aggregate_metrics.get("per_template", {}).items()):
        print(f"  Template {template_id}: Acc={stats['accuracy']:.4f}, F1={stats.get('macro_f1', 0):.4f} ({stats['correct']}/{stats['num_samples']})")

    # Print signal contribution summary
    model.print_signal_contribution_summary()

    # Save results if output path specified
    if args.output:
        output_data = {
            "checkpoint": args.checkpoint,
            "use_noise": args.use_noise,
            "aggregate_metrics": aggregate_metrics,
            "signal_contribution": eval_results["signal_contribution"],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return eval_results


if __name__ == "__main__":
    main()
