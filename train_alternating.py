#!/usr/bin/env python3
"""
Small wrapper to run the existing LLaVA training pipeline with the alternating-attention Qwen2 model.

This script intentionally keeps changes minimal: it builds CLI arguments expected by
`llava.train.train.train()` (which uses `HfArgumentParser`) and calls that function.

Usage examples (from repo root):
  python train_alternating.py \
    --model_name_or_path YOUR/QWEN2-0.5B-PATH \
    --data_path path/to/llava_instructions.json \
    --output_dir outputs/alt_qwen2 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 3

Notes:
- `model_name_or_path` can be a local folder or a HF repo id. For LLaVA-OneVision (Qwen2-0.5B)
  please provide the correct checkpoint path you have available locally (or on HF).
- The script reuses the full training pipeline in `llava/train/train.py` including the
  `LLaVATrainer`, data preprocessing and alternating-attention aware model builder.
"""
import argparse
import os
import sys
from typing import Optional


def inspect_model_type(model_name_or_path: str) -> Optional[str]:
    """Return the HF config.model_type if available, otherwise None."""
    try:
        # Local import to avoid raising at module import time if transformers is not installed
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name_or_path)
        return getattr(cfg, "model_type", None)
    except Exception:
        return None


def build_and_call_train(argv=None):
    # Build thin wrapper around llava.train.train.train which reads args from sys.argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_name_or_path", required=False,
                        help="Primary model path or repo. If not provided, use --alt_checkpoint or --backbone_model.")
    parser.add_argument("--backbone_model", required=False, help="Path or repo id of the backbone pretrained LM to initialize from.")
    parser.add_argument("--alt_checkpoint", required=False, help="Path or repo id of an alternating-attn pretrained checkpoint to start from.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training")
    parser.add_argument("--bits", type=int, default=16, help="Model bits, e.g. 16, 8, 4")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Optional attn implementation to pass to train()")
    parser.add_argument(
        "--variant",
        choices=["auto", "alternating", "alternating_cross", "base"],
        default="auto",
        help="Which model variant you expect to train. 'auto' will accept whatever the checkpoint provides.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only load model config and print model_type, then exit")
    parser.add_argument("--force", action="store_true", help="If set, continue even when model_type doesn't match requested variant")
    parser.add_argument(
        "--init_from_backbone",
        action="store_true",
        help="If set and --backbone_model is provided and --variant is alternating or alternating_cross, attempt to instantiate the alternating model class and load matching weights from the backbone (partial load).",
    )
    parser.add_argument("--inspect_data", action="store_true", help="Load dataset and print sample count + one example, then exit")
    parser.add_argument("--inspect_weights", action="store_true", help="If used with --init_from_backbone, load backbone weights into alternating class and print missing/unexpected keys, then exit")
    parser.add_argument("--inspect_data_verbose", action="store_true", help="If set, call dataset[0] to show tokenized sample (may trigger image processing)")
    parser.add_argument("--throughput_log_steps", type=int, default=50, help="When training with the wrapper's trainer, log throughput every N steps.")

    # parse only to provide helpful errors; we'll forward args to the project's HfArgumentParser
    args, _ = parser.parse_known_args(argv)

    # Minimal validation
    if not os.path.exists(args.data_path):
        print(f"ERROR: data_path does not exist: {args.data_path}")
        sys.exit(2)

    if not args.model_name_or_path and not args.alt_checkpoint and not args.backbone_model:
        print("ERROR: please provide at least one of --model_name_or_path, --alt_checkpoint or --backbone_model")
        sys.exit(2)

    # Pre-flight: inspect model config.model_type
    # Decide which model path to inspect for model_type
    inspect_path = args.alt_checkpoint or args.model_name_or_path or args.backbone_model
    model_type = inspect_model_type(inspect_path)
    print(f"Detected model_type: {model_type}")

    # If dry-run, exit after printing the model_type
    if args.dry_run:
        print("Dry run requested; exiting after model_type inspection.")
        return

    # Validate requested variant against model_type
    def variant_matches(variant: str, model_type: Optional[str]) -> bool:
        if variant == "auto":
            return True
        if model_type is None:
            return False
        model_type_low = model_type.lower()
        if variant == "alternating":
            return "alternating_attn" in model_type_low or "with_alternating_attn" in model_type_low
        if variant == "alternating_cross":
            return "alternating_cross" in model_type_low or "with_alternating_cross_attn" in model_type_low
        if variant == "base":
            return not ("alternating_attn" in model_type_low or "alternating_cross" in model_type_low or "with_alternating_attn" in model_type_low or "with_alternating_cross_attn" in model_type_low)
        return False

    if not variant_matches(args.variant, model_type) and not args.force:
        print(
            "\nModel variant mismatch:\n",
            f"Requested variant: {args.variant}\n",
            f"Detected model_type: {model_type}\n",
        )
        print("Use --force to continue anyway, or point --model_name_or_path to a compatible checkpoint.")
        sys.exit(3)

    # If the user requested an explicit alt checkpoint, prefer that for direct training
    chosen_model_path = args.alt_checkpoint or args.model_name_or_path or args.backbone_model

    # Inspect dataset if requested (lightweight: shows count and raw first sample dict)
    if args.inspect_data:
        try:
            from transformers import AutoTokenizer
            from llava.train.train import make_supervised_data_module, DataArguments

            tokenizer_for_data = AutoTokenizer.from_pretrained(chosen_model_path or args.backbone_model, use_fast=False)
            data_args_obj = DataArguments(data_path=args.data_path)
            data_module = make_supervised_data_module(tokenizer=tokenizer_for_data, data_args=data_args_obj)
            dataset = data_module["train_dataset"]
            print(f"Dataset loaded. Number of samples: {len(dataset)}")
            if len(dataset) > 0:
                print("Sample (raw dict preview):")
                # show raw sample dict without heavy __getitem__ processing
                try:
                    print(dataset.list_data_dict[0])
                except Exception:
                    print("(could not access raw sample content)")
            print("Inspect data requested; exiting.")
            return
        except Exception as e:
            print("Failed to inspect data:", e)
            raise

    # Construct argv for HfArgumentParser used in llava.train.train.train
    # Note: We forward only standard training args; variant/dry_run/force are handled here.
    hf_args = [
        "--model_name_or_path",
        chosen_model_path,
        "--data_path",
        args.data_path,
        "--output_dir",
        args.output_dir,
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--bits",
        str(args.bits),
        "--do_train",
    ]

    if args.fp16:
        hf_args.append("--fp16")
    if args.bf16:
        hf_args.append("--bf16")
    # Make model_name more explicit if user wants alternating attn class name style
    # e.g. if user supplies special model path that includes `with_alternating_attn`

    # Set sys.argv so HfArgumentParser sees these args
    sys_argv = [sys.argv[0]] + hf_args
    sys.argv = sys_argv

    # If the user requested to initialize an alternating model from backbone (no alt checkpoint available)
    if args.init_from_backbone and args.backbone_model and args.variant in ("alternating", "alternating_cross") and not args.alt_checkpoint:
        # We'll construct the alternating model class, try to load backbone weights where possible,
        # then build data module and run LLaVATrainer directly to start training.
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
            from llava.train.train import make_supervised_data_module, TrainingArguments as LLTrainingArguments, DataArguments as LLDataArguments
            from llava.train.llava_trainer import LLaVATrainer

            print("Initializing alternating-attn model from backbone checkpoint (partial weight load)...")

            # Import appropriate alternating class based on variant (Qwen2-specific here)
            if args.variant == "alternating":
                from llava.model.language_model.llava_qwen_with_alternating_attn import (
                    LlavaQwenWithAlternatingAttnForCausalLM,
                )
                AltClass = LlavaQwenWithAlternatingAttnForCausalLM
            else:
                from llava.model.language_model.llava_qwen_with_alternating_cross_attn import (
                    LlavaQwenWithAlternatingCrossAttnForCausalLM,
                )
                AltClass = LlavaQwenWithAlternatingCrossAttnForCausalLM

            # Load backbone tokenizer and config
            tokenizer = AutoTokenizer.from_pretrained(args.backbone_model, use_fast=False)
            base_cfg = AutoConfig.from_pretrained(args.backbone_model)

            # Create alternating model instance from base config
            alt_model = AltClass(base_cfg)

            # Load backbone model weights and copy matching keys (selective copy)
            print("Loading backbone weights (this may be large)...")
            base_model = AutoModelForCausalLM.from_pretrained(args.backbone_model, low_cpu_mem_usage=True)
            base_state = base_model.state_dict()

            # Build selective mapping: only copy keys that exist in both models and have the same shape,
            # excluding known multimodal / projector keys.
            alt_state = alt_model.state_dict()
            mm_exclude = ["mm_projector", "vision_resampler", "vision_tower", "mm_projector.bin", "projector"]
            matched = {}
            for k, v in base_state.items():
                if k in alt_state:
                    if any(ex in k for ex in mm_exclude):
                        continue
                    try:
                        if v.shape == alt_state[k].shape:
                            matched[k] = v
                    except Exception:
                        # skip if shape attribute missing or mismatch
                        continue

            print(f"Found {len(matched)} matching params to copy from backbone to alternating model.")
            _ = alt_model.load_state_dict(matched, strict=False)

            # For reporting, compute missing/unexpected relative to attempted load
            missing = [k for k in alt_state.keys() if k not in matched]
            unexpected = [k for k in matched.keys() if k not in alt_state]
            print(f"Selective copy complete. Matched keys: {len(matched)}, Alt model total keys: {len(alt_state)}")
            if args.inspect_weights:
                print("--- sample copied keys ---")
                for k in list(matched.keys())[:20]:
                    print(k)
                print("--- sample missing keys (first 20) ---")
                for k in (missing[:20]):
                    print(k)
                print("Inspect weights requested; exiting after inspection.")
                return

            # Build training args and data module
            training_args = LLTrainingArguments(output_dir=args.output_dir, per_device_train_batch_size=args.per_device_train_batch_size, num_train_epochs=args.num_train_epochs)
            data_args = LLDataArguments(data_path=args.data_path)
            data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

            # Attach throughput callback when we construct trainer here
            from transformers import TrainerCallback
            import time
            import torch

            class ThroughputCallback(TrainerCallback):
                """Logs approximate samples/sec and GPU memory every N steps."""

                def __init__(self, log_steps=50):
                    self.log_steps = log_steps
                    self.last_time = None
                    self.last_step = None

                def on_train_begin(self, args, state, control, **kwargs):
                    self.last_time = time.time()
                    self.last_step = state.global_step

                def on_train_batch_end(self, args, state, control, **kwargs):
                    # state.global_step is incremented at step end
                    if self.last_step is None:
                        self.last_step = state.global_step
                        self.last_time = time.time()
                        return
                    step_delta = state.global_step - self.last_step
                    if step_delta >= self.log_steps:
                        now = time.time()
                        elapsed = now - self.last_time
                        # Estimate effective samples per step across all processes
                        per_device = getattr(args, "per_device_train_batch_size", 1)
                        grad_acc = getattr(args, "gradient_accumulation_steps", 1)
                        world_size = getattr(args, "world_size", 1)
                        samples_per_step = per_device * grad_acc * world_size
                        samples = step_delta * samples_per_step
                        s_per_s = samples / elapsed if elapsed > 0 else float("inf")
                        # GPU memory
                        if torch.cuda.is_available():
                            mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                            mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                        else:
                            mem_alloc = mem_reserved = 0.0
                        print(f"Throughput: {s_per_s:.1f} samples/s over {step_delta} steps ({elapsed:.2f}s). GPU alloc={mem_alloc:.2f}GB reserved={mem_reserved:.2f}GB")
                        self.last_time = now
                        self.last_step = state.global_step

            cb = ThroughputCallback(log_steps=args.throughput_log_steps)
            trainer = LLaVATrainer(model=alt_model, tokenizer=tokenizer, args=training_args, callbacks=[cb], **data_module)
            trainer.train()
            trainer.save_state()
            print("Training (partial-init) finished or stopped.")
            return
        except Exception as e:
            print("Failed to init alternating model from backbone:", e)
            raise

    # Import and call the train() entrypoint for normal flows (alt checkpoint or backbone as model)
    try:
        from llava.train.train import train as llava_train

        # train() accepts an optional attn_implementation argument; forward if provided
        if args.attn_implementation is not None:
            llava_train(attn_implementation=args.attn_implementation)
        else:
            llava_train()
    except Exception as e:
        print("Training failed with exception:", e)
        raise


def main():
    build_and_call_train()


if __name__ == "__main__":
    main()
