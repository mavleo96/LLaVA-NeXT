#!/usr/bin/env python3
"""
Simple script to:
 - download/load a backbone LM checkpoint (e.g., lmms-lab/llava-onevision-qwen2-0.5b-ov)
 - instantiate an alternating-attention model class (Qwen alternating or alternating_cross)
 - selectively copy matching weights (by name+shape) from backbone -> alternating model
 - load a small text-only sample set from a JSON/JSONL manifest and run a short training loop

This is intentionally minimal and avoids the more complex upstream trainer. It's meant for
quick tests and to produce a working initialized alternating model you can continue training.

Notes:
 - The script prefers text-only samples. If your dataset includes images and you want to include them,
   enhance the data loader to use the repository's `LazySupervisedDataset` and image folder settings.
 - Large models will be downloaded; ensure you have enough disk space. The script loads weights in CPU memory
   and will move the alternating model to the device you specify (default: cuda if available else cpu).
"""

import argparse
import json
import os
import random
from collections import OrderedDict

# defer heavy imports like torch to runtime to keep import-time light


def parse_args():
    p = argparse.ArgumentParser(description="Simple init-from-backbone and short train for alternating-attn model")
    p.add_argument("--backbone_model", required=True, help="HF repo id or local path for backbone LM (e.g. lmms-lab/llava-onevision-qwen2-0.5b-ov)")
    p.add_argument("--variant", choices=["alternating","alternating_cross"], default="alternating")
    p.add_argument("--data_path", required=True, help="Path to JSON or JSONL manifest (list of samples in LLAVA format)")
    p.add_argument("--use_lazy_dataset", action="store_true", help="Use the repository's LazySupervisedDataset to load multimodal data (images). If set, data_path may be a URL->local mapping JSON which will be converted to a LLAVA manifest automatically")
    p.add_argument("--image_folder", default=None, help="If your manifest image paths are absolute, this can be left empty. Otherwise set the common image folder for the manifest. If not provided the script will infer the common parent of image paths when converting a mapping.")
    p.add_argument("--output_dir", default="outputs/simple_alt", help="Where to save the initialized/fine-tuned alternating model")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_samples", type=int, default=200, help="Limit samples for a quick run (set small for tests)")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_tensorboard", action="store_true", help="Write training metrics to TensorBoard (requires torch.utils.tensorboard)")
    p.add_argument("--log_dir", default="runs/init_alt", help="TensorBoard log directory")
    p.add_argument("--log_steps", type=int, default=2, help="How often (in steps) to log metrics to console and TensorBoard")
    p.add_argument("--val_data_path", default=None, help="Optional validation manifest (JSON/JSONL) for periodic evaluation")
    p.add_argument("--eval_every_steps", type=int, default=50, help="Run evaluation every N training steps (0 = disabled)")
    p.add_argument("--eval_max_batches", type=int, default=5, help="Maximum number of validation batches to evaluate each time")
    return p.parse_args()


def load_manifest(path, max_samples=None):
    samples = []
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    break
    else:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # some manifests are {"samples": [...]}
                # try to find the list
                if "samples" in data and isinstance(data["samples"], list):
                    data = data["samples"]
                else:
                    # fallback, wrap single dict
                    data = [data]
            if max_samples:
                data = data[:max_samples]
            samples.extend(data)
    return samples


def convert_url_local_mapping_to_manifest(mapping_path: str, out_manifest_path: str, image_folder: str = None, max_samples: int = None):
    """Convert a user-provided mapping JSON (original_url -> {"image": "/abs/path.jpg", "text": "/abs/text.txt"})
    into a LLAVA JSONL manifest where each line is a sample like {"image": "rel/path.jpg", "conversations": [{"from":"human","value":"..."}]}

    Assumptions:
    - mapping JSON is a dict mapping strings to dicts with keys 'image' and 'text' (text can be a path or a string).
    - If image paths are absolute, the script will infer the common parent as image_folder and store relative paths in the manifest.
    """
    import os

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    entries = []
    image_paths = []
    for k, v in mapping.items():
        img = v.get("image") or v.get("img")
        txt = v.get("text") or v.get("caption") or v.get("txt")
        if img is None or txt is None:
            continue
        image_paths.append(img)
        # read text if it's a path
        if os.path.exists(txt):
            try:
                with open(txt, "r") as tf:
                    text = tf.read().strip()
            except Exception:
                text = str(txt)
        else:
            text = str(txt)

        entries.append({"image": img, "conversations": [{"from": "human", "value": text}]})
        if max_samples and len(entries) >= max_samples:
            break

    if len(entries) == 0:
        raise RuntimeError("No valid entries found in mapping file")

    if image_folder is None:
        # infer common parent
        try:
            image_folder = os.path.commonpath(image_paths)
        except Exception:
            image_folder = os.path.dirname(image_paths[0])

    # write manifest as json (list) so LazySupervisedDataset can read it
    # convert absolute image paths to relative paths relative to image_folder
    rel_entries = []
    for e in entries:
        img_abs = e["image"]
        try:
            rel = os.path.relpath(img_abs, image_folder)
        except Exception:
            rel = os.path.basename(img_abs)
        rel_entries.append({"image": rel, "conversations": e["conversations"]})

    with open(out_manifest_path, "w") as out_f:
        json.dump(rel_entries, out_f)

    return out_manifest_path, image_folder


def make_text_from_sample(sample):
    # Minimal text-only prompt construction: join conversation values
    # The LLAVA format usually contains `conversations` as a list of {"from":..., "value":...}
    if "conversations" in sample:
        parts = []
        for turn in sample["conversations"]:
            # include speaker token if available
            speaker = turn.get("from", "")
            val = turn.get("value", "")
            if speaker:
                parts.append(f"{speaker}: {val}")
            else:
                parts.append(val)
        return "\n".join(parts)
    # fallback to `text` or `prompt` keys if present
    for key in ("text", "prompt", "instruction"):
        if key in sample:
            return sample[key]
    # final fallback: stringify
    return json.dumps(sample)


def selective_copy_weights(backbone_state, alt_state, skip_prefixes=("mm_projector", "vision_resampler", "mm_")):
    matched = OrderedDict()
    for k, v in backbone_state.items():
        if k in alt_state and alt_state[k].shape == v.shape:
            skip = False
            for p in skip_prefixes:
                if k.startswith(p):
                    skip = True
                    break
            if skip:
                continue
            matched[k] = v
    return matched


def main():
    args = parse_args()
    import torch
    import time
    import math
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        SummaryWriter = None

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Use the repository builder to load the backbone model and tokenizer.
    # This ensures custom LLaVA config/model registrations are handled correctly.
    from llava.model.builder import load_pretrained_model

    print(f"Loading backbone model and tokenizer via llava.model.builder: {args.backbone_model}")
    # load_pretrained_model returns (tokenizer, model, image_processor, context_len)
    # Note: do NOT pass low_cpu_mem_usage here — the builder calls `.from_pretrained(..., low_cpu_mem_usage=True, **kwargs)`
    # and passing it again in kwargs causes a duplicate-key TypeError. Let the builder control that flag.
    tokenizer, backbone, image_processor, context_len = load_pretrained_model(
        args.backbone_model, None, args.backbone_model, device_map="auto"
    )
    base_config = getattr(backbone, "config", None)

    # 2) choose alternating class and build an alternating config from base config
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

    print("Creating alternating-attn config from backbone config (conservative) ...")
    if base_config is not None:
        try:
            alt_config = AltClass.config_class(**base_config.to_dict())
        except Exception:
            # fallback: try to copy only keys that config accepts
            cfg_dict = base_config.to_dict()
            # remove unknown keys if necessary
            alt_config = AltClass.config_class(**{k: v for k, v in cfg_dict.items() if k in AltClass.config_class.__dict__})
    else:
        # no base config available; use default alt config
        alt_config = AltClass.config_class()

    # instantiate alt model (random init for new params)
    print("Instantiating alternating-attn model (random init for unmatched params)...")
    alt_model = AltClass(alt_config)
    print(f"Using variant:={args.variant}, model class={AltClass.__name__}")

    print("Collecting state dicts for selective copy from backbone model...")
    backbone_state = {k: v.cpu() for k, v in backbone.state_dict().items()}
    alt_state = alt_model.state_dict()

    matched = selective_copy_weights(backbone_state, alt_state)
    print(f"Matched {len(matched)} parameters to copy from backbone into alternating model (out of {len(alt_state)} alt params)")

    # perform the partial load
    missing_keys = []
    try:
        alt_model.load_state_dict(matched, strict=False)
    except Exception as e:
        print("Warning: load_state_dict partial load raised an exception:", e)

    # move alt model to device
    alt_model.to(device)

    # 4) prepare dataset: either use the repo's LazySupervisedDataset (recommended for multimodal)
    if args.use_lazy_dataset:
        print(f"Preparing multimodal dataset using LazySupervisedDataset from {args.data_path}")
        # If the provided mapping is a URL->local mapping, convert it to a LLAVA manifest and infer image_folder
        # Heuristic: if the top-level JSON is a dict mapping strings to dicts with 'image' keys, treat it as mapping
        import tempfile
        maybe_manifest = args.data_path
        image_folder = args.image_folder
        try:
            with open(args.data_path, "r") as f:
                top = json.load(f)
            if isinstance(top, dict) and any(isinstance(v, dict) and ("image" in v or "img" in v) for v in top.values()):
                os.makedirs(args.output_dir, exist_ok=True)
                out_manifest = os.path.join(args.output_dir, "converted_manifest.json")
                maybe_manifest, image_folder = convert_url_local_mapping_to_manifest(args.data_path, out_manifest, image_folder, max_samples=args.max_samples)
                print(f"Converted mapping -> manifest {maybe_manifest}, inferred image_folder={image_folder}")
        except Exception:
            # fall back to treating data_path as a LLAVA manifest
            pass

        # use LazySupervisedDataset
        from llava.train.train import DataArguments, make_supervised_data_module

        data_args = DataArguments(data_path=maybe_manifest, image_folder=image_folder, is_multimodal=True)
        # set image processor so dataset can preprocess images
        # LazySupervisedDataset expects data_args to have an image_processor attribute
        setattr(data_args, "image_processor", image_processor)
        # create dataset and collator
        dm = make_supervised_data_module(tokenizer, data_args)
        train_dataset = dm["train_dataset"]
        data_collator = dm["data_collator"]
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
        using_lazy = True
    else:
        # 4) prepare a tiny text-only dataset from the manifest
        print(f"Loading manifest {args.data_path} (max_samples={args.max_samples})")
        samples = load_manifest(args.data_path, max_samples=args.max_samples)
        text_samples = []
        for s in samples:
            # prefer text-only samples; skip those with 'image' if we don't have image folder
            if "image" in s and "image_folder" not in s:
                # skip sample with image if we can't load it here
                continue
            txt = make_text_from_sample(s)
            text_samples.append(txt)
        if len(text_samples) == 0:
            raise RuntimeError("No usable text-only samples were found in the manifest. Provide text samples or extend the loader to handle images.")

        # tokenize all samples once (simple approach)
        tokenized = tokenizer(text_samples, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")
        labels = input_ids.clone()

        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        using_lazy = False

    # 5) optimizer
    optimizer = torch.optim.AdamW(alt_model.parameters(), lr=args.lr)

    # prepare validation loader (optional)
    val_dataloader = None
    if args.val_data_path is not None and args.eval_every_steps and args.eval_every_steps > 0:
        print(f"Preparing validation data from {args.val_data_path}")
        if args.use_lazy_dataset:
            # try conversion if mapping
            maybe_manifest = args.val_data_path
            val_image_folder = args.image_folder
            try:
                with open(args.val_data_path, "r") as f:
                    top = json.load(f)
                if isinstance(top, dict) and any(isinstance(v, dict) and ("image" in v or "img" in v) for v in top.values()):
                    os.makedirs(args.output_dir, exist_ok=True)
                    out_manifest = os.path.join(args.output_dir, "converted_val_manifest.json")
                    maybe_manifest, val_image_folder = convert_url_local_mapping_to_manifest(args.val_data_path, out_manifest, val_image_folder, max_samples=args.max_samples)
                    print(f"Converted val mapping -> manifest {maybe_manifest}, inferred image_folder={val_image_folder}")
            except Exception:
                pass

            data_args = DataArguments(data_path=maybe_manifest, image_folder=val_image_folder, is_multimodal=True)
            setattr(data_args, "image_processor", image_processor)
            dm = make_supervised_data_module(tokenizer, data_args)
            val_dataset = dm["train_dataset"]
            val_collator = dm["data_collator"]
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_collator)
        else:
            val_samples = load_manifest(args.val_data_path, max_samples=args.eval_max_batches * args.batch_size)
            val_texts = [make_text_from_sample(s) for s in val_samples if "image" not in s]
            if len(val_texts) > 0:
                tokenized = tokenizer(val_texts, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
                v_input_ids = tokenized["input_ids"]
                v_attn = tokenized.get("attention_mask")
                v_labels = v_input_ids.clone()
                v_dataset = torch.utils.data.TensorDataset(v_input_ids, v_attn, v_labels)
                val_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size=args.batch_size, shuffle=False)

    def run_evaluation(model, dataloader, device, max_batches=5):
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        batches = 0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    b_input_ids = batch["input_ids"].to(device)
                    b_attn_mask = batch.get("attention_mask")
                    if b_attn_mask is not None:
                        b_attn_mask = b_attn_mask.to(device)
                    b_labels = batch["labels"].to(device)
                    images = batch.get("images")
                    if images is not None:
                        if isinstance(images, list):
                            images = [im.to(device) if hasattr(im, "to") else im for im in images]
                        else:
                            images = images.to(device)
                    image_sizes = batch.get("image_sizes")
                    modalities = batch.get("modalities")
                    outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_labels, images=images, image_sizes=image_sizes, modalities=modalities)
                else:
                    b_input_ids, b_attn_mask, b_labels = [t.to(device) for t in batch]
                    modality_ids = b_input_ids.new_zeros(b_input_ids.shape, dtype=torch.long)
                    outputs = model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_labels, modality_ids=modality_ids)

                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss = loss.mean().item()
                total_loss += loss
                total_tokens += b_input_ids.numel() if isinstance(b_input_ids, torch.Tensor) else 0
                batches += 1
                if batches >= max_batches:
                    break
        model.train()
        avg_loss = total_loss / max(1, batches)
        try:
            ppl = float(torch.exp(torch.tensor(avg_loss)))
        except Exception:
            ppl = float('inf')
        return avg_loss, ppl

    # optional tensorboard writer
    writer = None
    if args.use_tensorboard:
        if SummaryWriter is None:
            print("TensorBoard SummaryWriter not available (torch.utils.tensorboard). Install tensorboard or use --use_tensorboard=False")
        else:
            os.makedirs(args.log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=args.log_dir)

    # quick training loop (with metrics)
    alt_model.train()
    global_step = 0
    step_t0 = time.time()
    samples_since_t0 = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            if using_lazy:
                # batch is a dict produced by DataCollatorForSupervisedDataset
                b_input_ids = batch["input_ids"].to(device)
                b_attn_mask = batch.get("attention_mask")
                if b_attn_mask is not None:
                    b_attn_mask = b_attn_mask.to(device)
                b_labels = batch["labels"].to(device)

                images = batch.get("images")
                # images may be a list of tensors or a tensor; move tensors to device
                if images is not None:
                    if isinstance(images, list):
                        images = [im.to(device) if hasattr(im, "to") else im for im in images]
                    else:
                        images = images.to(device)

                image_sizes = batch.get("image_sizes")
                modalities = batch.get("modalities")

                outputs = alt_model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_labels, images=images, image_sizes=image_sizes, modalities=modalities)
            else:
                b_input_ids, b_attn_mask, b_labels = [t.to(device) for t in batch]
                # Provide modality IDs (0 = text, 1 = image). For text-only training set all zeros.
                modality_ids = b_input_ids.new_zeros(b_input_ids.shape, dtype=torch.long)
                outputs = alt_model(input_ids=b_input_ids, attention_mask=b_attn_mask, labels=b_labels, modality_ids=modality_ids)

            # outputs is a CausalLMOutputWithPast or similar; loss on index 'loss'
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss = loss.mean()

            loss.backward()

            # gradient norm for logging
            total_norm = 0.0
            for p in alt_model.parameters():
                if p.grad is not None:
                    try:
                        param_norm = p.grad.detach().float().norm(2).item()
                    except Exception:
                        param_norm = 0.0
                    total_norm += param_norm * param_norm
            grad_norm = math.sqrt(total_norm) if total_norm > 0 else 0.0

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # samples processed in this batch
            try:
                bsz = b_input_ids.size(0)
            except Exception:
                bsz = args.batch_size
            samples_since_t0 += bsz

            if global_step % args.log_steps == 0:
                elapsed = time.time() - step_t0
                samples_per_sec = samples_since_t0 / elapsed if elapsed > 0 else 0.0

                # GPU memory
                gpu_mem_gb = None
                if torch.cuda.is_available():
                    try:
                        gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024.0 ** 3)
                    except Exception:
                        gpu_mem_gb = None

                # learning rate (first param group)
                lr = optimizer.param_groups[0].get("lr") if len(optimizer.param_groups) > 0 else None

                print(f"Epoch {epoch} step {global_step} loss={loss.item():.4f} samples/s={samples_per_sec:.2f} lr={lr} grad_norm={grad_norm:.4f} gpu_mem_gb={gpu_mem_gb}")

                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/samples_per_sec", samples_per_sec, global_step)
                    if lr is not None:
                        writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/grad_norm", grad_norm, global_step)
                    if gpu_mem_gb is not None:
                        writer.add_scalar("train/gpu_mem_gb", gpu_mem_gb, global_step)

                # run validation if requested
                if val_dataloader is not None and args.eval_every_steps and args.eval_every_steps > 0:
                    val_loss, val_ppl = run_evaluation(alt_model, val_dataloader, device, max_batches=args.eval_max_batches)
                    print(f"*** Eval @ step {global_step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
                    if writer is not None:
                        writer.add_scalar("eval/loss", val_loss, global_step)
                        writer.add_scalar("eval/ppl", val_ppl, global_step)

                # reset timer
                step_t0 = time.time()
                samples_since_t0 = 0

    # 6) save the model and tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving fine-tuned alternating model to {args.output_dir}")
    try:
        alt_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        print("Warning: failed to save with save_pretrained(), attempting torch.save state_dict:", e)
        torch.save(alt_model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))

    print("Done.")


if __name__ == "__main__":
    main()
