## Quickstart: Training alternating-attention models (concise)

This quickstart shows the minimal steps to train an alternating-attention model using the repository's existing training code and the helper wrapper `train_alternating.py`.

Two common flows:

- A — you have an alternating-attn checkpoint already (use `--model_name_or_path`).
- B — you have only a backbone LM checkpoint and want to initialize an alternating-attn model from it (use `--init_from_backbone`).

1) Inspect dataset (always a good first step)

```bash
python train_alternating.py \
  --model_name_or_path lmms-lab/llava-onevision-qwen2-0.5b-ov \
  --data_path /path/to/laion_subset.json \
  --inspect_data
```

2A) If you have alternating-attn checkpoint, run training directly:

```bash
python train_alternating.py \
  --model_name_or_path your_repo/alternating_attn_checkpoint \
  --data_path /path/to/laion_subset.json \
  --variant alternating \
  --output_dir outputs/alt_run \
  --per_device_train_batch_size 8 \
  --num_train_epochs 3
```

2B) If you only have a backbone LM, inspect selective init first:

```bash
python train_alternating.py \
  --backbone_model your_repo/backbone_ckpt \
  --data_path /path/to/laion_subset.json \
  --init_from_backbone \
  --variant alternating \
  --inspect_weights
```

If `--inspect_weights` shows a reasonable number of matched parameters, start full training (same flags but without `--inspect_weights`). You can also add `--throughput_log_steps 50` to get samples/sec traces.

3) Quick note on verification: If you want to verify the wrapper will instantiate the alternating class without downloading weights, use `--dry_run`:

```bash
python train_alternating.py --model_name_or_path your_repo/alternating_attn_checkpoint --dry_run
```

That's it. See `docs/TRAIN_ALTERNATING_EXTENDED.md` for more examples, the selective-copy code snippet, and troubleshooting tips.
