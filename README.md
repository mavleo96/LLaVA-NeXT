

# LLaVA-Alternating-Attn

This repo is a fork of **LLaVA-NeXT** adapted for the new **LLaVA-Alternating-Attn** architecture. It adds alternating self-attention and cross-modality attention schedules for Qwen2 and Mistral based LLaVA models, plus utilities for BLINK evaluation and attention visualization.

## What’s here
- Alternating-attn language models: `llava_qwen_with_alternating_attn.py` (alternating modality-aware masks), `llava_qwen_with_alternating_cross_attn.py` (alternating cross-modality masks), and `llava_mistral_with_alternating_attn.py` (alternating modality-aware masks).
- Blink eval + attention capture: `scripts/blink_eval.py` and `playground/attention_matrix_save_for_blink.py`.
- Quick commands and checkpoints: see `playground/QUICK_CMDS.md`.

## Setup
```bash
conda env create -f env.yml
conda activate llava
pip install -e .
```

## Getting checkpoints
- Alternating-Attn checkpoints: [mavleo96/LLaVA-Alternating-Attn](https://huggingface.co/mavleo96/LLaVA-Alternating-Attn)
  ```bash
  huggingface-cli download mavleo96/LLaVA-Alternating-Attn \
    --local-dir /workspace/checkpoints/llava-alternating-attn \
    --local-dir-use-symlinks False
  ```

## Alternating attention: how it works
- Layer schedule: even-numbered layers use the standard causal mask; odd-numbered layers swap in a modality-aware mask; the final layer always reverts to the standard causal mask. See `llava_qwen_with_alternating_attn.py` and `llava_qwen_with_alternating_cross_attn.py`.
- Mask types (`mask_utils.py`):
  - `modality_ids_to_modality_attention_mask`: isolates text↔text and image↔image (no cross-modal attention).
  - `modality_ids_to_cross_modality_attention_mask`: zeros self-attn diagonals to encourage text↔image mixing.
- Which model name to load:
  - `llava_qwen_with_alternating_attn` → alternating self-attn with modality isolation on odd layers.
  - `llava_qwen_with_alternating_cross_attn` → alternating cross-attn on odd layers for cross-modal mixing.

## Directory highlights
- `llava/model/language_model/`: alternating-attn Qwen2 and Mistral implementations and helpers.
- `playground/`: analysis scripts (attention dumps, quick command cheatsheet).
- `scripts/`: evaluation utilities (e.g., BLINK).

## License
Apache 2.0 (upstream LLaVA-NeXT license applies).