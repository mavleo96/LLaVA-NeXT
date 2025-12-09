### BLINK Evaluation Commands
#### Usage:
```
python scripts/blink_eval.py \
  --model_path "liuhaotian/llava-v1.6-mistral-7b" \
  --model_name "llava_mistral" \
  --output_path "results/llava-v1.6-mistral-7b-visual_correspondence.json" \
  --subtask "Visual_Correspondence" \
  --device "cuda:1" \
  --conv_template "manual"
```

#### Usage with LoRA finetuned model:
```
python scripts/blink_eval.py \
  --model_path "/workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn" \
  --model_name "llava_qwen"  \
  --lora_weights_path "/workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn-finetune/checkpoint-11000" \
  --subtask "Visual_Correspondence" \
  --device "cuda:1" \
  --conv_template "qwen_2"
```

#### Options:
| model_path | model_name | conv_template |
| ---------- | ---------- | ------------- |
| "liuhaotian/llava-v1.6-mistral-7b" | "llava_mistral" | "manual" |
| "lmms-lab/llava-onevision-qwen2-7b-ov" | "llava_qwen" | "qwen_2" |
| "lmms-lab/llava-onevision-qwen2-0.5b-ov" | "llava_qwen" | "qwen_2" |

### Command to save attention matrix for Blink evaluation:
```
python playground/attention_matrix_save_for_blink.py \
  --model_path "/workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn" \
  --model_name "llava_qwen_with_alternating_attn" \
  --lora_weights_path "/workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn-finetune/checkpoint-11000" \
  --subtask Visual_Correspondence \
  --device "cuda:1" \
  --conv_template "qwen_2" \
  --output_path "/data/vmurugan/llava-next/attention_outputs/llava_qwen2-0.5b_aa_unfinetuned_visualcorres_attention"
```

### Command to create unfinetuned model checkpoint:
```
huggingface-cli download lmms-lab/llava-onevision-qwen2-0.5b-ov \
    --local-dir /workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn \
    --local-dir-use-symlinks False
```
