python llava/eval/blink_eval.py \
  --model_path "liuhaotian/llava-v1.6-mistral-7b" \
  --model_name "llava_mistral" \
  --subtask "Visual_Correspondence" \
  --device "cuda:1" \
  --conv_template "manual"

python llava/eval/blink_eval.py \
  --model_path "lmms-lab/llava-onevision-qwen2-0.5b-ov" \
  --model_name "llava_qwen"  \
  --subtask Visual_Correspondence \
  --device "cuda:1" \
  --conv_template "qwen_1_5"

python scripts/download_laion_subset.py \
  --num_samples 1000 \
  --data_dir /data/vmurugan/laion_subset \
  --workers 32
