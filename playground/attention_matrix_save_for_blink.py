#!/usr/bin/env python3

import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
import json
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init

from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
import gc


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-NeXT on Blink benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--model_name", type=str, default="llava_mistral", help="Model name")
    parser.add_argument("--output_path", type=str, default="blink_results.json", help="Output path for results")
    parser.add_argument("--conv_template", type=str, default="mistral_instruct", help="Conversation template")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8bit")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4bit")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--subtask", type=str, default="Visual_Correspondence", help="BLINK subtask")
    
    args = parser.parse_args()
    
    # Disable torch init for faster loading
    disable_torch_init()
    
    # Load model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device_map=args.device,
        torch_dtype="float16",
        attn_implementation="eager"
    )
    # Note: using anyres increase the memory usage and sometimes not suitable for multi-image setting.
    model.config.image_aspect_ratio = "nobase"
    
    # Load dataset
    dataset = load_dataset("BLINK-Benchmark/BLINK", args.subtask, split="val")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    results = {
        "modality_ids": [],
    }

    # Streaming stats for attention matrices to avoid storing all samples in memory
    running_mean = None  # np.ndarray (float64 for numeric stability)
    running_M2 = None    # np.ndarray (float64)
    sample_count = 0
    for item in tqdm(dataset, desc="Evaluating Blink"):
        image_prompt = "".join(f"Image {i+1}: {DEFAULT_IMAGE_TOKEN}\n" for i in range(4) if item[f"image_{i+1}"] is not None)
        question_text = "Question: " + item["question"]
        detailed_prompt = "Details: " + item["prompt"]
        directive = "Answer with the option’s letter from the given choices directly."
        if args.conv_template == "manual":
            prompt = (
                image_prompt + "\n" +
                question_text + "\n" +
                detailed_prompt + "\n" +
                directive + "\n"
            )
        else:
            conv = conv_templates[args.conv_template].copy()
            conv.append_message(conv.roles[0], image_prompt + "\n" + question_text + "\n" + detailed_prompt + "\n" + directive)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(args.device)

        images = [
            item[f"image_{i+1}"].convert('RGB')
            for i in range(4) if item[f"image_{i+1}"] is not None
        ]
        image_tensors = process_images(images, image_processor, model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=args.device) for _image in image_tensors]
        image_sizes = [image.size for image in images]

        with torch.inference_mode():
            output, modality_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                modalities=["image"],
                do_sample=False,
                max_new_tokens=256,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        attention_matrices = output.attentions
        # Convert only the first attention "step" to numpy (to match prior behavior)
        # and update streaming mean/variance to avoid storing per-sample matrices
        if attention_matrices:
            first_step = attention_matrices[0]
            a_layer_numpy_list = []
            for a_layer in first_step:
                # average over batch and head dimensions
                a_layer_numpy_list.append(a_layer.mean(axis=(0, 1)).detach().cpu().numpy())
            current_attn = np.stack(a_layer_numpy_list, axis=0).astype(np.float32)

            # Welford's algorithm for numerically stable online mean/std
            if running_mean is None:
                running_mean = np.zeros_like(current_attn, dtype=np.float64)
                running_M2 = np.zeros_like(current_attn, dtype=np.float64)
                sample_count = 0
            sample_count += 1
            delta = current_attn.astype(np.float64) - running_mean
            running_mean += delta / sample_count
            delta2 = current_attn.astype(np.float64) - running_mean
            running_M2 += delta * delta2

        # Downcast modality_ids to int8 to save memory (values expected small)
        results["modality_ids"].append(modality_ids.detach().cpu().numpy().astype(np.int8))

        # Proactively free GPU/CPU memory between samples
        del output, attention_matrices, image_tensors, images, input_ids
        torch.cuda.empty_cache()
        gc.collect()

    final_modality_ids = np.stack(results["modality_ids"], axis=0)

    # Compute final mean/std from streaming stats (avoid storing all per-sample matrices)
    if sample_count > 1:
        variance = running_M2 / (sample_count - 1)
    else:
        variance = np.zeros_like(running_M2)
    final_attention_matrix_mean = running_mean.astype(np.float32)
    final_attention_matrix_std = np.sqrt(variance).astype(np.float32)

    final_dict = {
        "attention_matrix_mean": final_attention_matrix_mean,
        "attention_matrix_std": final_attention_matrix_std,
        "modality_ids": final_modality_ids,
    }
    np.savez_compressed(args.output_path, **final_dict)


if __name__ == "__main__":
    main()

