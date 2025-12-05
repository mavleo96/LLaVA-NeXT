#!/usr/bin/env python3
"""
Blink Benchmark Evaluation Script for LLaVA-NeXT

This script evaluates the model's performance on the Blink benchmark,
which tests visual correspondence between images.
"""

import argparse
import json
import re
import torch
from tqdm import tqdm
from datasets import load_dataset
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from peft import PeftModel

from collections import Counter

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")


def extract_and_validate_answer(response, correct_answer):
    # Strip the response of (, ), " ", \n, \t, etc.
    response = response.strip("()\"\n\t")
    # Remove common prefixes and convert to uppercase
    response = response.replace("Answer:", "").replace("ANSWER:", "").replace("Point", "").strip().upper()
    
    # Extract just the letter if it exists
    letter_match = re.search(r'[A-D]', response)
    if letter_match:
        response = letter_match.group()
    else:
        response = ""
    
    response = f"({response})" if response else "()"

    correct = response == correct_answer
    return correct, response



def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-NeXT on Blink benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--model_name", type=str, default="llava_mistral", help="Model name")
    parser.add_argument("--lora_weights_path", type=str, default=None, help="Path to lora weights")
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
    if args.lora_weights_path:
        model = PeftModel.from_pretrained(model, args.lora_weights_path)
        model = model.merge_and_unload()
    
    # Load dataset
    dataset = load_dataset("BLINK-Benchmark/BLINK", args.subtask, split="val")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    results = []
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

        output = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            modalities=["image"],
            do_sample=False,
            max_new_tokens=256,
            use_cache=True,
            output_attentions=False,
            return_dict_in_generate=True,
        )
        response = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
        correct, response_extracted = extract_and_validate_answer(response, item["answer"])

        results.append({
            "id": item["idx"],
            "raw_answer": response,
            "answer": response_extracted,
            "correct_answer": item["answer"],
            "correct": correct,
            "prompt": prompt,
        })

    with open(args.output_path, "w") as f:
        json.dump(results, f)

    # Print final accuracy
    correct_count = sum(result["correct"] for result in results)
    total_count = len(results)
    accuracy = correct_count / total_count
    print(f"Final accuracy: {accuracy:.2f}")

    # Distribution of predicted answers vs correct answers
    predicted_answers = [result["answer"] for result in results]
    correct_answers = [result["correct_answer"] for result in results]
    print(f"Distribution of predicted answers vs correct answers:")
    predicted_answers = sorted(Counter(predicted_answers).items(), key=lambda x: x[0])
    correct_answers = sorted(Counter(correct_answers).items(), key=lambda x: x[0])
    print(f"Predicted answers: {predicted_answers}")
    print(f"Correct answers: {correct_answers}")


if __name__ == "__main__":
    main()

