# Query the model on test image

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
import torch
from PIL import Image
import copy

# model_path = "liuhaotian/llava-v1.6-mistral-7b"
# model_name = "llava_mistral_with_alternating_attn"
# model_path = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
model_name = "llava_qwen_with_alternating_attn"
# model_name = "llava_mistral"
# model_path = "liuhaotian/llava-v1.5-7b"
# model_name = "llava-v1.5-7b"
model_base = None
load_8bit = False
load_4bit = False
device_map = "auto"
device = "cuda:1"
attn_implementation = "eager"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=model_base,
    model_name=model_name,
    load_8bit=load_8bit,
    load_4bit=load_4bit,
    device_map=device_map,
    torch_dtype="float16",
    attn_implementation=attn_implementation,
    multimodal=True
)
model.config.image_aspect_ratio = "nobase"

image_file1 = "/home/vmurugan/LLaVA-NeXT/mario.png"
image_file2 = "/home/vmurugan/LLaVA-NeXT/test_image.jpg"
image_file3 = "/home/vmurugan/LLaVA-NeXT/mario.png"
# images = [Image.open(image_file1), Image.open(image_file2)]
# Convert all images to RGB to ensure they have 3 dimensions
images = [Image.open(image_file1).convert('RGB'), Image.open(image_file2).convert('RGB'), Image.open(image_file3).convert('RGB')]

# Process images using the proper LLaVA-NeXT method
image_tensors = process_images(images, image_processor, model.config)
image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
print("image_tensors shape: ", [i.shape for i in image_tensors])
print("image_sizes: ", [image.size for image in images])
print("images: ", images)

# Use proper conversation template for Qwen OneVision
conv_template = "qwen_1_5"
conv = copy.deepcopy(conv_templates[conv_template])

# Define questions for each image
questions = [
    "A point is circled on the first image, labeled with REF. We change the camera position or lighting and shoot the second image. You are given multiple red-circled points on the second image, choices of \"A, B, C, D\" are drawn beside each circle. Which point on the second image corresponds to the point in the first image? Select from the following options.\n(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D"
]

# Build prompts using conversation template
prompts = []
for question in questions:
    conv_copy = copy.deepcopy(conv)
    # Add user message with image tokens (2 images for this example)
    user_message = f"{DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN}\n{question}"
    conv_copy.append_message(conv_copy.roles[0], user_message)
    conv_copy.append_message(conv_copy.roles[1], None)  # None = assistant should respond
    prompts.append(conv_copy.get_prompt())

# Tokenize prompts
input_ids = [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device) for prompt in prompts]

# Pad sequences to the same length
max_len = max(seq.shape[1] for seq in input_ids)
input_ids = [torch.nn.functional.pad(seq, (0, max_len - seq.shape[1]), value=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id) for seq in input_ids]
input_ids = torch.cat(input_ids, dim=0)

print("input_ids.shape: ", input_ids.shape)
print("image_tensors length: ", len(image_tensors))
print("image_sizes: ", [image.size for image in images])

temperature = 0.0

# Set up stopping criteria using the conversation template
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
print(f"Using stop string: {stop_str}")

with torch.inference_mode():
    # Generate with attention weights
    generation_output = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=[image.size for image in images],
        modalities=["image"] * input_ids.shape[0],
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        max_new_tokens=1024,
        output_attentions=True,
        return_dict_in_generate=True,
        use_cache=True
    )
    
    # Extract the generated token IDs and attention weights
    output_ids = generation_output.sequences
    attention_weights = generation_output.attentions
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

print("output_ids.shape: ", output_ids.shape)
print("Attention weights available: ", attention_weights is not None)

if attention_weights is not None:
    print(f"\n=== ATTENTION WEIGHTS ANALYSIS ===")
    print(f"Number of generation steps: {len(attention_weights)}")
    
    if len(attention_weights) > 0:
        print(f"Number of layers per step: {len(attention_weights[0])}")
        
        # Print shapes for first few steps and layers
        for step_idx in range(min(3, len(attention_weights))):
            print(f"\n--- Step {step_idx} ---")
            for layer_idx in range(min(5, len(attention_weights[step_idx]))):
                attn_shape = attention_weights[step_idx][layer_idx].shape
                print(f"  Layer {layer_idx}: {attn_shape}")
            
            if len(attention_weights[step_idx]) > 5:
                print(f"  ... and {len(attention_weights[step_idx]) - 5} more layers")
        
        if len(attention_weights) > 3:
            print(f"\n... and {len(attention_weights) - 3} more steps")
        
        # Summary statistics
        first_step_first_layer = attention_weights[0][0]
        print(f"\n=== SUMMARY ===")
        print(f"Attention tensor dtype: {first_step_first_layer.dtype}")
        print(f"Attention tensor device: {first_step_first_layer.device}")
        print(f"First step, first layer shape: {first_step_first_layer.shape}")
        print(f"Expected shape format: (batch_size, num_heads, seq_len, seq_len)")
    else:
        print("No attention weights found")
else:
    print("Attention weights not available")

for q, o in zip(prompts, outputs):
    print("Query: ", q)
    print("Answer: ", o)
    print("-" * 100)