import os
from datasets import load_dataset
from tqdm import tqdm
import json

# Cauldron datasets
# CONFIG_NAME = "aokvqa(cauldron,llava_format)"
# CONFIG_FILE_NAME = "ov_aokvqa_cauldron_llava_format.json"
# CONFIG_NAME = "chartqa(cauldron,llava_format)"
# CONFIG_FILE_NAME = "ov_chartqa_cauldron_llava_format.json"
# CONFIG_NAME = "clevr(cauldron,llava_format)"
# CONFIG_FILE_NAME = "ov_clevr_cauldron_llava_format.json"
# CONFIG_NAME = "tqa(cauldron,llava_format)"
# CONFIG_FILE_NAME = "ov_tqa_cauldron_llava_format.json"
# CONFIG_NAME = "raven(cauldron)"
# CONFIG_FILE_NAME = "ov_raven_cauldron.json"
CONFIG_NAME = "visual7w(cauldron,llava_format)"
CONFIG_FILE_NAME = "ov_visual7w_cauldron_llava_format.json"

# Vision Flan
# CONFIG_NAME = "vision_flan(filtered)"
# CONFIG_FILE_NAME = "ov_vision_flan_filtered.json"

# Image Captioning
# CONFIG_NAME = "image_textualization(filtered)"
# CONFIG_FILE_NAME = "ov_image_textualization_filtered.json"

data = load_dataset(
    "lmms-lab/LLaVA-OneVision-Data",
    CONFIG_NAME,
    split="train",
)

image_folder = "/workspace/data/LLaVA-OneVision-Data"

converted_data = []

for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        # Normalize to a .jpg path and ensure parent directories exist
        stem, _ = os.path.splitext(da["id"])
        json_data["image"] = f"{stem}.jpg"
        out_path = os.path.join(image_folder, json_data["image"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = da["image"]
        # JPEG only supports RGB; convert any other mode (RGBA, palette, grayscale, etc.) to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out_path, format="JPEG")
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)


with open(os.path.join(image_folder, CONFIG_FILE_NAME), "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
