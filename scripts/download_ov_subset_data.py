import os
from datasets import load_dataset
from tqdm import tqdm
import json

# CONFIG_NAME = "aokvqa(cauldron,llava_format)"
# CONFIG_FILE_NAME = "ov_aokvqa_cauldron_llava_format.json"
# CONFIG_NAME = "chartqa(cauldron,llava_format)"
# CONFIG_FILE_NAME = "ov_chartqa_cauldron_llava_format.json"
# CONFIG_NAME = "clevr(cauldron,llava_format)"
# CONFIG_FILE_NAME = "ov_clevr_cauldron_llava_format.json"

CONFIG_NAME = "llava_wild_4v_39k_filtered"
CONFIG_FILE_NAME = "ov_llava_wild_4v_39k_filtered.json"

# CONFIG_NAME = "figureqa(cauldron,llava_format)"
# CONFIG_FILE_NAME = "ov_figureqa_cauldron_llava_format.json"
# CONFIG_NAME = "geomverse(cauldron)"
# CONFIG_FILE_NAME = "ov_geomverse_cauldron.json"
# CONFIG_NAME = "hateful_memes(cauldron,llava_format)"
# CONFIG_FILE_NAME = "ov_hateful_memes_cauldron_llava_format.json"

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
        # JPEG does not support RGBA; convert if needed
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(out_path, format="JPEG")
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)


with open(os.path.join(image_folder, CONFIG_FILE_NAME), "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
