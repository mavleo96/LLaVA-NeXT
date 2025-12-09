import os
import json
from datasets import load_dataset, Features, Image, Value
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from tqdm import tqdm

# Define features
features = Features({
    'image_1': Image(),      # Reference image
    'image_2': Image(),      # Candidate image with 4 points (A-D)
    'id': Value('string'),
    'source': Value('string'),
    'labels_dict': Value('string'),  # JSON: positions of A, B, C, D in image_2
    'ref_label': Value('string'),     # Ground truth: which label (A-D) is correct
    'ref_label_pos': Value('string'), # JSON: position of reference point in image_1
})

# Load dataset
dataset = load_dataset(
    'parquet',
    data_files='/workspace/data/Synthetic-Data/parquet_data/samples_*.parquet',
    features=features
)["train"]

image_folder = "/workspace/data/synthetic_visualcorres/images"
os.makedirs(image_folder, exist_ok=True)


image_prompt = f"Image 1: {DEFAULT_IMAGE_TOKEN}\n Image 2: {DEFAULT_IMAGE_TOKEN}\n "
question_text = "Question: Which point is corresponding to the reference point?"
detailed_prompt = "Details: A point is circled on the first image, labeled with REF. We change the camera position or lighting and shoot the second image. You are given multiple red-circled points on the second image, choices of \"A, B, C, D\" are drawn beside each circle. Which point on the second image corresponds to the point in the first image? Select from the following options.\n(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D"
directive = "Answer with the option’s letter from the given choices directly."

converted_data = []
for da in tqdm(dataset):
    json_data = {}
    json_data["id"] = da["id"]

    # Save images
    stem, _ = os.path.splitext(da["id"])
    out_path_1 = os.path.join(image_folder, f"{stem}_1.jpg")
    out_path_2 = os.path.join(image_folder, f"{stem}_2.jpg")
    img_1 = da["image_1"]
    img_2 = da["image_2"]
    if img_1.mode != "RGB":
        img_1 = img_1.convert("RGB")
    img_1.save(out_path_1, format="JPEG")
    if img_2.mode != "RGB":
        img_2 = img_2.convert("RGB")
    img_2.save(out_path_2, format="JPEG")
    json_data["image"] = [out_path_1, out_path_2]

    # Save conversations
    json_data["conversations"] = [
        {"from": "human", "value": image_prompt + "\n" + question_text + "\n" + detailed_prompt + "\n" + directive},
        {"from": "gpt", "value": f"({da['ref_label']})"}
    ]

    # Save data
    converted_data.append(json_data)

with open(os.path.join(image_folder, "synthetic_visualcorres_data.json"), "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
