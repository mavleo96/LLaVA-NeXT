import os
import json
import random
import requests
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io
from tqdm import tqdm

# Step 1: Boost parallelism
os.environ["HF_DATASETS_DOWNLOAD_PARALLELISM"] = "16"

# Step 2: Stream LAION dataset
dataset = load_dataset("laion/laion400m", split="train", streaming=True)

# Step 3: Define folders
imgFolder = r"./../data/laion_subset/laion_images"
txtFolder = r"./../data/laion_subset/laion_txts"
mapFile = r"./../data/laion_subset/download_map.json"

# Step 4: Create local folders
os.makedirs(imgFolder, exist_ok=True)
os.makedirs(txtFolder, exist_ok=True)

# Step 5: Load existing URL→filename map
if os.path.exists(mapFile):
    with open(mapFile, "r", encoding="utf-8") as f:
        url_map = json.load(f)
else:
    url_map = {}

# Step 6: Robust field access
def get_url(example):
    for k in ["URL", "url", "image_url"]:
        if k in example and example[k]:
            return example[k]
    return None

def get_text(example):
    for k in ["TEXT", "text", "caption", "DESCRIPTION"]:
        if k in example and example[k]:
            return example[k]
    return ""

# Step 7: Filters
def quality_filter(example):
    caption = get_text(example)
    return (
        example.get("NSFW") == "UNLIKELY"
        and example.get("similarity", 0) > 0.4
        and len(caption.strip()) > 5
    )

# Step 8: Downloader with resolution check + JSON resume
def download_example(example, idx):
    url = get_url(example)
    caption = get_text(example)

    if not url or not caption:
        return False

    # Resume: skip if URL already in map
    if url in url_map:
        return True

    img_path = os.path.join(imgFolder, f"{idx}.jpg")
    txt_path = os.path.join(txtFolder, f"{idx}.txt")

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            if img.width >= 512 and img.height >= 512:
                img.save(img_path)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                # Update map
                url_map[url] = {"image": img_path, "text": txt_path}
                return True
    except:
        pass  # silently skip errors

    return False

# Step 9: Collect a larger pool, then sample 50k
pool_size = 100000   # collect more than needed
max_samples = 50000

filtered_examples = []

for example in dataset:
    if len(filtered_examples) >= pool_size:
        break
    if quality_filter(example):
        filtered_examples.append(example)

# Randomly sample 50k from the pool

sampled_examples = random.sample(filtered_examples, max_samples)

# Step 10: Download sampled examples in parallel
with ThreadPoolExecutor(max_workers=32) as executor:
    futures = []
    for idx, example in enumerate(sampled_examples):
        futures.append(executor.submit(download_example, example, idx))

    for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
        f.result()

# Step 11: Save updated URL→filename map
with open(mapFile, "w", encoding="utf-8") as f:
    json.dump(url_map, f, indent=2)

print(f"Downloaded {len(sampled_examples)} randomly sampled high-quality samples")
