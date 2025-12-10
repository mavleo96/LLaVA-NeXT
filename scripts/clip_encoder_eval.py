import argparse
from functools import partial
import json
from typing import Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from datasets import Features, Value, load_dataset
from datasets import Image as DatasetImage
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, SiglipVisionModel


DatasetExample = Dict[str, Union[str, dict, list, Image.Image]]

DATA_FILES = "/workspace/data/Synthetic-Visual-Correspondence-Data/parquet_data/samples_*.parquet"
NUM_SAMPLES = 1000
PATCH_SIZE = 256
CLIP_NAME = "openai/clip-vit-large-patch14-336"
SIGLIP_NAME = "google/siglip-so400m-patch14-384"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def build_dataset(data_files: str, num_samples: int):
    """Load parquet shards and return a shuffled subset."""
    features = Features(
        {
            "image_1": DatasetImage(),
            "image_2": DatasetImage(),
            "id": Value("string"),
            "source": Value("string"),
            "labels_dict": Value("string"),
            "ref_label": Value("string"),
            "ref_label_pos": Value("string"),
        }
    )
    ds = load_dataset("parquet", data_files=data_files, features=features)["train"]
    print("Rows in loaded shard:", len(ds))
    subset = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))
    print("Using subset size:", len(subset))
    return subset


def _ensure_pil_list(images: List[Image.Image]) -> List[Image.Image]:
    return [im if isinstance(im, Image.Image) else Image.fromarray(np.array(im)) for im in images]


@torch.no_grad()
def embed_clip(
    images: List[Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor,
) -> torch.Tensor:
    """Return L2-normalized CLIP image embeddings."""
    images = _ensure_pil_list(images)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_image_features(**inputs)
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.no_grad()
def embed_siglip(
    images: List[Image.Image],
    model: SiglipVisionModel,
    processor: AutoProcessor,
) -> torch.Tensor:
    """Return L2-normalized SigLIP embeddings."""
    images = _ensure_pil_list(images)
    inputs = processor(images=images, return_tensors="pt").to(device)
    outputs = model(**inputs)
    feats = outputs.pooler_output
    return feats / feats.norm(dim=-1, keepdim=True)


def _parse_json_field(field_value, field_name: str):
    """Parse JSON-ish strings; accept already-parsed dict/list."""
    if field_value is None:
        raise ValueError(f"{field_name} is None")
    if isinstance(field_value, (dict, list)):
        return field_value
    if isinstance(field_value, str):
        try:
            return json.loads(field_value)
        except json.JSONDecodeError:
            import ast

            return ast.literal_eval(field_value)
    raise ValueError(f"Unsupported type for {field_name}: {type(field_value)}")


def crop_and_pad(pil_img: Image.Image, center_xy: Tuple[float, float], patch_size: int) -> Image.Image:
    """Center crop with padding to keep patch_size square."""
    if not isinstance(pil_img, Image.Image):
        pil_img = Image.fromarray(np.array(pil_img))

    w, h = pil_img.size
    x, y = float(center_xy[0]), float(center_xy[1])

    half = patch_size // 2
    x0, y0 = int(round(x - half)), int(round(y - half))
    x1, y1 = x0 + patch_size, y0 + patch_size

    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(w, x1), min(h, y1)

    if ix0 >= ix1 or iy0 >= iy1:
        return Image.new("RGB", (patch_size, patch_size), color=(127, 127, 127))

    cropped = pil_img.crop((ix0, iy0, ix1, iy1))
    out = Image.new("RGB", (patch_size, patch_size), color=(127, 127, 127))
    out.paste(cropped, (ix0 - x0, iy0 - y0))
    return out


def safe_get_candidate_list(labels_dict) -> Tuple[List[str], List[Tuple[float, float]]]:
    """Return ordered labels and centers from dict or list."""
    if isinstance(labels_dict, dict):
        items = list(labels_dict.items())
    elif isinstance(labels_dict, list):
        items = labels_dict
    else:
        raise ValueError("Unsupported labels_dict format")

    labels, centers = [], []
    for lab, pos in items:
        labels.append(str(lab))
        if isinstance(pos, dict):
            if "x" in pos and "y" in pos:
                centers.append((float(pos["x"]), float(pos["y"])))
            elif "col" in pos and "row" in pos:
                centers.append((float(pos["col"]), float(pos["row"])))
            else:
                vals = list(pos.values())
                centers.append((float(vals[0]), float(vals[1])))
        elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
            centers.append((float(pos[0]), float(pos[1])))
        else:
            raise ValueError(f"Unsupported coordinate format for label {lab}")
    return labels, centers


def evaluate_visual_correspondence(
    examples: Iterable[DatasetExample],
    embed_fn: Callable[[List[Image.Image]], torch.Tensor],
    patch_size: int,
) -> dict:
    """Compute accuracy and bookkeeping stats."""
    correct = total = skipped = invalid = 0

    for ex in tqdm(examples, desc="Evaluating visual correspondence"):
        try:
            labels_dict = _parse_json_field(ex.get("labels_dict"), "labels_dict")
            ref_label_name = ex.get("ref_label")
            ref_pos = _parse_json_field(ex.get("ref_label_pos"), "ref_label_pos")

            if ref_label_name is None:
                skipped += 1
                continue

            cand_labels, cand_centers = safe_get_candidate_list(labels_dict)
            if str(ref_label_name) not in cand_labels:
                skipped += 1
                continue

            gt_idx = cand_labels.index(str(ref_label_name))
            if isinstance(ref_pos, dict):
                if "x" in ref_pos and "y" in ref_pos:
                    ref_center = (float(ref_pos["x"]), float(ref_pos["y"]))
                elif "col" in ref_pos and "row" in ref_pos:
                    ref_center = (float(ref_pos["col"]), float(ref_pos["row"]))
                else:
                    vals = list(ref_pos.values())
                    ref_center = (float(vals[0]), float(vals[1]))
            elif isinstance(ref_pos, (list, tuple)) and len(ref_pos) >= 2:
                ref_center = (float(ref_pos[0]), float(ref_pos[1]))
            else:
                invalid += 1
                continue

            img_ref, img_cand = ex.get("image_1"), ex.get("image_2")
            if img_ref is None or img_cand is None:
                skipped += 1
                continue

            ref_patch = crop_and_pad(img_ref, ref_center, patch_size)
            cand_patches = [crop_and_pad(img_cand, c, patch_size) for c in cand_centers]
            if not cand_patches:
                skipped += 1
                continue

            ref_feat = embed_fn([ref_patch])
            cand_feats = embed_fn(cand_patches)

            sims = (ref_feat @ cand_feats.T).squeeze(0)
            if int(torch.argmax(sims).item()) == gt_idx:
                correct += 1
            total += 1
        except Exception:
            invalid += 1

    return {
        "accuracy": (correct / total) if total else 0.0,
        "total": total,
        "correct": correct,
        "skipped": skipped,
        "invalid": invalid,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate visual correspondence.")
    parser.add_argument("--model", choices=["clip", "siglip"], default="clip", help="Which vision encoder to use.")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--data-files", type=str, default=DATA_FILES)
    args = parser.parse_args()

    subset = build_dataset(args.data_files, args.num_samples)

    if args.model == "clip":
        model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
        processor = CLIPProcessor.from_pretrained(CLIP_NAME)
        embed_fn = partial(embed_clip, model=model, processor=processor)
    else:
        processor = AutoProcessor.from_pretrained(SIGLIP_NAME)
        model = SiglipVisionModel.from_pretrained(SIGLIP_NAME).to(device)
        model.eval()
        embed_fn = partial(embed_siglip, model=model, processor=processor)

    result = evaluate_visual_correspondence(subset, embed_fn, patch_size=args.patch_size)
    print(
        f"\nResult: accuracy={result['accuracy']:.4f}, correct={result['correct']}, "
        f"total={result['total']}, skipped={result['skipped']}, invalid={result['invalid']}"
    )


if __name__ == "__main__":
    main()
