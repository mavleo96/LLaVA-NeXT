import argparse
import io
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def quality_filter(example):
    caption = example.get("caption", "")
    return (
        example.get("NSFW") == "UNLIKELY"
        and example.get("similarity", 0) > 0.4
        and len(caption.strip()) > 5
    )

def download_example(example, idx, img_folder, txt_folder):
    url = example.get("url")
    caption = example.get("caption", "")

    if not url:
        return None

    try:
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))

        if img.width >= 512 and img.height >= 512:
            img_path = os.path.join(img_folder, f"{idx}.jpg")
            txt_path = os.path.join(txt_folder, f"{idx}.txt")
            img.save(img_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
            return True
    except Exception:
        # Silently skip errors for individual downloads
        pass
    return None

# Stream and download with retry logic
def get_dataset_with_retry(max_retries=5, retry_delay=5):
    """Load dataset with retry logic for connection timeouts"""
    for attempt in range(max_retries):
        try:
            print(f"Loading dataset (attempt {attempt + 1}/{max_retries})...")
            # Increase timeout and add retry configuration
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
            os.environ["HF_DATASETS_OFFLINE"] = "0"
            dataset = load_dataset("laion/laion400m", split="train", streaming=True)
            return dataset
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Failed to load dataset: {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise

def main():
    parser = argparse.ArgumentParser(description="Download a subset of LAION samples")
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of high-quality samples to download",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/vmurugan/laion_subset",
        help="Directory to save downloaded images and texts (default: data/vmurugan/laion_subset)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel download workers (default: 8)",
    )
    args = parser.parse_args()
    max_samples = max(args.num_samples, 0)
    os.environ["HF_DATASETS_DOWNLOAD_PARALLELISM"] = str(args.workers)

    # Setup directories
    img_folder = os.path.join(args.data_dir, "images")
    txt_folder = os.path.join(args.data_dir, "captions")
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(txt_folder, exist_ok=True)

    dataset = get_dataset_with_retry()
    dataset_iter = iter(dataset)
    pbar = tqdm(total=max_samples, desc="Downloading")

    successful_downloads = 0
    next_idx = 0

    def process_done(finished_futures):
        nonlocal successful_downloads
        for future in finished_futures:
            try:
                if future.result():
                    successful_downloads += 1
                    pbar.update(1)
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending = set()

        try:
            while successful_downloads < max_samples:
                if pending and successful_downloads + len(pending) >= max_samples:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    process_done(done)
                    continue

                try:
                    example = next(dataset_iter)
                except StopIteration:
                    print("\nDataset exhausted before reaching requested samples")
                    break
                except Exception as e:
                    print(f"\nError streaming dataset: {e}")
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                    dataset = get_dataset_with_retry(max_retries=3, retry_delay=5)
                    dataset_iter = iter(dataset)
                    continue

                if not quality_filter(example):
                    continue

                future = executor.submit(
                    download_example, example, next_idx, img_folder, txt_folder
                )
                pending.add(future)
                next_idx += 1

                if pending:
                    done, pending = wait(pending, timeout=0)
                    process_done(done)
        finally:
            if pending:
                done, _ = wait(pending)
                process_done(done)
            pbar.close()

    print(f"Complete: {successful_downloads} / {max_samples} downloads")


if __name__ == "__main__":
    main()