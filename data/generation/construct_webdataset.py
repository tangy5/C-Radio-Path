#!/usr/bin/env python3
"""
Constructs a webdataset from a directory of images.

Features:
- Scans a user-supplied directory for images
- Shuffles the images
- Creates tarfile shards with a configurable max number of images per shard
- Generates a JSON with the sha256 hash for each image
- Organizes tarfiles into subdirectories with max 1000 files each
- Multiprocessing support for faster processing of large datasets
"""

import argparse
import hashlib
import json
import os
import random
import sys
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from queue import Empty
from typing import List, Optional

import webdataset as wds
from tqdm import tqdm

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif'}


def compute_sha256(data: bytes) -> str:
    """Compute the sha256 hash of data."""
    m = hashlib.sha256()
    m.update(data)
    return m.hexdigest()


def find_images(input_dir: str) -> List[str]:
    """Recursively find all image files in the input directory."""
    image_files = []
    input_path = Path(input_dir)

    for root, _, files in os.walk(input_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, file))

    return image_files


def get_subdir_for_shard(shard_idx: int, max_per_subdir: int = 1000) -> str:
    """
    Get the subdirectory name for a given shard index.
    E.g., shard 0-999 -> "1000", shard 1000-1999 -> "2000", etc.
    """
    subdir_idx = (shard_idx // max_per_subdir) + 1
    return str(subdir_idx * max_per_subdir)


def process_image(image_path: str, idx: int) -> Optional[dict]:
    """
    Process a single image file and return a webdataset sample.
    Returns None if the image cannot be read.
    """
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
    except Exception as e:
        return None

    # Compute sha256 hash
    sha256_hash = compute_sha256(image_data)

    # Determine the image extension (without the dot)
    ext = Path(image_path).suffix.lower().lstrip('.')
    # Normalize jpeg to jpg for consistency
    if ext == 'jpeg':
        ext = 'jpg'

    # Create sample key using index (zero-padded)
    key = f"{idx:09d}"

    # Build the sample
    sample = {
        "__key__": key,
        ext: image_data,
        "json": json.dumps({"sha256": sha256_hash}).encode('utf-8'),
    }

    return sample


class SubdirShardWriter:
    """
    A shard writer that organizes tarfiles into subdirectories,
    with a maximum number of tarfiles per subdirectory.
    Supports distributed writing with rank/world_size to avoid shard collisions.
    """
    def __init__(
        self,
        output_dir: str,
        max_count: int,
        max_per_subdir: int = 1000,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.output_dir = output_dir
        self.max_count = max_count
        self.max_per_subdir = max_per_subdir
        self.rank = rank
        self.world_size = world_size

        self._shard_idx = rank  # Start at rank to interleave shards
        self._sample_count = 0
        self._writer = None

    def _get_current_shard_path(self) -> str:
        """Get the path for the current shard."""
        subdir = get_subdir_for_shard(self._shard_idx, self.max_per_subdir)
        shard_dir = os.path.join(self.output_dir, subdir)
        os.makedirs(shard_dir, exist_ok=True)
        return os.path.join(shard_dir, f"shard_{self._shard_idx:06d}.tar")

    def _open_new_shard(self):
        """Open a new shard for writing."""
        if self._writer is not None:
            self._writer.close()

        shard_path = self._get_current_shard_path()
        self._writer = wds.TarWriter(shard_path)
        self._sample_count = 0

    def write(self, sample: dict):
        """Write a sample to the current shard."""
        # Open new shard if needed
        if self._writer is None or self._sample_count >= self.max_count:
            if self._writer is not None:
                self._shard_idx += self.world_size  # Skip by world_size to avoid collisions
            self._open_new_shard()

        self._writer.write(sample)
        self._sample_count += 1

    def close(self):
        """Close the current writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def worker_process(
    rank: int,
    world_size: int,
    work_queue: Queue,
    output_dir: str,
    shard_size: int,
    max_per_subdir: int,
    progress_queue: Queue,
):
    """
    Worker process that pulls images from a work queue and writes to shards.
    Each worker writes to non-overlapping shard indices based on rank.
    """
    writer = SubdirShardWriter(
        output_dir=output_dir,
        max_count=shard_size,
        max_per_subdir=max_per_subdir,
        rank=rank,
        world_size=world_size,
    )

    try:
        while True:
            try:
                item = work_queue.get_nowait()
            except Empty:
                break

            idx, image_path = item
            sample = process_image(image_path, idx)

            if sample is not None:
                writer.write(sample)
                progress_queue.put(1)
            else:
                progress_queue.put(0)  # Signal progress even on failure
    finally:
        writer.close()

    # Signal that this worker is done
    progress_queue.put(None)


def main():
    parser = argparse.ArgumentParser(
        description='Construct a webdataset from a directory of images.'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='The input directory containing images to process.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='The output directory where the webdataset shards will be written.'
    )
    parser.add_argument(
        '-s', '--shard-size',
        type=int,
        default=10000,
        help='Maximum number of images per shard/tarfile. Default: 10000'
    )
    parser.add_argument(
        '--max-per-subdir',
        type=int,
        default=1000,
        help='Maximum number of tarfiles per subdirectory. Default: 1000'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help='Number of worker processes. Default: number of CPUs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling. Default: 42'
    )

    args = parser.parse_args()

    # Determine number of workers
    num_workers = args.workers if args.workers is not None else cpu_count()
    num_workers = max(1, num_workers)

    # Find all images
    print(f"Scanning {args.input} for images...", file=sys.stderr)
    image_files = find_images(args.input)
    print(f"Found {len(image_files)} images.", file=sys.stderr)

    if len(image_files) == 0:
        print("No images found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Shuffle images
    print("Shuffling images...", file=sys.stderr)
    random.seed(args.seed)
    random.shuffle(image_files)

    # Calculate expected number of shards
    num_shards = (len(image_files) + args.shard_size - 1) // args.shard_size
    num_subdirs = (num_shards + args.max_per_subdir - 1) // args.max_per_subdir
    print(f"Will create ~{num_shards} shards across ~{num_subdirs} subdirectories.", file=sys.stderr)
    print(f"Using {num_workers} worker processes.", file=sys.stderr)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create work queue and populate with (index, path) tuples
    work_queue = Queue()
    for idx, image_path in enumerate(image_files):
        work_queue.put((idx, image_path))

    # Create progress queue for tracking
    progress_queue = Queue()

    # Start worker processes
    workers = []
    for rank in range(num_workers):
        p = Process(
            target=worker_process,
            args=(
                rank,
                num_workers,
                work_queue,
                args.output,
                args.shard_size,
                args.max_per_subdir,
                progress_queue,
            ),
        )
        p.start()
        workers.append(p)

    # Track progress with tqdm
    completed_workers = 0
    successful_images = 0
    failed_images = 0

    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        while completed_workers < num_workers:
            try:
                result = progress_queue.get(timeout=300)
                if result is None:
                    # Worker finished
                    completed_workers += 1
                elif result == 1:
                    successful_images += 1
                    pbar.update(1)
                else:
                    failed_images += 1
                    pbar.update(1)
            except Empty:
                # Check if any workers died unexpectedly
                alive = sum(1 for p in workers if p.is_alive())
                if alive == 0:
                    break

    # Wait for all workers to finish
    for p in workers:
        p.join()

    print(f"\nDone! Processed {successful_images} images successfully.", file=sys.stderr)
    if failed_images > 0:
        print(f"Warning: {failed_images} images failed to process.", file=sys.stderr)


if __name__ == '__main__':
    main()
