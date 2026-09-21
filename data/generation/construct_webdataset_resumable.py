#!/usr/bin/env python3
"""
Resumable WebDataset Construction for Image-Text Pairs.

Features:
- Processes paired image (.png) and caption (.txt) files
- Resumable: tracks progress and can continue after interruption
- Automatic checkpointing every N samples
- Signal handling for graceful shutdown
- Multiprocessing support for fast processing
- State persistence to disk for crash recovery

Designed for 4-hour runtime limits (checkpoint and resume capability).
"""

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time
import pickle
from datetime import datetime
from multiprocessing import Process, Queue, cpu_count, Manager
from pathlib import Path
from queue import Empty
from typing import List, Optional, Dict, Tuple, Set
import logging

import webdataset as wds
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('webdataset_conversion.log')
    ]
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif'}
TEXT_EXTENSIONS = {'.txt'}

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False
CHECKPOINT_INTERVAL = 50000  # Save progress every N samples


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global SHUTDOWN_REQUESTED
    logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
    SHUTDOWN_REQUESTED = True


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def compute_sha256(data: bytes) -> str:
    """Compute the sha256 hash of data."""
    m = hashlib.sha256()
    m.update(data)
    return m.hexdigest()


def find_image_text_pairs(input_dir: str) -> List[Tuple[str, str]]:
    """
    Find all paired image and text files in the input directory.
    Returns list of (image_path, text_path) tuples.
    """
    pairs = []
    input_path = Path(input_dir)
    
    logger.info(f"Scanning {input_dir} for image-text pairs...")
    
    for root, _, files in os.walk(input_path):
        # Find all images in this directory
        image_files = {}
        text_files = {}
        
        for file in files:
            file_path = os.path.join(root, file)
            ext = Path(file).suffix.lower()
            stem = Path(file).stem
            
            if ext in IMAGE_EXTENSIONS:
                image_files[stem] = file_path
            elif ext in TEXT_EXTENSIONS:
                text_files[stem] = file_path
        
        # Match pairs by filename stem
        for stem, img_path in image_files.items():
            if stem in text_files:
                pairs.append((img_path, text_files[stem]))
    
    return pairs


def get_subdir_for_shard(shard_idx: int, max_per_subdir: int = 1000) -> str:
    """
    Get the subdirectory name for a given shard index.
    E.g., shard 0-999 -> "1000", shard 1000-1999 -> "2000", etc.
    """
    subdir_idx = (shard_idx // max_per_subdir) + 1
    return str(subdir_idx * max_per_subdir)


def process_sample(image_path: str, text_path: str, idx: int) -> Optional[dict]:
    """
    Process a single image-text pair and return a webdataset sample.
    Returns None if files cannot be read.
    """
    try:
        # Read image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Read text
        with open(text_path, 'r', encoding='utf-8') as f:
            text_data = f.read()
    except Exception as e:
        logger.warning(f"Failed to read {image_path} or {text_path}: {e}")
        return None

    # Compute sha256 hash of image
    sha256_hash = compute_sha256(image_data)

    # Determine the image extension (without the dot)
    ext = Path(image_path).suffix.lower().lstrip('.')
    if ext == 'jpeg':
        ext = 'jpg'

    # Create sample key using index (zero-padded)
    key = f"{idx:09d}"

    # Build the sample
    sample = {
        "__key__": key,
        ext: image_data,
        "txt": text_data.encode('utf-8'),
        "json": json.dumps({"sha256": sha256_hash, "caption": text_data[:100] + "..." if len(text_data) > 100 else text_data}).encode('utf-8'),
    }

    return sample


class ResumableShardWriter:
    """
    A shard writer that supports resumable writing with state tracking.
    """
    def __init__(
        self,
        output_dir: str,
        max_count: int,
        max_per_subdir: int = 1000,
        rank: int = 0,
        world_size: int = 1,
        resume_from_shard: int = 0,
    ):
        self.output_dir = output_dir
        self.max_count = max_count
        self.max_per_subdir = max_per_subdir
        self.rank = rank
        self.world_size = world_size
        self.resume_from_shard = resume_from_shard

        self._shard_idx = max(rank, resume_from_shard)  # Start at resume point
        self._sample_count = 0
        self._writer = None
        self.samples_written = 0
        self.shards_written = 0

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
            self.shards_written += 1

        shard_path = self._get_current_shard_path()
        self._writer = wds.TarWriter(shard_path)
        self._sample_count = 0
        logger.info(f"[Rank {self.rank}] Opened shard: {shard_path}")

    def write(self, sample: dict):
        """Write a sample to the current shard."""
        # Open new shard if needed
        if self._writer is None or self._sample_count >= self.max_count:
            if self._writer is not None:
                self._shard_idx += self.world_size
            self._open_new_shard()

        self._writer.write(sample)
        self._sample_count += 1
        self.samples_written += 1

    def close(self):
        """Close the current writer."""
        if self._writer is not None:
            self._writer.close()
            self.shards_written += 1
            self._writer = None

    def get_state(self) -> dict:
        """Get current state for resume."""
        return {
            'shard_idx': self._shard_idx,
            'samples_written': self.samples_written,
            'shards_written': self.shards_written,
        }


class ConversionState:
    """
    Manages the state of the conversion process for resumability.
    """
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.completed_samples: Set[int] = set()
        self.completed_shards: Set[int] = set()
        self.total_samples = 0
        self.shuffled_indices: List[int] = []
        self.last_update = datetime.now()
        
    def save(self):
        """Save state to disk."""
        state = {
            'completed_samples': self.completed_samples,
            'completed_shards': self.completed_shards,
            'total_samples': self.total_samples,
            'shuffled_indices': self.shuffled_indices,
            'last_update': datetime.now().isoformat(),
        }
        temp_file = self.state_file + '.tmp'
        with open(temp_file, 'wb') as f:
            pickle.dump(state, f)
        os.rename(temp_file, self.state_file)
        logger.info(f"State saved to {self.state_file}")
        
    def load(self) -> bool:
        """Load state from disk. Returns True if successful."""
        if not os.path.exists(self.state_file):
            return False
        try:
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
            self.completed_samples = state['completed_samples']
            self.completed_shards = state['completed_shards']
            self.total_samples = state['total_samples']
            self.shuffled_indices = state['shuffled_indices']
            logger.info(f"State loaded from {self.state_file}")
            logger.info(f"Resuming: {len(self.completed_samples)}/{self.total_samples} samples completed")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False


def worker_process(
    rank: int,
    world_size: int,
    work_queue: Queue,
    output_dir: str,
    shard_size: int,
    max_per_subdir: int,
    progress_queue: Queue,
    resume_shards: Set[int],
):
    """
    Worker process that pulls image-text pairs from queue and writes to shards.
    """
    # Calculate starting shard index for this worker, skipping completed shards
    start_shard = rank
    while start_shard in resume_shards:
        start_shard += world_size
    
    writer = ResumableShardWriter(
        output_dir=output_dir,
        max_count=shard_size,
        max_per_subdir=max_per_subdir,
        rank=rank,
        world_size=world_size,
        resume_from_shard=start_shard,
    )

    try:
        while not SHUTDOWN_REQUESTED:
            try:
                item = work_queue.get(timeout=1)
            except Empty:
                break

            idx, (image_path, text_path) = item
            sample = process_sample(image_path, text_path, idx)

            if sample is not None:
                writer.write(sample)
                progress_queue.put(('success', idx, writer._shard_idx))
            else:
                progress_queue.put(('failed', idx, -1))
    except Exception as e:
        logger.error(f"[Rank {rank}] Worker error: {e}")
        progress_queue.put(('error', str(e)))
    finally:
        writer.close()
        progress_queue.put(('done', rank, writer.get_state()))


def main():
    parser = argparse.ArgumentParser(
        description='Construct a resumable webdataset from image-text pairs.'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='The input directory containing paired image and text files.'
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
    parser.add_argument(
        '--state-file',
        type=str,
        default='conversion_state.pkl',
        help='Path to state file for resume. Default: conversion_state.pkl'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=50000,
        help='Save state every N samples. Default: 50000'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous state if available'
    )

    args = parser.parse_args()

    # Initialize state manager
    state = ConversionState(args.state_file)
    
    # Try to load previous state if resume flag is set
    if args.resume and state.load():
        logger.info("Resuming from previous state...")
    else:
        logger.info("Starting fresh conversion...")

    # Determine number of workers
    num_workers = args.workers if args.workers is not None else cpu_count()
    num_workers = max(1, num_workers)

    # Find all image-text pairs
    if not state.shuffled_indices:
        logger.info(f"Scanning {args.input} for image-text pairs...")
        pairs = find_image_text_pairs(args.input)
        logger.info(f"Found {len(pairs)} image-text pairs.")

        if len(pairs) == 0:
            logger.error("No paired image-text files found. Exiting.")
            sys.exit(1)

        # Shuffle pairs
        logger.info("Shuffling pairs...")
        random.seed(args.seed)
        indices = list(range(len(pairs)))
        random.shuffle(indices)
        
        state.total_samples = len(pairs)
        state.shuffled_indices = indices
        state.save()
    else:
        # Reconstruct pairs from previous run
        logger.info("Reconstructing pairs list from previous state...")
        pairs = find_image_text_pairs(args.input)
        if len(pairs) != state.total_samples:
            logger.warning(f"Warning: found {len(pairs)} pairs, but state expects {state.total_samples}")
            state.total_samples = len(pairs)

    # Calculate expected number of shards
    num_shards = (state.total_samples + args.shard_size - 1) // args.shard_size
    num_subdirs = (num_shards + args.max_per_subdir - 1) // args.max_per_subdir
    logger.info(f"Will create ~{num_shards} shards across ~{num_subdirs} subdirectories.")
    logger.info(f"Using {num_workers} worker processes.")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create work queue with only incomplete samples
    work_queue = Queue()
    remaining_indices = [i for i in state.shuffled_indices if i not in state.completed_samples]
    logger.info(f"Remaining samples to process: {len(remaining_indices)}")
    
    for idx in remaining_indices:
        work_queue.put((idx, pairs[idx]))

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
                state.completed_shards,
            ),
        )
        p.start()
        workers.append(p)

    # Track progress with tqdm
    completed_workers = 0
    successful_images = len(state.completed_samples)
    failed_images = 0
    last_checkpoint = successful_images

    with tqdm(total=state.total_samples, initial=successful_images, desc="Processing samples") as pbar:
        while completed_workers < num_workers and not SHUTDOWN_REQUESTED:
            try:
                result = progress_queue.get(timeout=5)
                msg_type = result[0]
                
                if msg_type == 'done':
                    # Worker finished
                    rank, worker_state = result[1], result[2]
                    logger.info(f"Worker {rank} finished. Wrote {worker_state['samples_written']} samples to {worker_state['shards_written']} shards")
                    completed_workers += 1
                elif msg_type == 'error':
                    logger.error(f"Worker error: {result[1]}")
                elif msg_type == 'success':
                    idx, shard_idx = result[1], result[2]
                    state.completed_samples.add(idx)
                    state.completed_shards.add(shard_idx)
                    successful_images += 1
                    pbar.update(1)
                elif msg_type == 'failed':
                    failed_images += 1
                    pbar.update(1)
                    
                # Checkpoint periodically
                if successful_images - last_checkpoint >= args.checkpoint_interval:
                    logger.info(f"Checkpoint: {successful_images}/{state.total_samples} samples completed")
                    state.save()
                    last_checkpoint = successful_images
                    
            except Empty:
                # Check if workers are still alive
                alive = sum(1 for p in workers if p.is_alive())
                if alive == 0 and work_queue.empty():
                    break
                
                # Save state periodically even without new messages
                if datetime.now().timestamp() % 60 < 5:  # Roughly every minute
                    state.save()

    # Final save
    state.save()

    # Wait for all workers to finish
    for p in workers:
        p.join()

    logger.info(f"\nDone! Processed {successful_images} samples successfully.")
    if failed_images > 0:
        logger.warning(f"Warning: {failed_images} samples failed to process.")
        
    logger.info(f"State saved to {args.state_file} - keep this file for resume capability")
    
    if successful_images >= state.total_samples:
        logger.info("All samples completed! You can delete the state file.")
        if os.path.exists(args.state_file):
            os.remove(args.state_file)
            logger.info(f"Removed state file: {args.state_file}")


if __name__ == '__main__':
    main()
