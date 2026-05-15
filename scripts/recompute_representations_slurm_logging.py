# SLURM-COMPLIANT ONNX INFERENCE WITH BATCHING AND SAFE RESUME
# ================================================================
# This script:
# 1. Shards JSONL lines across SLURM ranks (NO torch.distributed)
# 2. Loads and batches audio per rank
# 3. Runs batched encoder inference via IndicASRModel.batched_encode
# 4. Saves per-utterance .pt files
# 5. Writes rank-local JSONL outputs (train + valid)
# 6. Is resume-safe (skips already computed .pt files)

import os
import numpy as np
import json
import torch
from tqdm import tqdm
from typing import List
import soundfile as sf
import torchaudio
from indic_conformer import IndicASRConfig, IndicASRModel
import logging
from datetime import datetime
import sys
# =========================
# CONSTANT PATHS
# =========================
ASSETS_TS_FOLDER = "/fsx/ashutosh.adhikari/representations/scripts"
#TRAIN_JSONL = "/fsx/ashutosh.adhikari/representations/scripts/og_data/Ashutosh_Work_DONOT_DELETE/utterance_best_metrics.jsonl"
TRAIN_JSONL = "/fsx/ashutosh.adhikari/representations/data/hi_mr_ta_te/remain.jsonl" 
#TRAIN_JSONL = "/fsx/ashutosh.adhikari/representations/hi_mr_ta_te_1000hr/hi_mr_ta_te_1000hr_errors_zero_bytes.jsonl"
#TRAIN_JSONL = "/fsx/ashutosh.adhikari/representations/scripts/og_data/Ashutosh_Work_DONOT_DELETE/utterance_best_metrics_next3langs_batch2.jsonl"
SAVE_BASE = "/fsx/ashutosh.adhikari/representations/hi_mr_ta_te_1000hr"
LOCAL_JSONL_BASE = "/fsx/ashutosh.adhikari/representations/data/hi_mr_ta_te/local_rank_outputs"
os.makedirs(LOCAL_JSONL_BASE, exist_ok=True)
logger = None



BATCH_SIZE = 4 
SAMPLE_RATE = 16000

# =========================
# SETUP (NO torch.distributed)
# =========================
def setup_inference():
    """
    Setup inference environment from SLURM variables.
    NO torch.distributed initialization needed.
    
    Returns:
        rank: Process rank (0 to world_size-1)
        device: torch.device for this process
        world_size: Total number of parallel processes
    """
    # Get rank/world_size from SLURM environment
    # Falls back to single-process mode if not in SLURM
    rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    # Set CUDA device (important for multi-GPU)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    print(f"[Rank {rank}/{world_size}] Using device: {device} (local_rank={local_rank})")
    
    return rank, device, world_size


# =========================
# AUDIO UTILITIES
# =========================
def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load and resample audio to target sample rate."""
    wav, sr = sf.read(path)
    # Convert to tensor first
    wav = torch.from_numpy(wav).float()

    # Handle stereo -> mono
    if wav.dim() == 2:
        wav = wav.mean(dim=1)

    # Resample if needed using torchaudio
    if sr != target_sr:
        # torchaudio expects [channels, samples]
        wav = wav.unsqueeze(0)
        
        # Resample
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
        
        # Back to 1D
        wav = wav.squeeze(0)

    return wav


def pad_batch(wavs: List[torch.Tensor]):
    """Pad a batch of waveforms to the same length."""
    lengths = torch.tensor([w.size(0) for w in wavs], dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.zeros(len(wavs), max_len)
    
    for i, w in enumerate(wavs):
        padded[i, : w.size(0)] = w
    
    return padded, lengths


# =========================
# SHARDING
# =========================
def shard_lines(lines: List[str], rank: int, world_size: int):
    """
    Yield lines assigned to this rank via round-robin sharding.
    
    Example: 4 lines, rank=1, world_size=2 yields lines[1, 3]
    """
    for i, line in enumerate(lines):
        if i % world_size == rank:
            yield line


# =========================
# CORE PROCESSING
# =========================
@torch.no_grad()
def run_split(jsonl_path: str, split: str, model, device, rank: int, world_size: int):
    """
    Process a dataset split in parallel across ranks.
    
    Each rank:
    - Processes its own shard of data (no inter-rank communication)
    - Writes to rank-specific output files
    - Safely resumes if interrupted
    """
    save_dir = os.path.join(SAVE_BASE, split)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(LOCAL_JSONL_BASE, exist_ok=True)

    # Each rank writes to its own JSONL file
    rank_jsonl = os.path.join(LOCAL_JSONL_BASE, f"{split}_rank{rank}.jsonl")

    # Load all lines
    with open(jsonl_path, "r") as f:
        lines = f.readlines()

    # Get this rank's shard
    shard = list(shard_lines(lines, rank, world_size))
    print(f"[Rank {rank}] Processing {len(shard)} samples for split '{split}'")

    batch_wavs, batch_meta = [], []

    # Process shard with progress bar (only rank 0 shows detailed progress)
    with open(rank_jsonl, "w") as jout:
        iterator = tqdm(shard, desc=f"{split} | rank {rank}") if rank == 0 else shard
        
        for line in iterator:
            sample = json.loads(line)
            pt_path = os.path.join(save_dir, os.path.basename(sample["pt_path"]))

            # Resume-safe skip: if already computed, just write to output
            if os.path.exists(pt_path):
                sample["pt_path"] = pt_path
                jout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

# Load audio
            if os.path.exists(sample["source"]):
                wav = load_audio(sample["source"])
                batch_wavs.append(wav)
                batch_meta.append((sample, pt_path))

    # Process when batch is full
                if len(batch_wavs) == BATCH_SIZE:
                        flush_batch(batch_wavs, batch_meta, model, device, jout)
                        batch_wavs, batch_meta = [], []
            else:
                continue

        # Process remaining samples
        if batch_wavs:
            flush_batch(batch_wavs, batch_meta, model, device, jout)

    print(f"[Rank {rank}] Complete. Output: {rank_jsonl}")

# Add this at the top of your script (after imports, before functions)
BATCH_COUNTER = 0
CLEANUP_EVERY_N_BATCHES = 5
def flush_batch(wavs: List[torch.Tensor], meta: List, model, device, jout):
    """
    Run batch inference and save results.
    
    Args:
        wavs: List of waveform tensors
        meta: List of (sample_dict, pt_path) tuples
        model: IndicASRModel instance
        device: torch.device
        jout: Open file handle for writing JSONL
    """
    global logger, BATCH_COUNTER, CLEANUP_EVERY_N_BATCHES
    logger.debug(f"flush_batch called with {len(wavs)} samples")

    # Pad batch
    logger.debug("Padding batch...")
    padded, lengths = pad_batch(wavs)
    logger.debug(f"Batch padded: shape={padded.shape}, lengths={lengths}")

    padded = padded.to(device)
    lengths = lengths.to(device)
    logger.debug(f"Tensors moved to {device}")

    # Run inference (batched encoding)
    logger.debug("Running model.batched_encode...")
    try:
        enc, enc_lens = model.batched_encode(padded, lengths)
        logger.debug(f"Encoding complete: enc.shape={enc.shape}, enc_lens={enc_lens[:5]}...")
        
        # Check for NaN/Inf in encoded output
        if np.isnan(enc).any():
            nan_indices = np.argwhere(np.isnan(enc))
            affected_samples = np.unique(nan_indices[:, 0])  # Which samples have NaN
            logger.error(f"NaN detected in encoding output for samples: {affected_samples.tolist()}")
            # Mark these samples as failed
            failed_encoding = set(affected_samples.tolist())
        else:
            failed_encoding = set()
            
    except Exception as e:
        logger.error(f"model.batched_encode failed completely: {e}", exc_info=True)
        # Skip this entire batch
        logger.error(f"Skipping entire batch of {len(meta)} samples")
        BATCH_COUNTER += 1
        return  # Exit early, don't try to save

    # Save per-utterance results
    logger.debug(f"Saving {len(meta)} outputs...")
    failed_samples = []

    for i, (sample, pt_path) in enumerate(meta):
        # Skip if encoding failed for this sample
        if i in failed_encoding:
            logger.warning(f"Skipping sample {i} ({sample.get('source', 'unknown')}) - encoding had NaN")
            failed_samples.append((i, sample.get('source', 'unknown'), 'NaN in encoding'))
            continue
        
        try:
            torch.save(
                {
                    "encoded": torch.from_numpy(enc[i, :, :enc_lens[i]]).unsqueeze(0),
                    "encoded_len": torch.tensor([enc_lens[i]]),
                },
                pt_path,
            )
            try:
                               
                # Update sample path and write to JSONL
                sample["pt_path"] = pt_path
                sample["encoder_len"] = int(enc_lens[i])
                jout.write(json.dumps(sample, ensure_ascii=False) + "\n")

                print("jout :",pt_path)
                if i == 0:
                    logger.debug(f"First sample saved: {pt_path} (enc_len={enc_lens[i]})")
             
            except Exception as e:
                logger.error(f"jout error!")
                logger.error(f"failed to save sample {i} ({sample.get('source', 'unknown')}): {e}")
                failed_samples.append((i, sample.get('source', 'unknown'), str(e)))
            # continue processing other samples
             
        except Exception as e:
            logger.error(f"failed to save sample {i} ({sample.get('source', 'unknown')}): {e}")
            failed_samples.append((i, sample.get('source', 'unknown'), str(e)))
            # continue processing other samples
        
    # Log summary of failures
    if failed_samples:
        logger.warning(f"Failed to process {len(failed_samples)}/{len(meta)} samples in this batch")
        for i, source, error in failed_samples[:5]:  # Show first 5
            logger.warning(f"LINCATCH981928  Sample {i} ({source}): {error}")

    # Increment counter
    BATCH_COUNTER += 1
  
    # Cleanup every N batches
    if BATCH_COUNTER % CLEANUP_EVERY_N_BATCHES == 0:
        logger.debug(f"Cleaning up (batch {BATCH_COUNTER})...")
        del enc, enc_lens, padded, lengths
        torch.cuda.empty_cache()
        logger.info(f"GPU memory cleaned after {BATCH_COUNTER} batches")
    else:
        logger.debug(f"Skipping cleanup (batch {BATCH_COUNTER}/{CLEANUP_EVERY_N_BATCHES})")
        del enc, enc_lens, padded, lengths


# =========================
# MAIN
# =========================
def main():
    """Main inference pipeline."""
    
    # Setup (no distributed init needed)
    rank, device, world_size = setup_inference()
    global logger
    # Load model
    print(f"[Rank {rank}] Loading model...")
    config = IndicASRConfig(ts_folder=ASSETS_TS_FOLDER)
    model = IndicASRModel(config, device=device, local_rank=rank % torch.cuda.device_count())
    model.eval()
    print(f"[Rank {rank}] Model loaded")
    # Create rank-specific log file
    os.makedirs(LOCAL_JSONL_BASE+"/log",exist_ok=True)
    log_file = os.path.join(LOCAL_JSONL_BASE+"/log", f"rank{rank}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Write to file
            logging.StreamHandler(sys.stdout)          # Also print to console
        ]
    )

    logger = logging.getLogger(__name__)
    # Process dataset splits
    run_split(TRAIN_JSONL, "main", model, device, rank, world_size)
    # Uncomment to also process validation split:
    # run_split(VALID_JSONL, "valid", model, device, rank, world_size)

    print(f"[Rank {rank}] All done!")


if __name__ == "__main__":
    main()

