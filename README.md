# slurm-repr-extractor

A distributed speech representation extraction system for Indic ASR (Automatic Speech Recognition) using SLURM clusters with GPU acceleration.

## Overview

`slurm-repr-extractor` processes large-scale audio datasets across multiple GPUs using SLURM job scheduling. It extracts audio representations using efficient ONNX-based Indic ASR models and saves learned representations as PyTorch tensors for downstream tasks like speaker diarization and speaker verification.

**Key Features:**
- 🚀 Distributed inference across multi-GPU SLURM clusters
- ⚡ Efficient ONNX-based encoder inference
- 🔄 Resume-safe processing (automatically skips completed samples)
- 📊 Support for multiple Indic languages (Hindi, Marathi, Tamil, Telugu)
- 📝 Batch processing with automatic audio resampling
- 🛡️ Comprehensive error handling and logging
- 💾 Per-utterance representation saving with metadata

## Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU inference)
- A SLURM cluster (or single machine with CUDA)

### Setup

```bash
# Clone the repository
git clone https://github.com/BharatGen-Tech/slurm-repr-extractor.git
cd slurm-repr-extractor

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Directory Structure

```
slurm-repr-extractor/
├── scripts/
│   ├── indic_conformer.py                    # IndicASR model wrapper
│   ├── recompute_representations_slurm_logging.py  # Main distributed inference
│   └── slurm/
│       ├── compute_reps_slurm.sbatch        # 6-GPU SLURM submission script
│       └── as_compute_reps_slurm.sbatch     # 4-GPU SLURM submission script
├── utils/
│   └── missing_data.py                      # Data validation utility
├── requirements.txt
├── README.md
└── LICENSE
```

### Input Format

Prepare a JSONL file where each line is a JSON object:

```json
{
  "source": "/path/to/audio.wav",
  "pt_path": "/path/to/output.pt",
  "utterance_id": "utt_001",
  "language": "hi"
}
```

### Basic Usage (Single Machine)

```bash
cd scripts
python recompute_representations_slurm_logging.py
```

### SLURM Cluster Usage

Submit a job using the provided SLURM templates:

```bash
# 6-GPU job
sbatch scripts/slurm/compute_reps_slurm.sbatch

# 4-GPU job
sbatch scripts/slurm/as_compute_reps_slurm.sbatch
```

The scripts will:
1. Automatically detect SLURM rank and distribute data
2. Load audio and batch process
3. Run inference on assigned GPU
4. Save representations to disk
5. Write metadata to rank-specific JSONL files

### Data Validation

Validate your input JSONL file before processing:

```bash
python utils/missing_data.py
```

This will check if all audio paths in the JSONL file exist and report missing files.

## Configuration

Edit these constants in `scripts/recompute_representations_slurm_logging.py`:

```python
ASSETS_TS_FOLDER = "/path/to/model/assets"  # Path to TorchScript assets
TRAIN_JSONL = "/path/to/input.jsonl"        # Input metadata file
SAVE_BASE = "/path/to/outputs"              # Output directory for .pt files
BATCH_SIZE = 4                              # Audio batch size per GPU
SAMPLE_RATE = 16000                         # Target sample rate (Hz)
```

## Model Components

The `IndicASRModel` uses:
- **Preprocessor** (TorchScript): Audio preprocessing and MFCC extraction
- **Encoder** (ONNX): Efficient acoustic representation extraction

Supported model formats:
- ONNX for encoder inference (GPU-optimized)
- TorchScript for preprocessing

## Output Format

For each input sample, a PyTorch `.pt` file is saved with:

```python
{
    "encoded": torch.Tensor,          # Encoded representations [1, T, D]
    "encoded_len": torch.Tensor       # Length of encoded sequence
}
```

Associated metadata JSONL includes:
```json
{
  "source": "/path/to/audio.wav",
  "pt_path": "/path/to/output.pt",
  "encoder_len": 123,
  "utterance_id": "utt_001",
  "language": "hi"
}
```

## Troubleshooting

### CUDA Device Issues
```python
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Loading Failures
Ensure `ASSETS_TS_FOLDER` points to valid TorchScript and ONNX assets:
- `{ASSETS_TS_FOLDER}/assets/preprocessor.ts`
- `{ASSETS_TS_FOLDER}/assets/encoder.onnx`

### Out of Memory
Reduce `BATCH_SIZE` or increase `CLEANUP_EVERY_N_BATCHES` for more frequent GPU cleanup.

## License

MIT License - see [LICENSE](LICENSE) file for details.