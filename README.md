# EasyEvo2

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/release/python-3110/)
[![PyPI](https://img.shields.io/pypi/v/easyevo2.svg?style=for-the-badge)](https://pypi.org/project/easyevo2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A command-line toolkit for genomic sequence analysis with
[Evo 2](https://github.com/ArcInstitute/evo2) and other biological foundation
models.

## Why EasyEvo2?

[Evo 2](https://www.nature.com/articles/s41586-026-10176-5) is a powerful DNA
foundation model, but installing it is notoriously difficult (CUDA 12.1+,
cuDNN 9.3+, transformer-engine, flash-attn) and using it requires substantial
boilerplate code for every task.

EasyEvo2 solves both problems:

- **Pre-built Docker/Singularity images** — no local installation of CUDA,
  transformer-engine, or flash-attn required.
- **Single-command workflows** — variant effect prediction, embedding extraction,
  sequence generation, and more from standard bioinformatics file formats.

## Installation

### Docker (Recommended)

The easiest way to use EasyEvo2. No Python, CUDA, or evo2 installation required.

**Option A — Wrapper script (acts like a native command):**

```bash
# Download the wrapper script (one-time)
curl -fsSL https://raw.githubusercontent.com/ylab-hi/EasyEvo2/main/easyevo2-docker \
  -o ~/.local/bin/easyevo2 && chmod +x ~/.local/bin/easyevo2

# Use it like any CLI tool — files in your current directory are accessible
easyevo2 list-models
easyevo2 vep variants.vcf reference.fa --model-type evo2_7b
easyevo2 embed sequences.fa --output embeddings
```

The wrapper mounts your current directory into the container, so all file
paths work relative to where you run the command. Models are cached in
`~/.cache/huggingface` and persist across runs.

**Option B — Docker directly:**

```bash
docker pull ghcr.io/ylab-hi/easyevo2:latest

docker run --rm --gpus all --shm-size=16g \
  -v ~/.cache/huggingface:/app/.cache/huggingface \
  -v "$(pwd)":/data -w /data \
  ghcr.io/ylab-hi/easyevo2:latest vep variants.vcf reference.fa
```

**Image variants:**

| Image | Models | GPU Requirement | Size |
|-------|--------|-----------------|------|
| `ghcr.io/ylab-hi/easyevo2:latest` | 7B (all 7B variants) | Any CUDA GPU | ~10 GB |
| `ghcr.io/ylab-hi/easyevo2:full` | All (1B, 7B, 20B, 40B) | Hopper GPU (H100/H200) | ~25 GB |

Switch images with the `EASYEVO2_IMAGE` environment variable:

```bash
# Use the full image for 20B/40B models
EASYEVO2_IMAGE=ghcr.io/ylab-hi/easyevo2:full easyevo2 embed seqs.fa --model-type evo2_20b
```

### Singularity / Apptainer (for HPC)

Most HPC clusters use Singularity or Apptainer instead of Docker. EasyEvo2
images can be pulled directly:

```bash
# Pull once (creates a .sif file — do this on a login node, not in a job)
# Set temp/cache to scratch if /tmp is small on your cluster
export APPTAINER_TMPDIR=/scratch/$USER/tmp
export APPTAINER_CACHEDIR=/scratch/$USER/cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

singularity pull easyevo2.sif docker://ghcr.io/ylab-hi/easyevo2:latest
```

```bash
# Run — your current directory and home are auto-mounted
singularity run --nv easyevo2.sif list-models
singularity run --nv easyevo2.sif vep variants.vcf reference.fa --model-type evo2_7b
singularity run --nv easyevo2.sif embed sequences.fa --output embeddings

# Or use exec (equivalent, more explicit)
singularity exec --nv easyevo2.sif easyevo2 vep variants.vcf reference.fa
```

**Example SLURM job script with Singularity:**

```bash
#!/bin/bash
#SBATCH --job-name=easyevo2-vep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G

singularity run --nv /projects/containers/easyevo2.sif \
  vep variants.vcf reference.fa --model-type evo2_7b --output results.tsv
```

**Notes for HPC users:**

- Use `--nv` for GPU passthrough (equivalent to Docker's `--gpus all`)
- Singularity auto-mounts `$HOME` and `$PWD` — no `-v` flags needed
- The `.sif` file is read-only and can be shared among all lab members
- For a clean environment, use `--contain --bind /path/to/data:/data`
- Set `APPTAINERENV_HF_HOME=/scratch/$USER/hf_cache` to cache models on scratch

### pip install (advanced)

Requires a working [evo2 installation](https://github.com/ArcInstitute/evo2#installation)
(CUDA 12.1+, cuDNN 9.3+, transformer-engine, flash-attn).

```bash
pip install easyevo2
```

### From source (for development)

```bash
git clone https://github.com/ylab-hi/EasyEvo2.git
cd EasyEvo2
uv sync -U
```

## Quick Start

```bash
easyevo2 list-models
easyevo2 vep variants.vcf reference.fa --model-type evo2_7b
easyevo2 embed sequences.fa --model-type evo2_7b --device cuda:0
easyevo2 generate --prompt ACGTACGT --n-tokens 1000 --model-type evo2_7b
easyevo2 score sequences.fa --model-type evo2_7b --window-size 512 --step-size 256
```

## CLI Commands

### `easyevo2 vep`

Zero-shot variant effect prediction from VCF files. Computes the change in
sequence log-likelihood between reference and alternate alleles, following the
methodology from the Evo 2 paper (Fig. 3). Supports SNVs, insertions,
deletions, and multi-allelic records.

```bash
easyevo2 vep variants.vcf reference.fa --model-type evo2_7b --context-length 8192
```

Output TSV columns: `chrom`, `pos`, `ref`, `alt`, `ref_score`, `alt_score`,
`delta_score`.

### `easyevo2 embed`

Extract sequence embeddings from FASTA/FASTQ files using Evo 2 intermediate
layers.

```bash
easyevo2 embed input.fa --model-type evo2_7b --layer-name blocks.26
easyevo2 embed input.fa --model-type evo2_7b --output embeddings --merge
```

Output is saved in safetensors format:

```python
from easyevo2.io import load_tensor

embeddings = load_tensor("embeddings.evo2_7b.blocks.26.safetensors")
# {"seq1": tensor([...]), "seq2": tensor([...])}
```

### `easyevo2 generate`

Generate DNA sequences with optional species-specific phylogenetic prompting
via GBIF taxonomy tags.

```bash
easyevo2 generate --prompt ACGTACGT --n-tokens 1000
easyevo2 generate --prompt-file prompts.fa --n-tokens 5000 --output generated.fa
easyevo2 generate --species "Homo sapiens" --n-tokens 10000
```

### `easyevo2 score`

Calculate sequence log-likelihoods with configurable sliding windows.

```bash
easyevo2 score sequences.fa --model-type evo2_7b --window-size 512 --step-size 256
```

### `easyevo2 classify-exons`

Classify exonic positions using Evo 2 embeddings and the pre-trained exon
classifier.

```bash
easyevo2 classify-exons positions.bed reference.fa --model-type evo2_7b_base
```

### `easyevo2 slurm`

Generate SLURM job scripts for HPC submission.

```bash
easyevo2 slurm --job-name brca1-vep --partition gpu \
  --time 4:00:00 --gpu-count 1 --gpu-type a100 --memory 64G \
  --command "easyevo2 vep variants.vcf ref.fa --model-type evo2_7b"
```

### `easyevo2 list-models`

List all available Evo 2 model checkpoints.

## Supported Models

| Model | Parameters | Context | GPU Memory | FP8 Required |
|-------|-----------|---------|------------|-------------|
| `evo2_1b_base` | 1B | 8k | ~1.5 GB | Yes |
| `evo2_7b_base` | 7B | 8k | ~15 GB | No |
| `evo2_7b` | 7B | 1M | ~15 GB | No |
| `evo2_7b_262k` | 7B | 262k | ~15 GB | No |
| `evo2_7b_microviridae` | 7B | - | ~15 GB | No |
| `evo2_20b` | 20B | 1M | ~40 GB | Yes |
| `evo2_40b_base` | 40B | 8k | >80 GB | Yes |
| `evo2_40b` | 40B | 1M | >80 GB | Yes |

FP8 models require NVIDIA Hopper GPUs (H100/H200) and the `full` Docker image.

## Building Docker Images Locally

```bash
# Light image (7B models, any CUDA GPU)
make docker-build-light

# Full image (all models, requires Hopper GPU)
make docker-build

# Push full image to GHCR (light is pushed automatically by CI)
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
make docker-push
```

## Development

```bash
uv sync -U        # Install dependencies
make test          # Run tests (fast only)
make test-all      # Run all tests including GPU
make lint          # Check linting and formatting
make format        # Auto-fix formatting
make coverage      # Run tests with coverage
```

## Citation

If you use EasyEvo2 in your research, please cite:

```bibtex
@software{li_easyevo2_2026,
  author = {Li, Yangyang},
  title = {EasyEvo2: A command-line toolkit for genomic sequence analysis with Evo 2},
  year = {2026},
  url = {https://github.com/ylab-hi/EasyEvo2},
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
