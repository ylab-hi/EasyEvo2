# EasyEvo2

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/release/python-3110/)
[![pypi](https://img.shields.io/pypi/v/easyevo2.svg?style=for-the-badge)](https://pypi.org/project/easyevo2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A Python toolkit for easily using Evo2 models in bioinformatics workflows, particularly in HPC environments.

## Description

EasyEvo2 provides a simplified interface to Evo2 foundation models for sequence embedding.
It enables biologists and bioinformaticians to efficiently extract embeddings from DNA, RNA, or protein sequences without extensive deep learning expertise. It's specially designed to work well in High-Performance Computing (HPC) environments.

# Installation

```bash
# Install from PyPI
pip install easyevo2

# Or install from source
git clone https://github.com/ylab-hi/EasyEvo2.git
cd EasyEvo2
pip install .
```

## Usage

### Basic Usage

```bash
# Embed sequences from a FASTA/FASTQ file using the default model (evo2_7b)
easyevo2 embed input.fa

# Specify a different model and specific layer
easyevo2 embed input.fa --model-type evo2_40b --layer-name blocks.28.mlp.l3

# Specify a different model and multiple layers
easyevo2 embed input.fa --model-type evo2_40b --layer-name blocks.28.mlp.l3 blocks.28.mlp.l2

# Save to a specific output file
easyevo2 embed input.fa --output my_embeddings
```

The output will be a safetensor file containing the embeddings for each sequence in the input file.
We can load the embeddings using the `load_tensor` function:

```python
from easyevo2.io import load_tensor

embeddings = load_tensor("my_embeddings.mode.layer.safetensors")
print(embeddings)
# Output: {
# "seq1": torch.tensor([...]),
# "seq2": torch.tensor([...]),
# }
```

### Evo2 Memory Estimates

| Model         | GPU Memory Usage | Embedding Dimension | Batch Size |
| ------------- | ---------------- | ------------------- | ---------- |
| Evo2 1B Base  | 1.5 GB           | 2048                | 1          |
| Evo2 7B       | 15 GB            | 4096                | 1          |
| Evo2 40B Base | >80 GB\*         | --                  | 1          |
| Evo2 40B      | >80 GB\*         | --                  | 1          |

\* Estimated based on scaling from other models

**Notes:**

- Longer sequences require proportionally more memory
- H100 GPUs (80GB) can accommodate the 7B model with batch size 1 but may struggle with the 40B model

## Development

This project uses a Makefile to automate common development tasks:

```bash
# Show available commands
make help

# Run tests
make test

# Lint code
make lint

# Format code
make format

# Build package
make build
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
