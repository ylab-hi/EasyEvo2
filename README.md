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

# Or use self-contain and monolithic installation
wget https://raw.githubusercontent.com/ylab-hi/EasyEvo2/refs/heads/main/easyevo2.pyz
python easyevo2.pyz embed input.fa
```

## Usage

### Basic Usage

```bash
# Embed sequences from a FASTA file using the default model (evo2_7b)
easyevo2 embed input.fa

# Specify a different model
easyevo2 embed input.fa --model-type evo2_40b

# Extract embeddings from a specific layer
easyevo2 embed input.fa --layer-name blocks.28.mlp.l3 --layer-name  blocks.28.mlp.l3

# Use CPU instead of GPU
easyevo2 embed input.fa --device cpu

# Save to a specific output file
easyevo2 embed input.fa --output my_embeddings
```

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
