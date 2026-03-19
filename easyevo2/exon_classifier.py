from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pyfastx
import torch
import typer
from transformers import AutoModel

from easyevo2.model import ModelType, load_model
from easyevo2.utils import check_cuda, log


class FlankingSequences:
    """Get the flanking sequences from a BED file and a FASTA file."""

    def __init__(self, fasta_file: str | Path):
        self.fasta_file = Path(fasta_file)
        self.fasta = pyfastx.Fasta(str(self.fasta_file))

    def flank(
        self, chrom: str, pos: int, flanking_length: int = 8192
    ) -> tuple[str, str]:
        """
        Get the flanking sequence (upstream of the position).

        Parameters
        ----------
        chrom : str
            The chromosome/sequence name.
        pos : int
            The position of the exon center, 1-based.
        flanking_length : int, optional
            The length of the flanking sequence. Defaults to 8192.

        Returns
        -------
        tuple of (str, str)
            Forward and reverse flanking sequences.
        """
        return self.fasta.flank(chrom, pos, pos, flanking_length)

    def get_flanking_sequences_from_bed(self, bed_file: Path):
        """
        Get flanking sequences from a BED file and a FASTA file.

        The BED file should contain three columns: name, chrom, start.

        Parameters
        ----------
        bed_file : Path
            Path to the BED file.

        Yields
        ------
        tuple of (str, tuple of (str, str))
            Identifier and forward/reverse flanking sequences.
        """
        with bed_file.open("r") as f:
            for line in f:
                name, chrom, start = line.strip().split("\t")
                yield f"{name}_{chrom}_{start}", self.flank(chrom, int(start))


def get_final_token_embedding(
    sequence: str, model: Any, layer_name: str, device: str
) -> np.ndarray:
    """
    Get the final token embedding of a sequence.

    Parameters
    ----------
    sequence : str
        The DNA sequence to embed.
    model : Any
        The Evo2 model instance.
    layer_name : str
        Name of the layer to extract embeddings from.
    device : str
        Device to run inference on.

    Returns
    -------
    numpy.ndarray
        The embedding vector of the final token with shape ``(hidden_dim,)``.
    """
    input_ids = (
        torch.tensor(
            model.tokenizer.tokenize(sequence),
            dtype=torch.int,
        )
        .unsqueeze(0)
        .to(device)
    )
    with torch.inference_mode():
        _, embeddings = model(
            input_ids, return_embeddings=True, layer_names=[layer_name]
        )

    return (
        embeddings[layer_name][0, -1, :].cpu().to(torch.float32).numpy()
    )  # shape: (hidden_dim,)


def classify_exons(
    bed_file: Path,
    fasta_file: Path,
    layer_name: Annotated[
        str | None,
        typer.Option(
            help="Layer name to extract embeddings from.",
        ),
    ] = "blocks.26",
    model_type: Annotated[
        ModelType,
        typer.Option(
            help="Model type to use for embedding.",
        ),
    ] = ModelType.evo2_7b,
    output_file: Annotated[
        Path | None,
        typer.Option(help="Output file to save the exon classifier results."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            help="Device to run the model on (e.g., 'cuda:0' or 'cpu').",
        ),
    ] = "cuda:0",
) -> None:
    """Classify exonic positions using Evo2 embeddings."""
    check_cuda(device)
    model = load_model(model_type)
    log.debug(f"Model state dict keys: {model.model.state_dict().keys()}")

    flanking_sequencer = FlankingSequences(fasta_file)
    flanking_seqs = flanking_sequencer.get_flanking_sequences_from_bed(bed_file)

    embeddings = {}
    for name, (forward_seq, reverse_seq) in flanking_seqs:
        emb_forward = get_final_token_embedding(forward_seq, model, layer_name, device)
        emb_reverse = get_final_token_embedding(reverse_seq, model, layer_name, device)
        emb_concat = np.concatenate([emb_forward, emb_reverse])
        embeddings[name] = {
            "emb_concat": emb_concat,
        }

    exon_classifier = AutoModel.from_pretrained(
        "schmojo/evo2-exon-classifier",
        trust_remote_code=True,
    ).to(device)

    probs = {}
    for name, embedding in embeddings.items():
        embedding_tensor = (
            torch.tensor(embedding["emb_concat"], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(1)
            .to(device)
        )
        with torch.inference_mode():
            probs[name] = exon_classifier(embedding_tensor)["logits"].item()

    if output_file is None:
        output_file = Path(bed_file).with_suffix(".exon_classifier.tsv")

    with output_file.open("w") as f:
        for name, prob in probs.items():
            f.write(f"{name}\t{prob}\n")
