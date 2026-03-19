"""Zero-shot variant effect prediction using Evo2 log-likelihoods."""

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import pyfastx
import pysam
import typer

from easyevo2.model import ModelType, load_model
from easyevo2.utils import check_cuda, log


def _build_variant_records(
    vcf_path: Path,
    ref_fasta: pyfastx.Fasta,
    context_length: int,
) -> tuple[list[dict], list[str], dict[str, int]]:
    """
    Parse VCF and build ref/alt sequence pairs for scoring.

    Parameters
    ----------
    vcf_path : Path
        Path to the VCF file.
    ref_fasta : pyfastx.Fasta
        Reference genome FASTA.
    context_length : int
        Context window size around each variant in base pairs.

    Returns
    -------
    tuple of (list of dict, list of str, dict of str to int)
        Variant records, unique reference sequences, and mapping from
        reference sequence to its index in the unique list.
    """
    records: list[dict] = []
    unique_ref_seqs: list[str] = []
    ref_seq_to_index: dict[str, int] = {}

    vcf = pysam.VariantFile(str(vcf_path))
    for rec in vcf:
        chrom = rec.chrom
        pos = rec.pos  # 1-based in VCF
        ref_allele = rec.ref
        alt_alleles = rec.alts

        if alt_alleles is None:
            continue

        # Calculate the context window (0-based coordinates for FASTA slicing)
        half_ctx = context_length // 2
        chrom_len = len(ref_fasta[chrom])
        window_start = max(0, pos - 1 - half_ctx)
        window_end = min(chrom_len, window_start + context_length)

        # Position of the variant within the window (0-based)
        var_offset = (pos - 1) - window_start

        # Extract reference context sequence
        ref_seq = str(ref_fasta[chrom][window_start:window_end])

        # Deduplicate reference sequences (many variants share the same window)
        if ref_seq not in ref_seq_to_index:
            ref_seq_to_index[ref_seq] = len(unique_ref_seqs)
            unique_ref_seqs.append(ref_seq)

        ref_idx = ref_seq_to_index[ref_seq]

        # Process each ALT allele separately (handles multi-allelic VCFs)
        for alt_allele in alt_alleles:
            # Construct the alt sequence by replacing ref with alt in the window
            ref_len = len(ref_allele)
            alt_seq = (
                ref_seq[:var_offset] + alt_allele + ref_seq[var_offset + ref_len :]
            )

            records.append(
                {
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref_allele,
                    "alt": alt_allele,
                    "alt_seq": alt_seq,
                    "ref_idx": ref_idx,
                }
            )

    vcf.close()
    return records, unique_ref_seqs, ref_seq_to_index


def vep(
    vcf_file: Annotated[
        Path,
        typer.Argument(help="Input VCF file with variants."),
    ],
    reference: Annotated[
        Path,
        typer.Argument(help="Reference genome FASTA file."),
    ],
    model_type: Annotated[
        ModelType,
        typer.Option(help="Model type to use for scoring."),
    ] = ModelType.evo2_7b,
    context_length: Annotated[
        int,
        typer.Option(
            help="Context window around each variant in bp (default matches Evo2 paper).",
        ),
    ] = 8192,
    device: Annotated[
        str,
        typer.Option(help="Device to run the model on (e.g., 'cuda:0' or 'cpu')."),
    ] = "cuda:0",
    output: Annotated[
        Path | None,
        typer.Option(help="Output TSV file path."),
    ] = None,
) -> None:
    """
    Zero-shot variant effect prediction using Evo2 log-likelihoods.

    Scores each variant by computing the change in sequence log-likelihood
    between the reference and alternate alleles within a context window,
    following the methodology from the Evo2 paper (Brixi et al., Nature 2026).

    Supports SNVs, insertions, deletions, and multi-allelic records.
    """
    check_cuda(device)

    # Load reference genome
    log.info(f"Loading reference genome from {reference}")
    ref_fasta = pyfastx.Fasta(str(reference))

    # Parse VCF and build sequences
    log.info(f"Parsing variants from {vcf_file}")
    records, unique_ref_seqs, _ref_map = _build_variant_records(
        vcf_file, ref_fasta, context_length
    )

    if not records:
        log.warning("No variants found in VCF file")
        return

    log.info(
        f"Found {len(records)} variants with {len(unique_ref_seqs)} unique reference windows"
    )

    # Load model
    log.info(f"Loading model {model_type.value}")
    model = load_model(model_type)

    # Score reference sequences (deduplicated)
    log.info(f"Scoring {len(unique_ref_seqs)} unique reference sequences")
    ref_scores = model.score_sequences(unique_ref_seqs)

    # Score alternate sequences
    alt_seqs = [rec["alt_seq"] for rec in records]
    log.info(f"Scoring {len(alt_seqs)} alternate sequences")
    alt_scores = model.score_sequences(alt_seqs)

    # Build results
    results = []
    for i, rec in enumerate(records):
        ref_score = ref_scores[rec["ref_idx"]]
        alt_score = alt_scores[i]
        delta_score = alt_score - ref_score

        results.append(
            {
                "chrom": rec["chrom"],
                "pos": rec["pos"],
                "ref": rec["ref"],
                "alt": rec["alt"],
                "ref_score": ref_score,
                "alt_score": alt_score,
                "delta_score": delta_score,
            }
        )

    df = pd.DataFrame(results)

    # Prepare output path
    if output is None:
        output = Path(vcf_file).with_suffix(f".vep_{model_type}.tsv")

    df.to_csv(output, sep="\t", index=False)

    # Save metadata
    metadata = {
        "model_type": model_type.value,
        "context_length": context_length,
        "device": device,
        "n_variants": len(records),
        "n_unique_ref_windows": len(unique_ref_seqs),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    metadata_path = output.with_suffix(".metadata.json")
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Results saved to {output}")
    log.info(f"Metadata saved to {metadata_path}")
