import gzip
from pathlib import Path
from typing import Any

import pyfastx


class ExonTransform:
    """Extract exon sequence of Gene from GTF file and save as FASTA file."""

    def __init__(self, gtf_file: str | Path, fasta_file: str | Path):
        self.gtf_file = Path(gtf_file)
        self.fasta_file = Path(fasta_file)
        self.fasta = pyfastx.Fasta(str(self.fasta_file))

    def _parse_gtf_line(self, line: str) -> dict[str, Any]:
        """Parse a GTF line and extract attributes."""
        if line.startswith("#"):
            return {}

        parts = line.strip().split("\t")
        if len(parts) < 9:
            return {}

        # Parse attributes (9th column)
        attributes = {}
        attr_str = parts[8]
        for attr in attr_str.split(";"):
            attr = attr.strip()
            if " " in attr:
                key, value = attr.split(" ", 1)
                # Remove quotes from value
                value = value.strip('"')
                attributes[key] = value

        return {
            "seqname": parts[0],
            "source": parts[1],
            "feature": parts[2],
            "start": int(parts[3]),
            "end": int(parts[4]),
            "score": parts[5],
            "strand": parts[6],
            "frame": parts[7],
            "attributes": attributes,
        }

    def _get_exons_for_gene(self, gene_id: str) -> list[tuple[str, int, int, str]]:
        """Extract exon coordinates for a given gene ID."""
        exons = []

        # Determine if file is gzipped
        open_func = gzip.open if self.gtf_file.suffix == ".gz" else open
        mode = "rt" if self.gtf_file.suffix == ".gz" else "r"

        with open_func(self.gtf_file, mode) as f:
            for line in f:
                # Ensure line is string
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                parsed = self._parse_gtf_line(line)
                if not parsed:
                    continue

                # Check if this is an exon for our gene
                if (
                    parsed["feature"] == "exon"
                    and parsed["attributes"].get("gene_id") == gene_id
                ):
                    exons.append(
                        (
                            parsed["seqname"],
                            parsed["start"],
                            parsed["end"],
                            parsed["strand"],
                        )
                    )

        return exons

    def _extract_sequence_with_flanking(
        self, chrom: str, start: int, end: int, strand: str, flanking_length: int
    ) -> str:
        """Extract sequence with flanking regions."""
        # Adjust coordinates for flanking regions
        flanked_start = max(1, start - flanking_length)
        flanked_end = end + flanking_length

        # Get sequence from FASTA
        try:
            sequence = str(self.fasta[chrom][flanked_start - 1 : flanked_end])
        except KeyError:
            # Try alternative chromosome names
            alt_names = [f"chr{chrom}", chrom.replace("chr", "")]
            sequence = None
            for alt_name in alt_names:
                try:
                    sequence = str(
                        self.fasta[alt_name][flanked_start - 1 : flanked_end]
                    )
                    break
                except KeyError:
                    continue

            if sequence is None:
                msg = f"Chromosome {chrom} not found in FASTA file"
                raise ValueError(msg) from None

        # Reverse complement if on negative strand
        if strand == "-":
            sequence = self._reverse_complement(sequence)

        return sequence

    def _reverse_complement(self, sequence: str) -> str:
        """Generate reverse complement of DNA sequence."""
        complement = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C",
            "N": "N",
            "a": "t",
            "t": "a",
            "c": "g",
            "g": "c",
            "n": "n",
        }
        return "".join(complement.get(base, base) for base in reversed(sequence))

    def transform(self, gene_id: str, flanking_length: int = 100) -> list[str]:
        """Extract exon sequence of Gene from GTF file and save as FASTA file."""
        # Get exon coordinates for the gene
        exons = self._get_exons_for_gene(gene_id)

        if not exons:
            msg = f"No exons found for gene {gene_id}"
            raise ValueError(msg)

        # Extract sequences for each exon
        exon_sequences = []
        for _i, (chrom, start, end, strand) in enumerate(exons):
            sequence = self._extract_sequence_with_flanking(
                chrom, start, end, strand, flanking_length
            )
            exon_sequences.append(sequence)
        return exon_sequences

    def transform_to_fasta(
        self, gene_id: str, output_file: str | Path, flanking_length: int = 100
    ) -> None:
        """Extract exon sequences and save to FASTA file."""
        exon_sequences = self.transform(gene_id, flanking_length)

        output_file = Path(output_file)
        with output_file.open("w") as f:
            for i, sequence in enumerate(exon_sequences):
                f.write(f">{gene_id}_exon_{i + 1}\n")
                f.write(f"{sequence}\n")

    def __call__(self, gene_id: str, flanking_length: int = 100):
        """Extract exon sequence of Gene from GTF file and save as FASTA file."""
        return self.transform(gene_id, flanking_length)
