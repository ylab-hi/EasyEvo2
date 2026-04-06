import typer

from easyevo2.embed import embed
from easyevo2.exon_classifier import classify_exons
from easyevo2.generate import generate
from easyevo2.list_models import list_models
from easyevo2.score import score
from easyevo2.slurm_cli import slurm
from easyevo2.vep import vep

# Define a command-line interface (CLI) using Typer
app = typer.Typer(
    epilog="EasyEvo2 — easy genomic analysis with Evo2.\n",
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.command(help="Extract sequence embeddings from FASTA/FASTQ files.")(embed)
app.command(help="List all available Evo2 model checkpoints.")(list_models)
app.command(help="Calculate sequence log-likelihoods with sliding windows.")(score)
app.command(help="Zero-shot variant effect prediction from VCF files.")(vep)
app.command(help="Generate DNA sequences with optional species prompting.")(generate)
app.command(help="Classify exonic positions using Evo2 embeddings.")(classify_exons)
app.command(help="Generate SLURM job scripts for HPC submission.")(slurm)

if __name__ == "__main__":
    app()
