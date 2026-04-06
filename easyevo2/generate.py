"""DNA sequence generation using Evo2."""

from pathlib import Path
from typing import Annotated

import typer

from easyevo2.dataloader import get_seq_from_fx
from easyevo2.model import ModelType, load_model
from easyevo2.utils import check_cuda, log


def generate(
    prompt: Annotated[
        str | None,
        typer.Option(help="Literal DNA sequence prompt."),
    ] = None,
    prompt_file: Annotated[
        Path | None,
        typer.Option(help="FASTA file containing prompt sequences."),
    ] = None,
    species: Annotated[
        str | None,
        typer.Option(
            help="Species name for phylogenetic tag prompting (e.g., 'Homo sapiens').",
        ),
    ] = None,
    n_tokens: Annotated[
        int,
        typer.Option(help="Number of tokens to generate per prompt."),
    ] = 1000,
    temperature: Annotated[
        float,
        typer.Option(help="Sampling temperature (higher = more diverse)."),
    ] = 1.0,
    top_k: Annotated[
        int,
        typer.Option(help="Top-k sampling parameter."),
    ] = 4,
    model_type: Annotated[
        ModelType,
        typer.Option(help="Model type to use."),
    ] = ModelType.evo2_7b,
    device: Annotated[
        str,
        typer.Option(help="Device to run the model on (e.g., 'cuda:0' or 'cpu')."),
    ] = "cuda:0",
    output: Annotated[
        Path | None,
        typer.Option(help="Output FASTA file. Prints to stdout if not specified."),
    ] = None,
) -> None:
    """
    Generate DNA sequences using Evo2.

    Provide exactly one prompt source: ``--prompt`` (literal sequence),
    ``--prompt-file`` (FASTA), or ``--species`` (phylogenetic tag from GBIF).

    Species-tagged generation uses the phylogenetic lineage of the given
    organism to condition generation, as described in the Evo2 paper.
    """
    # Validate that exactly one prompt source is provided
    sources = [prompt is not None, prompt_file is not None, species is not None]
    if sum(sources) != 1:
        msg = "Provide exactly one of --prompt, --prompt-file, or --species"
        raise typer.BadParameter(msg)

    check_cuda(device)

    # Build prompt list
    prompts: list[str] = []
    names: list[str] = []

    if prompt is not None:
        prompts = [prompt]
        names = ["prompt_0"]

    elif prompt_file is not None:
        for name, seq, *_rest in get_seq_from_fx(prompt_file):
            prompts.append(seq)
            names.append(name)
        if not prompts:
            msg = f"No sequences found in {prompt_file}"
            raise typer.BadParameter(msg)

    elif species is not None:
        from evo2.utils import make_phylotag_from_gbif

        tag = make_phylotag_from_gbif(species)
        log.info(f"Phylogenetic tag for '{species}': {tag}")
        prompts = [tag]
        names = [species.replace(" ", "_")]

    # Load model
    log.info(f"Loading model {model_type.value}")
    model = load_model(model_type)

    # Generate
    log.info(
        f"Generating {n_tokens} tokens for {len(prompts)} prompt(s) "
        f"(temperature={temperature}, top_k={top_k})"
    )
    result = model.generate(
        prompts,
        n_tokens=n_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    generated_seqs = result.sequences

    # Write output
    lines: list[str] = []
    for name, seq in zip(names, generated_seqs, strict=True):
        lines.append(f">{name}_generated_n{n_tokens}\n{seq}\n")

    output_text = "".join(lines)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(output_text)
        log.info(f"Generated sequences saved to {output}")
    else:
        typer.echo(output_text.rstrip())
