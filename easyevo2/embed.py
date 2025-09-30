from collections import defaultdict
from pathlib import Path
from typing import Annotated

import torch
import typer

from easyevo2.dataloader import get_seq_from_fx
from easyevo2.io import cleanup_individual_files, merge_embedding_files, save_embeddings
from easyevo2.model import ModelType, load_model
from easyevo2.utils import check_cuda, log


def embed(
    filename: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to the FASTA or FASTQ file.",
        ),
    ],
    model_type: Annotated[
        ModelType,
        typer.Option(
            help="Model type to use for embedding.",
        ),
    ] = ModelType.evo2_7b,
    layer_name: Annotated[
        list[str] | None,
        typer.Option(
            help="Layer name to extract embeddings from.",
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            help="Device to run the model on (e.g., 'cuda:0' or 'cpu').",
        ),
    ] = "cuda:0",
    save_interval: Annotated[
        int,
        typer.Option(
            help="Save interval for the embeddings.",
        ),
    ] = 100,
    max_seq_length: Annotated[
        int | None,
        typer.Option(
            help="Maximum sequence length to process. If not provided, the model's max length will be used.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Output file to save the embeddings.",
        ),
    ] = None,
    merge: Annotated[
        bool,
        typer.Option(
            help="Merge the individual embeddings into a single file.",
        ),
    ] = False,
):
    """Embed a FASTA or FASTQ file."""
    # Load the model
    if layer_name is None:
        layer_name = ["blocks.26"]

    check_cuda(device)

    model = load_model(model_type)
    sequences = get_seq_from_fx(
        filename,
    )
    embeddings_with_name = {}
    failing_sequences = []
    embedding_paths = defaultdict(list)

    # Process sequences in batches
    for idx, seq_data in enumerate(sequences):
        name = seq_data[0]
        seq = seq_data[1]

        if max_seq_length is not None and len(seq) > max_seq_length:
            log.warning(
                f"Sequence {name} is longer than {max_seq_length} characters, truncating to {max_seq_length}"
            )
            seq = seq[:max_seq_length]

        try:
            # Tokenize and process the sequence
            input_ids = (
                torch.tensor(
                    model.tokenizer.tokenize(seq),
                    dtype=torch.int,
                )
                .unsqueeze(0)
                .to(device)
            )

            with torch.inference_mode():
                # Get embeddings
                outputs, embeddings = model(
                    input_ids, return_embeddings=True, layer_names=layer_name
                )

                # Move embeddings to CPU and store
                cpu_embeddings = {}
                for layer, tensor in embeddings.items():
                    if layer in layer_name:
                        cpu_embeddings[layer] = tensor.detach().cpu()
                        # Explicitly delete GPU tensor to free memory
                    del tensor

                # Store the embeddings
                embeddings_with_name[name] = cpu_embeddings

                # Clear GPU cache periodically to prevent memory buildup
                if torch.cuda.is_available() and idx % 50 == 0:
                    torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"Error processing sequence {name}: {e}")
            failing_sequences.append(name)
            continue
        finally:
            if "input_ids" in locals():
                del input_ids

        if (idx + 1) % save_interval == 0:
            log.info(f"Saving embeddings for {idx} sequences")

            # Save the embeddings to the output file
            for layer in layer_name:
                metadata = {
                    "model_type": model_type.value,
                    "layer_name": layer,
                    "output": str(output),
                    "sequence_processed": str(idx + 1),
                }

                if output is None:
                    layer_output = Path(filename).with_suffix(
                        f".{model_type}.{layer}.{idx + 1}.safetensors"
                    )
                else:
                    layer_output = Path(output).with_suffix(
                        f".{model_type}.{layer}.{idx + 1}.safetensors"
                    )

                layer_embeddings = {
                    name: embeddings[layer]
                    for name, embeddings in embeddings_with_name.items()
                }

                save_embeddings(
                    layer_embeddings,
                    layer_output,
                    metadata=metadata,
                )
                embedding_paths[layer].append(layer_output)

            log.info(f"Saved embeddings for {idx} sequences")
            # Clear embeddings from memory after saving
            embeddings_with_name.clear()

            # force garbage collection
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save any remaining embeddings
    if embeddings_with_name:
        log.info(f"Saving final {len(embeddings_with_name)} embeddings")

        for layer in layer_name:
            metadata = {
                "model_type": model_type.value,
                "layer_name": layer,
                "output": str(output),
            }

            if output is None:
                layer_output = Path(filename).with_suffix(
                    f".{model_type}.{layer}.final.safetensors"
                )
            else:
                layer_output = Path(output).with_suffix(
                    f".{model_type}.{layer}.final.safetensors"
                )

            layer_embeddings = {
                name: embeddings[layer]
                for name, embeddings in embeddings_with_name.items()
            }

            save_embeddings(
                layer_embeddings,
                layer_output,
                metadata=metadata,
            )
            embedding_paths[layer].append(layer_output)

    try:
        if merge and embedding_paths:
            for layer in layer_name:
                layer_files = embedding_paths[layer]
                merge_output = (
                    Path(filename).with_suffix(f".{model_type}.{layer}.safetensors")
                    if output is None
                    else Path(output).with_suffix(f".{model_type}.{layer}.safetensors")
                )
                metadata = {
                    "model_type": model_type.value,
                    "layer_name": layer,
                    "output": str(merge_output),
                }
                merge_embedding_files(layer_files, merge_output, metadata=metadata)
                cleanup_individual_files(layer_files)
    except Exception as e:
        log.error(f"Error merging embeddings: {e}")
    else:
        log.info("Merged embeddings into a single file")
    finally:
        # Save the failing sequences to a file
        if failing_sequences:
            with Path(f"{filename.stem}.failing_sequences.txt").open("w") as f:
                for seq in failing_sequences:
                    f.write(f"{seq}\n")
            log.warning(f"Failed to process {len(failing_sequences)} sequences")

    # Final cleanup
    del embeddings_with_name
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
