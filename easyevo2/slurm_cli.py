"""CLI command for generating SLURM job scripts."""

from typing import Annotated

import typer

from easyevo2.slurm import create_slurm_script
from easyevo2.utils import log


def slurm(
    job_name: Annotated[str, typer.Option(help="SLURM job name.")],
    partition: Annotated[str, typer.Option(help="Partition name.")],
    time: Annotated[str, typer.Option(help="Time limit (e.g., '4:00:00').")],
    command: Annotated[str, typer.Option(help="Command to execute in the job.")],
    gpu_count: Annotated[int, typer.Option(help="Number of GPUs to request.")] = 0,
    gpu_type: Annotated[
        str, typer.Option(help="GPU type (e.g., 'a100', 'v100').")
    ] = "",
    memory: Annotated[
        str, typer.Option(help="Memory requirement (e.g., '64G').")
    ] = "64G",
    cpus_per_task: Annotated[int, typer.Option(help="Number of CPUs per task.")] = 4,
    nodes: Annotated[int, typer.Option(help="Number of nodes.")] = 1,
    conda_env: Annotated[str, typer.Option(help="Conda environment to activate.")] = "",
    account: Annotated[str, typer.Option(help="Account to charge the job to.")] = "",
    email: Annotated[str, typer.Option(help="Email for job notifications.")] = "",
) -> None:
    """Generate a SLURM job script for HPC submission."""
    script_path = create_slurm_script(
        job_name=job_name,
        output_file=f"{job_name}_%j.out",
        error_file=f"{job_name}_%j.err",
        partition=partition,
        time_limit=time,
        nodes=nodes,
        cpus_per_task=cpus_per_task,
        memory=memory,
        command=command,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        conda_env=conda_env,
        account=account,
        email=email,
        email_type="BEGIN,END,FAIL" if email else "NONE",
    )
    log.info(f"SLURM script written to {script_path}")
