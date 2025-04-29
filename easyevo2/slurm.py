from pathlib import Path


def create_slurm_script(
    job_name: str,
    output_file: str,
    error_file: str,
    partition: str,
    time_limit: str,
    nodes: int = 1,
    tasks_per_node: int = 1,
    cpus_per_task: int = 1,
    memory: str = "4G",
    command: str = "",
    gpu_count: int = 0,
    gpu_type: str = "",
    modules: list | None = None,
    conda_env: str = "",
    email: str = "",
    email_type: str = "NONE",
    account: str = "",
    working_dir: str = "",
) -> str:
    """
    Create an optimized SLURM script for submitting a job to a SLURM scheduler with support for GPU acceleration.

    Args:
            job_name (str): Name of the job.
            output_file (str): Path to the output file.
            error_file (str): Path to the error file.
            partition (str): Partition name.
            time_limit (str): Time limit for the job (e.g., "01:00:00" for 1 hour).
            nodes (int, optional): Number of nodes to use. Defaults to 1.
            tasks_per_node (int, optional): Number of tasks per node. Defaults to 1.
            cpus_per_task (int, optional): Number of CPUs per task. Defaults to 1.
            memory (str, optional): Memory requirement (e.g., "4G"). Defaults to "4G".
            command (str, optional): Command to execute. Defaults to "".
            gpu_count (int, optional): Number of GPUs to request. Defaults to 0.
            gpu_type (str, optional): Type of GPU to request (e.g., "v100", "a100"). Defaults to "".
            modules (list, optional): List of modules to load. Defaults to None.
            conda_env (str, optional): Conda environment to activate. Defaults to "".
            email (str, optional): Email for job notifications. Defaults to "".
            email_type (str, optional): When to send email notifications (e.g., "BEGIN,END,FAIL"). Defaults to "NONE".
            account (str, optional): Account to charge the job to. Defaults to "".
            working_dir (str, optional): Working directory for the job. Defaults to "".

    Returns
    -------
            str: Path to the generated SLURM script.
    """
    # Build SLURM script with core settings
    slurm_script = "#!/bin/bash\n"
    slurm_script += f"#SBATCH --job-name={job_name}\n"
    slurm_script += f"#SBATCH --output={output_file}\n"
    slurm_script += f"#SBATCH --error={error_file}\n"
    slurm_script += f"#SBATCH --partition={partition}\n"
    slurm_script += f"#SBATCH --time={time_limit}\n"
    slurm_script += f"#SBATCH --nodes={nodes}\n"
    slurm_script += f"#SBATCH --ntasks-per-node={tasks_per_node}\n"
    slurm_script += f"#SBATCH --cpus-per-task={cpus_per_task}\n"
    slurm_script += f"#SBATCH --mem={memory}\n"
    slurm_script += "#SBATCH --export=ALL\n"

    # Add optional parameters if provided
    if gpu_count > 0:
        slurm_script += (
            f"#SBATCH --gres=gpu:{gpu_type + ',' if gpu_type else ''}{gpu_count}\n"
        )

    if email and email_type:
        slurm_script += f"#SBATCH --mail-user={email}\n"
        slurm_script += f"#SBATCH --mail-type={email_type}\n"

    if account:
        slurm_script += f"#SBATCH --account={account}\n"

    if working_dir:
        slurm_script += f"#SBATCH --chdir={working_dir}\n"

    # Add module loading section
    slurm_script += "\n# Load necessary modules\n"
    if modules:
        for module in modules:
            slurm_script += f"module load {module}\n"

    # Add conda environment activation if specified
    if conda_env:
        slurm_script += "\n# Activate conda environment\n"
        slurm_script += "source $(conda info --base)/etc/profile.d/conda.sh\n"
        slurm_script += f"conda activate {conda_env}\n"

    # Add performance optimizations
    slurm_script += "\n# Performance optimizations\n"
    if gpu_count > 0:
        slurm_script += "export CUDA_DEVICE_ORDER=PCI_BUS_ID\n"
        slurm_script += "export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS\n"

    # Add command to execute
    slurm_script += "\n# Run the command\n"
    slurm_script += f"{command}\n"

    # Write the script to a file
    script_path = f"{job_name}.slurm"
    with Path(script_path).open("w") as script_file:
        script_file.write(slurm_script)

    return script_path
