from pathlib import Path

from easyevo2.slurm import create_slurm_script


def test_create_basic_script(tmp_path):
    script_path = create_slurm_script(
        job_name=str(tmp_path / "test_job"),
        output_file="test.out",
        error_file="test.err",
        partition="gpu",
        time_limit="01:00:00",
        command="echo hello",
    )
    content = Path(script_path).read_text()
    assert "#!/bin/bash" in content
    assert "#SBATCH --partition=gpu" in content
    assert "#SBATCH --time=01:00:00" in content
    assert "echo hello" in content
    Path(script_path).unlink()


def test_gpu_gres_format_with_type(tmp_path):
    """GPU gres should use colon separator: gpu:a100:1."""
    script_path = create_slurm_script(
        job_name=str(tmp_path / "gpu_job"),
        output_file="test.out",
        error_file="test.err",
        partition="gpu",
        time_limit="01:00:00",
        gpu_count=1,
        gpu_type="a100",
        command="nvidia-smi",
    )
    content = Path(script_path).read_text()
    assert "#SBATCH --gres=gpu:a100:1" in content
    # Must NOT have comma format
    assert "gpu:a100," not in content
    Path(script_path).unlink()


def test_gpu_gres_format_without_type(tmp_path):
    """GPU gres without type: gpu:2."""
    script_path = create_slurm_script(
        job_name=str(tmp_path / "gpu_job2"),
        output_file="test.out",
        error_file="test.err",
        partition="gpu",
        time_limit="01:00:00",
        gpu_count=2,
        command="nvidia-smi",
    )
    content = Path(script_path).read_text()
    assert "#SBATCH --gres=gpu:2" in content
    Path(script_path).unlink()


def test_modules_loading(tmp_path):
    script_path = create_slurm_script(
        job_name=str(tmp_path / "mod_job"),
        output_file="test.out",
        error_file="test.err",
        partition="compute",
        time_limit="00:30:00",
        modules=["cuda/12.0", "python/3.11"],
        command="python script.py",
    )
    content = Path(script_path).read_text()
    assert "module load cuda/12.0" in content
    assert "module load python/3.11" in content
    Path(script_path).unlink()


def test_conda_env(tmp_path):
    script_path = create_slurm_script(
        job_name=str(tmp_path / "conda_job"),
        output_file="test.out",
        error_file="test.err",
        partition="compute",
        time_limit="00:30:00",
        conda_env="myenv",
        command="python script.py",
    )
    content = Path(script_path).read_text()
    assert "conda activate myenv" in content
    Path(script_path).unlink()
