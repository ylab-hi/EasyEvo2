from typer.testing import CliRunner

from easyevo2.__main__ import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "embed" in result.output
    assert "score" in result.output
    assert "vep" in result.output
    assert "generate" in result.output
    assert "slurm" in result.output


def test_list_models_cli():
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    assert "evo2_7b" in result.output
    assert "evo2_20b" in result.output
    assert "evo2_7b_262k" in result.output
