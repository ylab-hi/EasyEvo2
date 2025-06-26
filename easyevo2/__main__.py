import typer

from easyevo2.embed import embed
from easyevo2.model import ModelType
from easyevo2.score import score

# define a command-line interface (CLI) using Typer
# the cli include subcommands
app = typer.Typer(
    epilog="EasyEvo2 make life easier for you.\n",
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.command(
    help="Embed sequences using a model.", epilog="EasyEvo2 make life easier for you."
)(embed)


@app.command()
def list_models():
    """List all available model types."""
    models = ModelType.list_models()
    for model in models:
        print(model)


app.command(
    help="Score sequences using a model.", epilog="EasyEvo2 make life easier for you."
)(score)


if __name__ == "__main__":
    # run the CLI
    app()
