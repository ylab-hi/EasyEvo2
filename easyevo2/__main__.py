import typer

from easyevo2.embed import embed
from easyevo2.exon_classifier import classify_exons
from easyevo2.list_models import list_models
from easyevo2.score import score

# define a command-line interface (CLI) using Typer
# the cli include subcommands
app = typer.Typer(
    epilog="EasyEvo2 make life easier for you.\n",
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.command(help="Embed sequences using a model.")(embed)


app.command(help="List all available models.")(list_models)


app.command(help="Score sequences using a model.")(score)

app.command(help="Classify exons using a model.")(classify_exons)

if __name__ == "__main__":
    # run the CLI
    app()
