import typer

from rose_cli.models.add import add_model
from rose_cli.models.get import get_model
from rose_cli.models.list import list_models
from rose_cli.models.pull import pull_model

app = typer.Typer()

app.command(name="list")(list_models)
app.command(name="get")(get_model)
app.command(name="pull")(pull_model)
app.command(name="add")(add_model)
