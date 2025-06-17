import click

from .__main__ import main as interpret
from .example import example


@click.group()
def cli() -> None:
    """Simple-VS: a simple image 3D interpretation tool."""


cli.add_command(example, name="example")
cli.add_command(interpret, name="interpret")


if __name__ == "__main__":
    cli()
