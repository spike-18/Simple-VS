import click

from .io import load_image
from .pipeline import process_image


@click.command()
def example() -> None:
    image = load_image("example_images/example.png")

    process_image(image)
