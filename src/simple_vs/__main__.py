import click

from . import __version__
from .io import load_image, save_image
from .pipeline import process_image


@click.command()
@click.version_option(version=__version__)
@click.argument("file", type=click.File(mode="rb"))
@click.option("-v", "--verbose", is_flag=True, default=False, help="Print debug information.")
@click.option("--save", is_flag=True, default=False, help="Save rendered 3D image.")
def main(
    file: click.File,
    save: bool,
    verbose: bool,
) -> int | None:
    """Interpret input 2D image to 3D.

    Renders output image.
    Optionally saves the image as <filename>_interp.png.
    """
    if verbose:
        click.secho(f"Filename: {file.name}")
        click.secho(f"Save image? {'yes' if save else 'no'}")

    image = load_image(file)

    output_image = process_image(image)

    if save:
        save_image(output_image)


if __name__ == "__main__":
    click.echo(
        "Use 'poetry run example' to print example or 'poetry run interpret'\
        to interpret custom image.",
    )
