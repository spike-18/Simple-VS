import nox_poetry

# Default locations to check/format
locations = "src", "tests", "noxfile.py"


@nox_poetry.session(python=["3.13"])
def formatter(session) -> None:
    """Run ruff code formatter."""
    session.run("poetry", "install", external=True)
    session.run("ruff", "format")


@nox_poetry.session(python=["3.13"])
def linter(session) -> None:
    """Lint using ruff."""
    session.run("poetry", "install", external=True)
    session.run("ruff", "check", "--fix")


@nox_poetry.session(python=["3.13"])
def tests(session) -> None:
    """Run the test suite."""
    session.run("poetry", "install", external=True)
    session.run("pytest", "--cov")
