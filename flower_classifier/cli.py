import typer

from flower_classifier.datasets.oxford_flowers import app as oxford_app

app = typer.Typer()
app.add_typer(oxford_app, name="oxford")


if __name__ == "__main__":
    app()
