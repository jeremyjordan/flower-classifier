import typer

from flower_classifier.datasets.oxford_flowers import app as oxford_app
from flower_classifier.label import app as label_app

app = typer.Typer()
app.add_typer(oxford_app, name="oxford")
app.add_typer(label_app, name="label")


if __name__ == "__main__":
    app()
