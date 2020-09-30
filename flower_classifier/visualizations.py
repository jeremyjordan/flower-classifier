import numpy as np
import plotly.express as px


def generate_confusion_matrix(confusion_matrix: np.ndarray, class_names: str):
    fig = px.imshow(
        confusion_matrix, labels=dict(x="Predicted Class", y="True Class", color="Count"), x=class_names, y=class_names
    )
    fig.update_xaxes(side="bottom")
    return fig
