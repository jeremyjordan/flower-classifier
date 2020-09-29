import numpy as np
import plotly.figure_factory as ff


def generate_confusion_matrix(confusion_matrix: np.ndarray, class_names: str):
    fig = ff.create_annotated_heatmap(
        confusion_matrix,
        x=class_names,
        y=class_names,
        colorscale="Greens",
        showscale=True,
    )
    return fig
