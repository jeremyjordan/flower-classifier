from typing import List, Optional

import numpy as np
import plotly.express as px


def generate_confusion_matrix(confusion_matrix: np.ndarray, class_names: Optional[List[str]]):
    if class_names is None:
        class_names = np.arange(confusion_matrix.shape[0]).astype(str)
    fig = px.imshow(
        confusion_matrix,
        labels=dict(x="Predicted Class", y="True Class", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale="Greens",
    )
    fig.update_xaxes(side="bottom")
    fig.update(layout_coloraxis_showscale=False)
    return fig
