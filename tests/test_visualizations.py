import numpy as np
from plotly.graph_objects import Figure

from flower_classifier.visualizations import generate_confusion_matrix


def test_confusion_matrix_with_class_names():
    cm = np.array([[6, 0, 1], [2, 4, 0], [1, 1, 8]])
    class_names = ["a", "b", "c"]
    fig = generate_confusion_matrix(cm, class_names)
    assert isinstance(fig, Figure)


def test_confusion_matrix_without_class_names():
    cm = np.array([[6, 0, 1], [2, 4, 0], [1, 1, 8]])
    class_names = None
    fig = generate_confusion_matrix(cm, class_names)
    assert isinstance(fig, Figure)
