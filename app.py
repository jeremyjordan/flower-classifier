import plotly.express as px
import streamlit as st
import torch.nn as nn
import torchvision
from PIL import Image

from flower_classifier.datasets.oxford_flowers import NAMES as oxford_idx_to_names
from flower_classifier.models.classifier import FlowerClassifier

WEIGHTS_URL = "https://github.com/jeremyjordan/flower-classifier/releases/download/v0.1/efficientnet_b3a_example.ckpt"

st.title("Flower Classification")


@st.cache
def download_weights(weights_url):
    import requests

    weights_path = "model.ckpt"
    r = requests.get(weights_url, allow_redirects=True)
    with open(weights_path, "wb") as f:
        f.write(r.content)
    return weights_path


@st.cache
def load_model(weights_path):
    model = FlowerClassifier.load_from_checkpoint(checkpoint_path=weights_path, strict=False)
    model.freeze()
    return model


def get_photo():
    photo_bytes = st.file_uploader("Input image:", ["png", "jpg", "jpeg"])
    if not photo_bytes:
        st.stop()
    pil_image = Image.open(photo_bytes).convert("RGB")
    return pil_image


# MAIN
weights_path = download_weights(WEIGHTS_URL)
model = load_model(weights_path)
pil_image = get_photo()
st.image(pil_image, caption="Input image", use_column_width=True)
inputs = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)(pil_image)
logits = model(inputs.unsqueeze(0))
preds = nn.functional.softmax(logits, dim=1)
top_pred = preds.max(1)
label = oxford_idx_to_names[top_pred.indices.item()]
st.write(f"Prediction: {label}")

with st.beta_expander("View top 3 predictions"):
    top_3 = preds.topk(k=3, dim=1)
    labels = [oxford_idx_to_names[idx.item()] for idx in top_3.indices[0]]
    scores = top_3.values[0].detach().cpu().numpy()
    st.table({"predicted class": labels, "scores": scores})


with st.beta_expander("View full predictions distribution"):
    import pandas as pd

    data = {"scores": preds.squeeze().detach().numpy(), "flower": oxford_idx_to_names}
    df = pd.DataFrame(data)
    fig = px.bar(df, y="scores", x="flower", title="Prediction Distribution")
    # https://github.com/streamlit/streamlit/issues/2220
    st.plotly_chart(fig, use_container_width=True)
