import os
import tempfile

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import torchvision
from PIL import Image

from flower_classifier.artifacts import download_model_checkpoint
from flower_classifier.datasets.oxford_flowers import NAMES as oxford_idx_to_names
from flower_classifier.models.classifier import FlowerClassifier


def get_photo(use_sidebar=False):
    widget = st.sidebar if use_sidebar else st
    photo_bytes = widget.file_uploader("Input image:", ["png", "jpg", "jpeg"])
    if not photo_bytes:
        st.stop()
    pil_image = Image.open(photo_bytes).convert("RGB")
    return pil_image


@st.cache
def download_model_url(weights_url):
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "model.ckpt")
        r = requests.get(weights_url, allow_redirects=True)
        with open(checkpoint_path, "wb") as f:
            f.write(r.content)
        model = FlowerClassifier.load_from_checkpoint(checkpoint_path=checkpoint_path, strict=False)
        model.freeze()

    return model


@st.cache(allow_output_mutation=True)
def download_model_wandb(run_id, checkpoint_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = download_model_checkpoint(checkpoint_file, run_id, tmpdir)
        checkpoint_path = os.path.join(tmpdir, checkpoint)
        model = FlowerClassifier.load_from_checkpoint(checkpoint_path=checkpoint_path, strict=False, pretrained=False)
        model.freeze()
    return model


@st.cache(allow_output_mutation=True)
def make_prediction(model, pil_image):
    inputs = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(512),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(pil_image)
    logits = model(inputs.unsqueeze(0))
    return logits


def display_top_3_table(preds):
    with st.beta_expander("View top 3 predictions"):
        top_3 = preds.topk(k=3, dim=1)
        labels = [oxford_idx_to_names[idx.item()] for idx in top_3.indices[0]]
        scores = top_3.values[0].detach().cpu().numpy()
        st.table({"predicted class": labels, "scores": scores})


def display_prediction_distribution(preds):
    with st.beta_expander("View full prediction distribution"):
        data = {"scores": preds.squeeze().detach().numpy(), "flower": oxford_idx_to_names}
        df = pd.DataFrame(data)
        fig = px.bar(df, y="scores", x="flower", title="Prediction Distribution")
        st.plotly_chart(fig, use_container_width=True)
