import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st
import torch.nn as nn
import torchvision
from PIL import Image

from flower_classifier.artifacts import download_model_checkpoint, list_run_files
from flower_classifier.datasets.oxford_flowers import NAMES as oxford_idx_to_names
from flower_classifier.models.classifier import FlowerClassifier

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title("Model Comparison")


# ----- sidebar -----
def get_photo():
    photo_bytes = st.sidebar.file_uploader("Input image:", ["png", "jpg", "jpeg"])
    if not photo_bytes:
        st.stop()
    pil_image = Image.open(photo_bytes).convert("RGB")
    return pil_image


pil_image = get_photo()
st.sidebar.image(pil_image, caption="Input image", use_column_width=True)

# ----- model comparision -----
col1, col2 = st.beta_columns(2)


def select_model(model_id):
    run_id = st.text_input("Model training run ID:", key=model_id)
    if not run_id:
        st.stop()
    files = list_run_files(run_id=run_id)
    checkpoint_file = st.selectbox("Choose a checkpoint file:", [None, *files])
    return run_id, checkpoint_file


@st.cache(allow_output_mutation=True)
def download_model(run_id, checkpoint_file):
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


with col1:
    st.write("Model A")
    run_id, checkpoint_file = select_model("Model A")
    if checkpoint_file:
        model = download_model(run_id, checkpoint_file)
        logits = make_prediction(model, pil_image)
        preds = nn.functional.softmax(logits, dim=1)
        top_pred = preds.max(1)
        label = oxford_idx_to_names[top_pred.indices.item()]
        st.write(f"Prediction: {label}")

        display_top_3_table(preds)
        display_prediction_distribution(preds)


with col2:
    st.write("Model B")
    run_id, checkpoint_file = select_model("Model B")
    if checkpoint_file:
        model = download_model(run_id, checkpoint_file)
        logits = make_prediction(model, pil_image)
        preds = nn.functional.softmax(logits, dim=1)
        top_pred = preds.max(1)
        label = oxford_idx_to_names[top_pred.indices.item()]
        st.write(f"Prediction: {label}")

        display_top_3_table(preds)
        display_prediction_distribution(preds)
