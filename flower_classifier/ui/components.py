import os
import re
import tempfile

import hydra
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import torch.nn as nn
import torchvision
from omegaconf import OmegaConf
from PIL import Image

from flower_classifier.artifacts import download_model_checkpoint
from flower_classifier.datasets.google.client import query_google_images
from flower_classifier.models.classifier import FlowerClassifier

NON_ALPHABETIC_CHARS_PATTERN = re.compile(r"[^a-zA-Z ]+")


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
    transforms = OmegaConf.load("conf/transforms/predict.yaml")
    transforms = [hydra.utils.instantiate(t) for t in transforms.val]
    inputs = torchvision.transforms.Compose(transforms)(pil_image)
    logits = model(inputs.unsqueeze(0))
    preds = nn.functional.softmax(logits, dim=1)
    return preds


def display_prediction(model, preds):
    top_pred = preds.max(1)
    label = model.classes[top_pred.indices.item()]
    st.write(f"Prediction: {label}")
    return label


def display_top_3_table(model, preds):
    with st.beta_expander("View top 3 predictions"):
        top_3 = preds.topk(k=3, dim=1)
        labels = [model.classes[idx.item()] for idx in top_3.indices[0]]
        scores = top_3.values[0].detach().cpu().numpy()
        st.table({"predicted class": labels, "scores": scores})


def display_prediction_distribution(model, preds):
    with st.beta_expander("View full prediction distribution"):
        data = {"scores": preds.squeeze().detach().numpy(), "flower": model.classes}
        df = pd.DataFrame(data)
        fig = px.bar(df, y="scores", x="flower", title="Prediction Distribution")
        st.plotly_chart(fig, use_container_width=True)


def display_examples(label):
    with st.beta_expander(f"View other {label} examples"):
        image_urls = query_google_images(label)
        for i, image_url in enumerate(image_urls):
            st.image(image_url, caption=f"Example image {i}", use_column_width=True)


def ask_user_if_correct():
    is_correct = st.selectbox(
        "(Optional) Do you think the model's prediction is correct?", options=["Unsure", "Correct", "Wrong"]
    )
    is_correct_tags = {
        "Unsure": "user_judgement:unsure",
        "Correct": "user_judgement:correct",
        "Wrong": "user_judgement:wrong",
    }
    return is_correct_tags[is_correct]


def ask_user_for_breed():
    user_input = st.text_input("(Optional) Do you know what breed this flower is?")
    cleaned_input = re.sub(NON_ALPHABETIC_CHARS_PATTERN, "", user_input)
    cleaned_input = cleaned_input.replace(" ", "_")
    return f"user_input:{cleaned_input}"
