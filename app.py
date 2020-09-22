import streamlit as st
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from flower_classifier.artifacts import download_model_checkpoint, list_run_files, list_runs
from flower_classifier.datasets.oxford_flowers import DEFAULT_IMG_TRANSFORMS as oxford_transforms
from flower_classifier.datasets.oxford_flowers import NAMES as oxford_idx_to_names
from flower_classifier.models.baseline import BaselineResnet
from flower_classifier.models.train import FlowerClassifier

st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Flower classifier")

# support multiple classification taxonomies (eventually)
idx_to_name = {"Oxford Flowers 102": oxford_idx_to_names}
dataset = st.selectbox("Choose a classification dataset:", list(idx_to_name.keys()))

default_run = list_runs(project="flowers", entity="jeremytjordan")[0]
run_id = st.text_input("Model training run ID:", default_run)

if run_id:
    # TODO we can update our checkpoint callback to always save a best.ckpt and get rid of this
    files = list_run_files(run_id=run_id)
    checkpoint_file = st.selectbox("Choose a checkpoint file:", [None, *files])
    if checkpoint_file:
        checkpoint = download_model_checkpoint(checkpoint_file, run_id)
        network = BaselineResnet(pretrained=False)
        model = FlowerClassifier.load_from_checkpoint(checkpoint_path=checkpoint, network=network)
        st.write("Model is ready to make predictions!")

    photo_bytes = st.file_uploader("Input image:", ["png", "jpg", "jpeg"])
    if photo_bytes:
        pil_image = Image.open(photo_bytes).convert("RGB")
        inputs = transforms.Compose(oxford_transforms)(pil_image)
        logits = model(inputs.unsqueeze(0))
        pred = nn.functional.softmax(logits, dim=1).argmax(1).item()
        st.write(f"Prediction: {oxford_idx_to_names[pred]}")
