import os
import tempfile

import streamlit as st
import torch.nn as nn
import torchvision
from PIL import Image

from flower_classifier.datasets.oxford_flowers import NAMES as oxford_idx_to_names
from flower_classifier.models.baseline import BaselineResnet
from flower_classifier.models.train import FlowerClassifier

st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Flower classifier")

# support multiple classification taxonomies (eventually)
idx_to_name = {"Oxford Flowers 102": oxford_idx_to_names}
dataset = st.selectbox("Choose a classification dataset:", list(idx_to_name.keys()))

# authenticate with wandb
api_key = st.text_input("Enter WandB API key:", "")
os.environ["WANDB_API_KEY"] = api_key.strip(" ")
if api_key == "":
    st.stop()

from flower_classifier.artifacts import download_model_checkpoint, list_run_files, list_runs  # noqa: E402

runs = list_runs(project="flowers", entity="jeremytjordan")
run_id = st.selectbox("Model training run ID:", runs)

if not run_id:
    st.stop()

# TODO we can update our checkpoint callback to always save a best.ckpt and get rid of this
files = list_run_files(run_id=run_id)
checkpoint_file = st.selectbox("Choose a checkpoint file:", [None, *files])
if checkpoint_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        with st.spinner("Downloading model weights..."):
            checkpoint = download_model_checkpoint(checkpoint_file, run_id, tmpdir)
            checkpoint_path = os.path.join(tmpdir, checkpoint)
        network = BaselineResnet(pretrained=False)
        model = FlowerClassifier.load_from_checkpoint(checkpoint_path=checkpoint_path, network=network)
    st.write("Model is ready to make predictions!")

photo_bytes = st.file_uploader("Input image:", ["png", "jpg", "jpeg"])
if photo_bytes:
    pil_image = Image.open(photo_bytes).convert("RGB")
    st.image(pil_image, caption="Input image", width=480)
    inputs = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(pil_image)
    logits = model(inputs.unsqueeze(0))
    pred = nn.functional.softmax(logits, dim=1).argmax(1).item()
    st.write(f"Prediction: {oxford_idx_to_names[pred]}")
