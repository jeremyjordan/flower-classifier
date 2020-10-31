import streamlit as st
import torch.nn as nn

from flower_classifier.datasets.oxford_flowers import NAMES as oxford_idx_to_names
from flower_classifier.ui import components

WEIGHTS_URL = "https://github.com/jeremyjordan/flower-classifier/releases/download/v0.1/efficientnet_b3a_example.ckpt"

st.title("Flower Classification")
model = components.download_model_url(WEIGHTS_URL)
pil_image = components.get_photo()
st.image(pil_image, caption="Input image", use_column_width=True)
logits = components.make_prediction(model, pil_image)
preds = nn.functional.softmax(logits, dim=1)
top_pred = preds.max(1)
label = oxford_idx_to_names[top_pred.indices.item()]
st.write(f"Prediction: {label}")

components.display_top_3_table(preds)
components.display_prediction_distribution(preds)


with st.beta_expander("Supported flower breeds"):
    breeds = "The model can recognize the following breeds:"
    for flower in oxford_idx_to_names:
        breeds += f"\n - {flower}"
    st.markdown(breeds)
