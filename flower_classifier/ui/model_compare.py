"""
To run this app: streamlit run flower_classifier/ui/model_compare.py
This app is intended to run locally, where you have already authenticated
with Weights and Biases.
"""

import streamlit as st

from flower_classifier.artifacts import list_run_files
from flower_classifier.ui import components

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title("Model Comparison")

# ----- sidebar -----
pil_image = components.get_photo(use_sidebar=True)
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


with col1:
    st.write("Model A")
    run_id, checkpoint_file = select_model("Model A")
    if checkpoint_file:
        model = components.download_model_wandb(run_id, checkpoint_file)
        preds = components.make_prediction(model, pil_image)

        components.display_prediction(model, preds)
        components.display_top_3_table(model, preds)
        components.display_prediction_distribution(model, preds)


with col2:
    st.write("Model B")
    run_id, checkpoint_file = select_model("Model B")
    if checkpoint_file:
        model = components.download_model_wandb(run_id, checkpoint_file)
        preds = components.make_prediction(model, pil_image)

        components.display_prediction(model, preds)
        components.display_top_3_table(model, preds)
        components.display_prediction_distribution(model, preds)
