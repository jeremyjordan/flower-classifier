from datetime import datetime

import streamlit as st

from flower_classifier.datasets.flickr.client import get_authenticated_client, upload_photo
from flower_classifier.datasets.oxford_flowers import NAMES as oxford_idx_to_names
from flower_classifier.ui import components

WEIGHTS_URL = "https://github.com/jeremyjordan/flower-classifier/releases/download/v0.1/efficientnet_b3a_example.ckpt"

st.title("Flower Classification")
model = components.download_model_url(WEIGHTS_URL)
pil_image = components.get_photo()
st.image(pil_image, caption="Input image", use_column_width=True)
preds = components.make_prediction(model, pil_image)

components.display_prediction(preds)
components.display_top_3_table(preds)
components.display_prediction_distribution(preds)


with st.beta_expander("Supported flower breeds"):
    breeds = "The model can recognize the following breeds:"
    for flower in oxford_idx_to_names:
        breeds += f"\n - {flower}"
    st.markdown(breeds)


save_photo = st.button("Save photo to database")
st.markdown("*Sharing this photo will help us improve our machine learning models!*")
if save_photo:
    flickr_client = get_authenticated_client()
    current = datetime.now().strftime("%Y-%m-%d_%H-%m")
    upload_photo(flickr_client, filename=f"{current}.jpg", pil_image=pil_image)
    st.balloons()
    st.success("Thanks for sharing!")
