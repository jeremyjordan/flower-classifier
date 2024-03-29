import hashlib
from pathlib import Path

import streamlit as st

from flower_classifier.datasets.flickr.client import get_authenticated_client, upload_photo
from flower_classifier.ui import components

WEIGHTS_URL = "https://github.com/jeremyjordan/flower-classifier/releases/download/v1.1/efficient_net_b3_run_2sas9p42_epoch_44.ckpt"  # noqa: E501
MODEL_ID = "_".join(Path(WEIGHTS_URL).parts[-2:])

st.title("Flower Classification")
model = components.download_model_url(WEIGHTS_URL)
pil_image = components.get_photo()
st.image(pil_image, caption="Input image", use_column_width=True)
preds = components.make_prediction(model, pil_image)

predicted_class = components.display_prediction(model, preds)
components.display_examples(predicted_class)
components.display_top_3_table(model, preds)
components.display_prediction_distribution(model, preds)

with st.beta_expander("Supported flower breeds"):
    breeds = "The model can recognize the following breeds:"
    for flower in model.classes:
        breeds += f"\n - {flower}"
    st.markdown(breeds)


st.markdown(
    """
    Want to help us improve this model? Share the photo with us so we can include it
    in our training data! Just click the button labeled "Save photo to database".
    """
)
tags = set()
model_pred = predicted_class.replace(" ", "_")
tags.add(f"pred:{model_pred}")
tags.add(f"model:{MODEL_ID}")


# Does the user think the prediction is correct?
is_correct_tag = components.ask_user_if_correct()
tags.add(is_correct_tag)

# Does the user know what breed the flower actually is?
if is_correct_tag == "user_judgement:wrong":
    guess_breed_tag = components.ask_user_for_breed()
    tags.add(guess_breed_tag)

save_photo = st.button("Save photo to database")
if save_photo:
    flickr_client = get_authenticated_client()
    image_hash = hashlib.md5(pil_image.tobytes()).hexdigest()
    upload_photo(flickr_client, filename=f"{image_hash}.jpg", pil_image=pil_image, tags=tags)
    st.balloons()
    st.success("Thanks for sharing!")
