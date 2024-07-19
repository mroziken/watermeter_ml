import os
import streamlit as st
from google.cloud import storage, aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from PIL import Image
from dotenv import load_dotenv
import io
import base64

# Load environment variables
load_dotenv()

# Configuration from .env
PROJECT_ID = os.getenv("PROJECT_ID")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
LOCATION = os.getenv("LOCATION")

# Cropping parameters
CROP_PARAMS = [
    (530, 670, 574, 725), (574, 670, 618, 725), (618, 670, 662, 725), (662, 670, 706, 725),
    (706, 670, 750, 725), (750, 670, 794, 725), (794, 670, 838, 725), (838, 670, 882, 725)
]

def download_image(uri):
    storage_client = storage.Client()
    bucket_name, blob_name = uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    image_bytes = blob.download_as_bytes()
    return Image.open(io.BytesIO(image_bytes))

def crop_and_resize(image):
    cropped_images = []
    for crop_param in CROP_PARAMS:
        cropped_image = image.crop(crop_param)
        cropped_image = cropped_image.resize((640, int((640 / cropped_image.width) * cropped_image.height)), Image.Resampling.BICUBIC)
        cropped_images.append(cropped_image)
    return cropped_images

def predict_image_classification(image):
    client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_content = base64.b64encode(buffered.getvalue()).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(content=encoded_content).to_value()
    instances = [instance]
    parameters = predict.params.ImageClassificationPredictionParams(confidence_threshold=0.5, max_predictions=5).to_value()
    endpoint = client.endpoint_path(project=PROJECT_ID, location=LOCATION, endpoint=ENDPOINT_ID)
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    return response.predictions

def extract_label(prediction):
    try:
        display_names = prediction["displayNames"]
        if display_names:
            return display_names[0]
    except KeyError as e:
        print(f"KeyError: {e}")
    except AttributeError as e:
        print(f"AttributeError: {e}")
    return 'Unknown'

st.title("Image Classification App")

uri = st.text_input("Enter the Image URI (gs://):", value="gs://garden-watermeter-readings_processed/8f16e53375c98a38.jpeg")

if uri:
    original_image = download_image(uri)
    st.image(original_image, caption='Original Image', use_column_width=True)

    if st.button("Classify Image"):
        cropped_images = crop_and_resize(original_image)
        classifications = [predict_image_classification(image) for image in cropped_images]

        classification_labels = [extract_label(prediction) for classification in classifications for prediction in classification]

        st.write("Meter Reading:")
        meterReading = ""
        for i, label in enumerate(classification_labels):
            meterReading += label
        st.write(meterReading)