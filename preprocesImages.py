import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from google.cloud import storage

# Set up Google Cloud Storage client
def get_gcs_client():
    return storage.Client()

def download_image(image_url):
    response = requests.get(image_url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def upload_image_to_gcs(bucket_name, image, image_name):
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(image_name)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    blob.upload_from_file(buffer, content_type='image/jpeg')
    return f"https://storage.googleapis.com/{bucket_name}/{image_name}"

def process_image(image, crop_params, image_base_name):
    cropped_images = []
    for i, (left, top, right, bottom) in enumerate(crop_params, start=1):
        cropped_image = image.crop((left, top, right, bottom))
        cropped_image = upscale_image(cropped_image)
        cropped_image_name = f"{image_base_name}-{i}.jpeg"
        cropped_images.append((cropped_image, cropped_image_name))
    return cropped_images

def upscale_image(image, min_size=640):
    ratio = min_size / min(image.size)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    return image.resize(new_size, Image.Resampling.BICUBIC)  # Use LANCZOS

def process_and_upload_images(record, crop_params):
    image_uri = record['image_uri']
    image_id = os.path.splitext(os.path.basename(image_uri))[0]
    label = str(record['label'])  # Ensure label is treated as a string

    # Ensure label length is 8
    if len(label) != 8:
        raise ValueError("Label must be 8 characters long")
    
    # Download and process the image
    image = download_image(image_uri)
    cropped_images = process_image(image, crop_params, image_id)

    # Upload processed images to GCS and construct new records
    new_records = []
    for i, (cropped_image, cropped_image_name) in enumerate(cropped_images):
        new_image_uri = upload_image_to_gcs("garden_watermeter-training-data-all", cropped_image, cropped_image_name)
        new_records.append([record['id'], new_image_uri, label[i]])

    return new_records

def main():
    # Load the CSV file
    metadata_file = "gs://garden-watermeter-readings_metadata/metadata_cleaned.csv"
    df = pd.read_csv(metadata_file, dtype={'label': str})  # Read label as string

    # Define the cropping parameters (these should be provided)
    crop_params = [
        # These are example parameters (left, top, right, bottom)
        (530, 670, 574, 725), (574, 670, 618, 725), (618, 670, 662, 725), (662, 670, 706, 725),
        (706, 670, 750, 725), (750, 670, 794, 725), (794, 670, 838, 725), (838, 670, 882, 725)
    ]

    # Create a list to store new records
    new_records = []

    for _, record in df.iterrows():
        new_records.extend(process_and_upload_images(record, crop_params))

    # Create a new DataFrame for the new records
    new_df = pd.DataFrame(new_records, columns=['id', 'image_uri', 'label'])

    # Save the new DataFrame to a new CSV file in GCS
    new_file = "training-data-all.csv"
    client = get_gcs_client()
    bucket = client.bucket("garden-watermeter-readings_metadata")
    blob = bucket.blob(new_file)
    blob.upload_from_string(new_df.to_csv(index=False), content_type='text/csv')

if __name__ == "__main__":
    main()
