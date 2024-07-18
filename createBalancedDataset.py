import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from io import StringIO

# Initialize the Google Cloud Storage client
client = storage.Client()

# Define the bucket and file names
bucket_name = 'garden-watermeter-readings_metadata'
input_file_name = 'training-data-all.csv'
balanced_file_name = 'balanced-dataset.csv'
training_file_name = 'training-dataset.csv'
validation_file_name = 'validation-dataset.csv'
test_file_name = 'test-dataset.csv'

# Load the dataset from Google Cloud Storage
def load_dataset(bucket_name, file_name):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    content = blob.download_as_string()
    df = pd.read_csv(StringIO(content.decode('utf-8')))
    df['label'] = df['label'].astype(str)  # Ensure labels are treated as strings
    return df

# Save the dataset to Google Cloud Storage
def save_dataset(df, bucket_name, file_name):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(df.to_csv(index=False), 'text/csv')

# Step 1: Load the dataset
df = load_dataset(bucket_name, input_file_name)

# Step 2: Balance the dataset
min_count = df['label'].value_counts().min()
balanced_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min_count)).reset_index(drop=True)

# Step 3: Split the dataset
train_df, temp_df = train_test_split(balanced_df, test_size=0.3, stratify=balanced_df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Step 4: Save the datasets
save_dataset(balanced_df, bucket_name, balanced_file_name)
save_dataset(train_df, bucket_name, training_file_name)
save_dataset(val_df, bucket_name, validation_file_name)
save_dataset(test_df, bucket_name, test_file_name)

# Print statistics of the balanced dataset
print(f"Balanced dataset label distribution:\n{balanced_df['label'].value_counts()}")
print(f"Training dataset label distribution:\n{train_df['label'].value_counts()}")
print(f"Validation dataset label distribution:\n{val_df['label'].value_counts()}")
print(f"Test dataset label distribution:\n{test_df['label'].value_counts()}")