import os
import docx
import easyocr

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from sklearn.metrics import recall_score


# Function to preprocess labels
def preprocess_labels(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    onehot_encoded = to_categorical(integer_encoded, num_classes=num_classes)
    return onehot_encoded, num_classes

# Function to pad or truncate sequences
def pad_or_truncate(sequence, max_length):
    return pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

# Set the path to the directory
data_dir = r'C:\Users\91701\Downloads\assignments\assignments\assignment1\data'  # Use your actual data directory path

# Load labels from CSV files
train_labels_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_labels_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Function to check file type
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

def is_doc_file(filename):
    return filename.lower().endswith(('.docx', '.doc'))

# Function to read text from document files
def read_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Function to read text from image files using EasyOCR
def read_text_from_image_easyocr(file_path):
    try:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(file_path)
        text = ' '.join([entry[1] for entry in result])
        return text
    except Exception as e:
        print(f"Error reading text from image {file_path}: {str(e)}")
        return None

# List to store file paths
image_files_train = []
doc_files_train = []
image_files_test = []
doc_files_test = []

# Segregate files in the train folder
for filename in os.listdir(os.path.join(data_dir, 'train')):
    file_path = os.path.join(data_dir, 'train', filename)
    if is_image_file(filename):
        image_files_train.append(file_path)
    elif is_doc_file(filename):
        doc_files_train.append(file_path)

# Segregate files in the test folder
for filename in os.listdir(os.path.join(data_dir, 'test')):
    file_path = os.path.join(data_dir, 'test', filename)
    if is_image_file(filename):
        image_files_test.append(file_path)
    elif is_doc_file(filename):
        doc_files_test.append(file_path)

# Combine text data from document files and image files
X_text_train = [read_text_from_docx(file_path) for file_path in doc_files_train]
X_text_train += [read_text_from_image_easyocr(file_path) for file_path in image_files_train]

X_text_test = [read_text_from_docx(file_path) for file_path in doc_files_test]
X_text_test += [read_text_from_image_easyocr(file_path) for file_path in image_files_test]

# Load and preprocess labels for training dataset
y_train_agreement_value_train, num_classes_agreement_value = preprocess_labels(train_labels_df['Aggrement Value'])
y_train_agreement_start_date_train, num_classes_agreement_start_date = preprocess_labels(train_labels_df['Aggrement Start Date'])
y_train_agreement_end_date_train, num_classes_agreement_end_date = preprocess_labels(train_labels_df['Aggrement End Date'])
y_train_renewal_notice_days_train, num_classes_renewal_notice_days = preprocess_labels(train_labels_df['Renewal Notice (Days)'])
y_train_party_one_train, num_classes_party_one = preprocess_labels(train_labels_df['Party One'])
y_train_party_two_train, num_classes_party_two = preprocess_labels(train_labels_df['Party Two'])

# Load and preprocess labels for testing dataset
y_test_agreement_value_test, _ = preprocess_labels(test_labels_df['Aggrement Value'])
y_test_agreement_start_date_test, _ = preprocess_labels(test_labels_df['Aggrement Start Date'])
y_test_agreement_end_date_test, _ = preprocess_labels(test_labels_df['Aggrement End Date'])
y_test_renewal_notice_days_test, _ = preprocess_labels(test_labels_df['Renewal Notice (Days)'])
y_test_party_one_test, _ = preprocess_labels(test_labels_df['Party One'])
y_test_party_two_test, _ = preprocess_labels(test_labels_df['Party Two'])

# Pad the array with zeros to expand the second dimension from 5 to 9
y_train_renewal_notice_days_train = np.pad(y_train_renewal_notice_days_train, ((0, 0), (0, 4)), mode='constant')

print("New shape of y_train_renewal_notice_days_train:", y_train_renewal_notice_days_train.shape)
# Concatenate the target labels for training dataset

max_size = max(y_test_agreement_value_test.shape[1],
               y_test_agreement_start_date_test.shape[1],
               y_test_agreement_end_date_test.shape[1],
               y_test_renewal_notice_days_test.shape[1],
               y_test_party_one_test.shape[1],
               y_test_party_two_test.shape[1])

y_test_agreement_value_test = np.pad(y_test_agreement_value_test, ((0, 0), (0, max_size - y_test_agreement_value_test.shape[1])), mode='constant')
y_test_agreement_start_date_test = np.pad(y_test_agreement_start_date_test, ((0, 0), (0, max_size - y_test_agreement_start_date_test.shape[1])), mode='constant')
y_test_agreement_end_date_test = np.pad(y_test_agreement_end_date_test, ((0, 0), (0, max_size - y_test_agreement_end_date_test.shape[1])), mode='constant')
y_test_renewal_notice_days_test = np.pad(y_test_renewal_notice_days_test, ((0, 0), (0, max_size - y_test_renewal_notice_days_test.shape[1])), mode='constant')
y_test_party_one_test = np.pad(y_test_party_one_test, ((0, 0), (0, max_size - y_test_party_one_test.shape[1])), mode='constant')
y_test_party_two_test = np.pad(y_test_party_two_test, ((0, 0), (0, max_size - y_test_party_two_test.shape[1])), mode='constant')


min_samples_train = min(y_train_agreement_value_train.shape[0], y_train_agreement_start_date_train.shape[0],
                        y_train_agreement_end_date_train.shape[0], y_train_renewal_notice_days_train.shape[0],
                        y_train_party_one_train.shape[0], y_train_party_two_train.shape[0])

# Concatenate the target labels for training dataset
y_train = np.concatenate([
    y_train_agreement_value_train[:min_samples_train],
    y_train_agreement_start_date_train[:min_samples_train],
    y_train_agreement_end_date_train[:min_samples_train],
    y_train_renewal_notice_days_train[:min_samples_train],
    y_train_party_one_train[:min_samples_train],
    y_train_party_two_train[:min_samples_train]
], axis=1)



# Ensure all arrays have the same number of samples for testing dataset
min_samples_test = min(y_test_agreement_value_test.shape[0], y_test_agreement_start_date_test.shape[0],
                       y_test_agreement_end_date_test.shape[0], y_test_renewal_notice_days_test.shape[0],
                       y_test_party_one_test.shape[0], y_test_party_two_test.shape[0])

# Concatenate the target labels for testing dataset
y_test = np.concatenate([
    y_test_agreement_value_test[:min_samples_test],
    y_test_agreement_start_date_test[:min_samples_test],
    y_test_agreement_end_date_test[:min_samples_test],
    y_test_renewal_notice_days_test[:min_samples_test],
    y_test_party_one_test[:min_samples_test],
    y_test_party_two_test[:min_samples_test]
], axis=1)

# Assuming X_text_train contains your text data
# Convert text data to sequences using Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_text_train)
X_train_sequences = tokenizer.texts_to_sequences(X_text_train)

# Padding or truncating sequences to ensure they have the same length
max_sequence_length = 100  # Example maximum sequence length
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post', truncating='post')


X_test_sequences = tokenizer.texts_to_sequences(X_text_test)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post', truncating='post')


# Assuming num_classes is correctly defined
num_classes = y_train.shape[1]  # Assuming y_train_agreement_value_train has the correct shape
vocab_size = len(tokenizer.word_index) + 1 

embedding_dim = 100  # Example value, adjust as needed

# Define the model architecture
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes_agreement_value, activation='softmax', name='agreement_value_output'),  # Output layer for Agreement Value
    Dense(num_classes_agreement_start_date, activation='softmax', name='agreement_start_date_output'),  # Output layer for Agreement Start Date
    Dense(num_classes_agreement_end_date, activation='softmax', name='agreement_end_date_output'),  # Output layer for Agreement End Date
    Dense(num_classes_renewal_notice_days, activation='softmax', name='renewal_notice_days_output'),  # Output layer for Renewal Notice (Days)
    Dense(num_classes_party_one, activation='softmax', name='party_one_output'),  # Output layer for Party One
    Dense(num_classes_party_two, activation='softmax', name='party_two_output')  # Output layer for Party Two
])


# Define a dictionary to store training data and corresponding output names
train_data = {
    'agreement_value_output': y_train_agreement_value_train,
    'agreement_start_date_output': y_train_agreement_start_date_train,
    'agreement_end_date_output': y_train_agreement_end_date_train,
    'renewal_notice_days_output': y_train_renewal_notice_days_train,
    'party_one_output': y_train_party_one_train,
    'party_two_output': y_train_party_two_train
}

# Define a dictionary to store training history for each output
history_dict = {}

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Print model summary
model.summary()

# Train the model for each output
for output_name, y_train in train_data.items():
    print(f"Training model for {output_name}...")
    history = model.fit(X_train_padded, y_train, epochs=100, batch_size=32)
    history_dict[output_name] = history
    print(f"Training completed for {output_name}\n")


# Predict probabilities on the test set
y_pred_prob = model.predict(X_test_padded)

# Convert predicted probabilities to binary labels using a threshold of 0.5
y_pred_binary = (y_pred_prob > 0.5).astype(int)

# Debug: Print shapes of y_test_agreement_value_test and y_pred_binary
print("Shape of y_test_agreement_value_test:", y_test_agreement_value_test.shape)
print("Shape of y_pred_binary:", y_pred_binary.shape)

# Evaluate the model using binary labels
from sklearn.metrics import classification_report
print(classification_report(y_test_agreement_value_test.argmax(axis=1), y_pred_binary.argmax(axis=1)))

# Assuming X_text_test contains your text data
# Convert text data to sequences using Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_text_test)
X_test_sequences = tokenizer.texts_to_sequences(X_text_test)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Predict probabilities on the test set
y_pred_prob = model.predict(X_test_padded)

# Decode the predicted labels to get the extracted entities
predicted_entities = []

# Loop through each predicted label
for i in range(len(y_pred_prob)):
    entities = []
    # Check if the probability for this label is above a certain threshold (e.g., 0.5)
    for j in range(len(y_pred_prob[i])):
        if y_pred_prob[i][j] > 0.5:
            # If the probability is above the threshold, add the corresponding entity to the list
            entities.append(tokenizer.index_word[j])  # Use the Tokenizer's index_word dictionary to decode the index
    predicted_entities.append(entities)

# Print the extracted entities for each document or image
for i in range(len(X_text_test)):
    print("Document or Image", i+1)
    print("Extracted Entities:", predicted_entities[i])
    print()
    
    
# Define a function to calculate per field recall
def calculate_per_field_recall(true_values, predicted_values):
    per_field_recall = []
    for i in range(true_values.shape[1]):
        try:
            recall = recall_score(true_values[:, i], predicted_values[:, i])
            per_field_recall.append(recall)
        except IndexError:
            # Handle the case where there are no true samples for a field
            per_field_recall.append(0.0)
    return per_field_recall

# Calculate per field recall for the test set
per_field_recall = calculate_per_field_recall(y_test, y_pred_binary)
print("Per Field Recall:")
for i, field in enumerate(['Agreement Value', 'Agreement Start Date', 'Agreement End Date', 'Renewal Notice (Days)', 'Party One', 'Party Two']):
    print(f"{field}: {per_field_recall[i]}")
