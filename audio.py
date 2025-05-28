import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# Extract MFCC features
def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    audio, sample_rate = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)  # Return averaged MFCC features

# Visualize MFCC
def plot_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(label="MFCC Coefficients")
    plt.title("MFCC Spectrogram")
    plt.show()

# Test MFCC Extraction
audio_file = "/root/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2"  # Replace with an actual file
features = extract_mfcc(audio_file)
plot_mfcc(audio_file)

print("Extracted MFCC Features:", features.shape)


import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohammedabdeldayem/the-fake-or-real-dataset")

print("Path to dataset files:", path)

import os
import librosa
import numpy as np

# Define dataset paths
REAL_AUDIO_PATH = "/root/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-2sec/for-2seconds/training/real"
FAKE_AUDIO_PATH = "/root/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-2sec/for-2seconds/training/fake"
# Extract MFCC features
def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    audio, sample_rate = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)  # Averaging MFCCs

# Load real and fake audio files
real_audio_files = [os.path.join(REAL_AUDIO_PATH, f) for f in os.listdir(REAL_AUDIO_PATH) if f.endswith('.wav')]
fake_audio_files = [os.path.join(FAKE_AUDIO_PATH, f) for f in os.listdir(FAKE_AUDIO_PATH) if f.endswith('.wav')]

print(f"✅ Loaded {len(real_audio_files)} real audio samples and {len(fake_audio_files)} fake audio samples.")


import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

# Configuration
MAX_SAMPLES_PER_CLASS = 1000  # Maximum number of files to use from each class
RANDOM_SEED = 42

def load_subset(files, max_samples):
    """Randomly select a subset of files"""
    random.seed(RANDOM_SEED)
    return random.sample(files, min(max_samples, len(files)))

# Get subset of files
real_subset = load_subset(real_audio_files, MAX_SAMPLES_PER_CLASS)
fake_subset = load_subset(fake_audio_files, MAX_SAMPLES_PER_CLASS)

# Extract MFCC features for selected samples
X = []
y = []

for file in real_subset:
    features = extract_mfcc(file)
    if features is not None:  # Handle potential feature extraction errors
        X.append(features)
        y.append(0)  # 0 = Real

for file in fake_subset:
    features = extract_mfcc(file)
    if features is not None:
        X.append(features)
        y.append(1)  # 1 = Fake

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Check dataset balance
print(f"Loaded {len(y)} samples ({np.sum(y==0)} real, {np.sum(y==1)} fake)")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y  # Maintain class balance
)

# Train SVM Classifier
svm_model = SVC(
    kernel='linear',
    probability=True,
    random_state=RANDOM_SEED
)
svm_model.fit(X_train, y_train)

# Save Model
joblib.dump(svm_model, "svm_audio_model.pkl")

# Evaluate Model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ SVM Model Accuracy: {accuracy:.4f}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class AudioDataset(Dataset):
    def __init__(self, audio_files, labels):
        self.audio_files = audio_files
        self.labels = labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        feature = extract_mfcc(self.audio_files[idx])
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Create dataset
dataset = AudioDataset(real_audio_files + fake_audio_files, [0] * len(real_audio_files) + [1] * len(fake_audio_files))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define Neural Network with Attention Mechanism
class AudioClassifier(nn.Module):
    def __init__(self, input_size=13):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.attention = nn.Linear(128, 1)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        attn_weights = torch.softmax(self.attention(x), dim=0)
        x = x * attn_weights  # Apply attention mechanism
        x = self.fc2(x)
        return x

# Initialize Model
model = AudioClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Model
for epoch in range(10):
    total_loss = 0
    for features, labels in dataloader:
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "deepfake_audio_nn.pth")


def compare_results(audio_path, svm_model, nn_model, threshold=0.7):
    features = extract_mfcc(audio_path)

    # SVM Prediction
    svm_pred = svm_model.predict([features])[0]
    svm_conf = max(svm_model.predict_proba([features])[0])

    # Neural Network Prediction
    features_tensor = torch.tensor(features, dtype=torch.float32)
    nn_output = nn_model(features_tensor)
    nn_pred = torch.argmax(nn_output).item()
    nn_conf = torch.softmax(nn_output, dim=0)[nn_pred].item()

    print(f"🎙️ SVM Prediction: {'Fake' if svm_pred == 1 else 'Real'} (Confidence: {svm_conf:.2f})")
    print(f"🎙️ NN Prediction: {'Fake' if nn_pred == 1 else 'Real'} (Confidence: {nn_conf:.2f})")

    # Final Decision Logic
    if svm_pred == nn_pred:
        return "Final Prediction: " + ("Fake Audio" if nn_pred == 1 else "Real Audio")
    elif nn_conf > threshold:
        return "Final Prediction: Fake Audio (NN model confidence is high)"
    else:
        return "Final Prediction: Need further analysis"

# Load Models
svm_model = joblib.load("svm_audio_model.pkl")
nn_model = AudioClassifier()
nn_model.load_state_dict(torch.load("deepfake_audio_nn.pth"))
nn_model.eval()

# Test Decision
audio_path = "/content/file1.wav_16k.wav_norm.wav_mono.wav_silence.wav"
print(compare_results(audio_path, svm_model, nn_model))