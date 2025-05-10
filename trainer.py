import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import Dataset, DataLoader


def split_audio(input_file, segment_duration=5, output_folder="segments"):
    os.makedirs(output_folder, exist_ok=True)
    waveform, sample_rate = torchaudio.load(input_file)
    segment_samples = int(segment_duration * sample_rate)
    total_segments = waveform.shape[1] // segment_samples

    for i in range(total_segments):
        segment = waveform[:, i * segment_samples : (i + 1) * segment_samples]
        output_filename = os.path.join(output_folder, f"segment_{i:03d}.wav")
        torchaudio.save(output_filename, segment, sample_rate)


def pad_or_truncate(tensor, target_length=128):
    current_length = tensor.shape[2]
    if current_length < target_length:
        tensor = F.pad(tensor, (0, target_length - current_length))
    elif current_length > target_length:
        tensor = tensor[:, :, :target_length]
    return tensor

class WavDataset(Dataset):
    def __init__(self, reference_files, test_files, transform):
        self.reference_files = reference_files
        self.test_files = test_files
        self.transform = transform

    def __len__(self):
        return min(len(self.reference_files), len(self.test_files))

    def __getitem__(self, idx):
        ref_file = self.reference_files[idx % len(self.reference_files)]
        test_file = self.test_files[idx % len(self.test_files)]

        waveform_ref, _ = torchaudio.load(ref_file)
        waveform_test, _ = torchaudio.load(test_file)

        if waveform_ref.shape[0] > 1:
            waveform_ref = torch.mean(waveform_ref, dim=0, keepdim=True)
        if waveform_test.shape[0] > 1:
            waveform_test = torch.mean(waveform_test, dim=0, keepdim=True)

        spec_ref = self.transform(waveform_ref)
        spec_test = self.transform(waveform_test)

        spec_ref = pad_or_truncate(spec_ref, target_length=128)
        spec_test = pad_or_truncate(spec_test, target_length=128)

        return spec_ref, spec_test

# -------------------- DEFINICIÓN DE LA RED --------------------
class VoiceSimilarityNet(nn.Module):
    def __init__(self):
        super(VoiceSimilarityNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Última capa 

    def forward(self, ref, test):
        ref = self.conv1(ref)
        ref = self.conv2(ref)
        ref = self.conv3(ref)
        ref = ref.view(ref.size(0), -1)
        ref = F.relu(self.fc1(ref))
        ref = F.relu(self.fc2(ref))

        test = self.conv1(test)
        test = self.conv2(test)
        test = self.conv3(test)
        test = test.view(test.size(0), -1)
        test = F.relu(self.fc1(test))
        test = F.relu(self.fc2(test))

        similarity = F.cosine_similarity(ref, test)  
        similarity = torch.sigmoid(self.fc3(similarity.unsqueeze(1)))  
        return similarity

mel_transform = torch.nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=128),
    AmplitudeToDB()
)

reference_voice_dir = "segments_ref"
test_voice_dir = "segments_test"

reference_files = [os.path.join(reference_voice_dir, f) for f in os.listdir(reference_voice_dir) if f.endswith('.wav')]
test_files = [os.path.join(test_voice_dir, f) for f in os.listdir(test_voice_dir) if f.endswith('.wav')]

dataset = WavDataset(reference_files, test_files, mel_transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = VoiceSimilarityNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()  # Binary Cross Entropy

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for ref_spec, test_spec in dataloader:
        similarity = model(ref_spec, test_spec)
        target = torch.ones_like(similarity)  
        
        loss = loss_fn(similarity, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss / len(dataloader):.4f}")


model.eval()
with torch.no_grad():
    for i in range(len(test_files)):
        ref_spec, test_spec = dataset[i]
        similarity = model(ref_spec.unsqueeze(0), test_spec.unsqueeze(0))
        print(f"Similitud entre '{reference_files[i % len(reference_files)]}' y '{test_files[i]}': {similarity.item():.2f}")