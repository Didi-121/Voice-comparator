import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from audio_handler import AudioHandler  # Importamos solo el AudioHandler


# --- 1. Definición de la Red CNN ---
class VoiceSimilarityCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Bloque 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.2)
        
        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.2)
        
        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.2)
        
        # Capas densas (asume espectrogramas de 128x128 -> 16x16 después de 3 MaxPools)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.drop4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.flatten(x)
        x = self.drop4(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

# --- 2. Dataset y DataLoader ---
class VoiceDataset(Dataset):
    def __init__(self, voice_dir, negative_dir=None, audio_handler=None):
        self.audio_handler = audio_handler or AudioHandler()
        self.voice_files = [os.path.join(voice_dir, f) for f in os.listdir(voice_dir) if f.endswith('.wav')]
        
        # Usar 20% de los audios como negativos si no hay carpeta específica
        self.negative_files = (
            [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.wav')]
            if negative_dir
            else self.voice_files[:len(self.voice_files) // 5]
        )
        
        self.all_files = self.voice_files + self.negative_files
        self.labels = [1] * len(self.voice_files) + [0] * len(self.negative_files)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        audio_path = self.all_files[idx]
        spectrogram = self.audio_handler.process_audio_file(audio_path)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return spectrogram, label

# --- 3. Entrenamiento + Gráficos ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    # Configuración
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    
    # Inicializar modelo, loss y optimizador
    model = VoiceSimilarityCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Datos
    dataset = VoiceDataset(
        voice_dir="training_data",
        negative_dir=None,  # Opcional: "negative_data" si tienes ejemplos negativos
        audio_handler=AudioHandler()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Listas para métricas
    train_loss = []
    train_accuracy = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for spectrograms, labels in dataloader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            # Calcular accuracy
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()
        
        # Guardar métricas
        epoch_loss /= len(dataloader)
        epoch_accuracy = correct / total
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        
        print(f"Época {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2%}")
    
    # Guardar modelo
    torch.save(model.state_dict(), "voice_similarity_model.pth")
    
    # Graficar
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, 'r-', label='Loss')
    plt.xlabel('Época')
    plt.title('Pérdida durante el entrenamiento')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, 'b-', label='Accuracy')
    plt.xlabel('Época')
    plt.title('Accuracy durante el entrenamiento')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == "__main__":
    train_model()