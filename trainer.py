import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from audio_handler import AudioHandler

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
        
        # Capas densas
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

class VoiceDataset(Dataset):
    def __init__(self, voice_dir, negative_dir=None, audio_handler=None):
        self.audio_handler = audio_handler or AudioHandler()
        print(f"\nCargando audios positivos de: {voice_dir}")
        self.voice_files = [os.path.join(voice_dir, f) for f in os.listdir(voice_dir) if f.endswith('.wav')]
        
        if negative_dir:
            print(f"Cargando audios negativos de: {negative_dir}")
            self.negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.wav')]
        else:
            print("Generando audios negativos a partir de positivos (20%)")
            self.negative_files = self.voice_files[:len(self.voice_files) // 5]
        
        self.all_files = self.voice_files + self.negative_files
        self.labels = [1] * len(self.voice_files) + [0] * len(self.negative_files)
        print(f"Total muestras: {len(self.all_files)} (Positivas: {len(self.voice_files)}, Negativas: {len(self.negative_files)})")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        audio_path = self.all_files[idx]
        spectrogram = self.audio_handler.process_audio_file(audio_path)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return spectrogram, label

def plot_metrics(train_loss, train_accuracy, save_path='training_metrics.png'):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, 'r-', label='Loss')
    plt.xlabel('Época')
    plt.title('Pérdida durante el entrenamiento')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, 'b-', label='Accuracy')
    plt.xlabel('Epoca')
    plt.title('Accuracy durante el entrenamiento')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nGraficas guardadas en: {save_path}")
    plt.close()

def train_model():
    try:
        # Configuración inicial
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*50}")
        print(f"Iniciando entrenamiento en dispositivo: {device}")
        print(f"{'='*50}\n")
        
        # Hiperparámetros
        batch_size = 32
        epochs = 50
        learning_rate = 0.001
        
        # Inicializar componentes
        model = VoiceSimilarityCNN().to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Carga de datos
        print("Preparando dataset...")
        dataset = VoiceDataset(
            voice_dir=r"training_data/target",
            negative_dir=r"training_data/others",
            audio_handler=AudioHandler()
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Variables para métricas
        train_loss = []
        train_accuracy = []
        best_accuracy = 0.0

        print("\nComenzando entrenamiento...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            print(f"\nÉpoca {epoch+1}/{epochs}")
            print("-"*30)
            
            for i, (spectrograms, labels) in enumerate(dataloader):
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(spectrograms)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                
                predicted = (outputs.squeeze() > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                epoch_loss += loss.item()
                
                if (i+1) % 5 == 0 or (i+1) == len(dataloader):
                    print(f"Batch {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}")
            
            epoch_loss /= len(dataloader)
            epoch_accuracy = correct / total
            train_loss.append(epoch_loss)
            train_accuracy.append(epoch_accuracy)
            
            print(f"\nResumen Época {epoch + 1}:")
            print(f"Loss promedio: {epoch_loss:.4f}")
            print(f"Accuracy: {epoch_accuracy:.2%}")
            
            # Guardar el mejor modelo
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                torch.save(model.state_dict(), r"models/voice_similarity_model_best.pth")
                print("¡Nuevo mejor modelo guardado!")
        
        # Guardado final
        torch.save(model.state_dict(), r"models/voice_similarity_model_final.pth")
        print("\nEntrenamiento completado!")
        print(f"Mejor accuracy alcanzado: {best_accuracy:.2%}")
        
        # Generar gráficas
        plot_metrics(train_loss, train_accuracy)
        
    except KeyboardInterrupt:
        torch.save(model.state_dict(), r"models/voice_similarity_model_interrupted.pth")
        print("Modelo interrumpido guardado como 'voice_similarity_model_interrupted.pth'")
        
        if len(train_loss) > 0:
            plot_metrics(train_loss, train_accuracy, 'training_metrics_interrupted.png')
        else:
            print("No hay suficientes datos para generar gráficas")
        
    except Exception as e:
        print(e)
        print(f"\nError durante el entrenamiento: {str(e)}")
        print("Intentando guardar el modelo actual...")
        if 'model' in locals():
            torch.save(model.state_dict(), "voice_similarity_model_error.pth")
            print("Modelo guardado como 'voice_similarity_model_error.pth'")

if __name__ == "__main__":
    train_model()