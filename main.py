import torch
from audio_handler import AudioHandler
from trainer import VoiceSimilarityCNN  

class VoiceSimilarityPredictor:
    def __init__(self, model_path="voice_similarity_model.pth"):
        # Configuración
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_handler = AudioHandler()
        
        # Cargar modelo
        self.model = VoiceSimilarityCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Modo evaluación

    def predict(self, audio_path):
        # Procesar audio
        spectrogram = self.audio_handler.process_audio_file(audio_path)
        spectrogram = spectrogram.unsqueeze(0).to(self.device)  # Añadir dimensión de batch
        # Predecir
        with torch.no_grad():
            similarity = self.model(spectrogram).item()
        return similarity * 100  # Convertir a porcentaje

if __name__ == "__main__":
    predictor = VoiceSimilarityPredictor()
    similarity = predictor.predict("current.wav")
    print(f"\nPorcentaje de similitud con la voz objetivo: {similarity:.2f}%")