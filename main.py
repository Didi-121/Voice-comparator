import torch
from audio_handler import AudioHandler
from trainer import VoiceSimilarityCNN  

class VoiceSimilarityPredictor:
    def __init__(self, model_path=r"models/voice_similarity_model_best.pth"):
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
        spectrogram = spectrogram.unsqueeze(0).to(self.device)  
        # Predecir
        with torch.no_grad():
            similarity = self.model(spectrogram).item()
        return similarity * 100  

if __name__ == "__main__":
    predictor = VoiceSimilarityPredictor()
    similarity = predictor.predict("test.wav")
    print(f"\nPorcentaje de similitud con la voz objetivo: {similarity:.2f}%")