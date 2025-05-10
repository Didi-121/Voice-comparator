import torch
import librosa
import numpy as np
import noisereduce as nr
from torchaudio.transforms import MelSpectrogram

class AudioHandler:
    def __init__(self, target_sr=16000, n_mels=128, n_fft=2048, hop_length=512, target_length=128):
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = target_length
   
        self.mel_spec_transform = MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
    
    def load_and_preprocess(self, audio_path, duration=None):
        audio, sr = librosa.load(audio_path, sr=self.target_sr, duration=duration)
        audio = nr.reduce_noise(y=audio, sr=sr)
        audio = librosa.util.normalize(audio)
        
        return torch.from_numpy(audio).float()
    
    def audio_to_spectrogram(self, audio_tensor):
        spectrogram = self.mel_spec_transform(audio_tensor)
        
        # Convertir a dB
        spectrogram = librosa.power_to_db(spectrogram.numpy())
        spectrogram = torch.from_numpy(spectrogram).float()
        
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        spectrogram = self._adjust_length(spectrogram)
        spectrogram = spectrogram.unsqueeze(0)  # (1, n_mels, target_length)
        
        return spectrogram
    
    def _adjust_length(self, spectrogram):
        current_length = spectrogram.shape[1]
        
        if current_length < self.target_length:
            # Padding con ceros
            pad_amount = self.target_length - current_length
            spectrogram = torch.nn.functional.pad(
                spectrogram, (0, pad_amount), mode='constant', value=0
            )
        elif current_length > self.target_length:
            spectrogram = spectrogram[:, :self.target_length]
            
        return spectrogram
    
    def process_audio_file(self, audio_path):
        audio = self.load_and_preprocess(audio_path)
        spectrogram = self.audio_to_spectrogram(audio)
        return spectrogram
