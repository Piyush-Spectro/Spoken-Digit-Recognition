import os
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
import soundfile as sf

class DigitAudioDataset(Dataset):
    """
    Custom PyTorch Dataset for Spoken Digit Recognition.
    Converts 1D audio waveforms into 2D Log-Mel Spectrograms
    and applies SpecAugment data augmentation during training.
    """
    def __init__(self, df, data_dir, sample_rate=16000, max_length=16000, n_mels=64, is_train=True):
        self.df = df
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.is_train = is_train

        # Audio Feature Extraction (Mel Spectrogram)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        # SpecAugment Data Augmentation (Time & Frequency Masking)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, f"{row['id']}.wav")

        # Load audio signal via soundfile
        audio_array, sr = sf.read(file_path)

        if audio_array.ndim == 1:
            waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        else:
            waveform = torch.tensor(audio_array, dtype=torch.float32).t()
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to uniform sample rate
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Pad or truncate to uniform 1 second length
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Extract Mel Spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)

        # Apply SpecAugment during training phase
        if self.is_train:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)

        # Standard score normalization
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        if 'label' in row:
            return mel_spec, torch.tensor(row['label'], dtype=torch.long)
        else:
            return mel_spec, row['id']
