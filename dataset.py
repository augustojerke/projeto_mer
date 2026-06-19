import os
import json
import torch
import torchaudio.transforms as T
import librosa
import numpy as np
from torch.utils.data import Dataset

SAMPLE_RATE    = 22050
JANELA_SEG     = 4.0
JANELA_SAMPLES = int(JANELA_SEG * SAMPLE_RATE)
N_MELS         = 128
N_MFCC         = 13
N_FFT          = 1024
HOP_LENGTH     = 512
TRI_TARGET_F   = 128  
STFT_TARGET_F  = 128 

transforms = {
    "stft": torch.nn.Sequential(
        T.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH),
        T.AmplitudeToDB(stype="power") 
    ),
    "mel": torch.nn.Sequential(
        T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                         hop_length=HOP_LENGTH, n_mels=N_MELS),
        T.AmplitudeToDB(stype="power")
    ),
    "mfcc": T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS}
    ),
}

STATS_PATH = "data/stats.json"
_stats = {}
if os.path.exists(STATS_PATH):
    with open(STATS_PATH) as f:
        _stats = json.load(f)


def _resize_freq(feat: torch.Tensor, target_f: int) -> torch.Tensor:
    """Redimensiona dimensão de frequência de um tensor 1×F×T para 1×target_f×T."""
    if feat.shape[1] == target_f:
        return feat
    return torch.nn.functional.interpolate(
        feat.unsqueeze(0), size=(target_f, feat.shape[2]),
        mode='bilinear', align_corners=False
    ).squeeze(0)


def _extrair_trecho(waveform, frame_time):
    centro = int(frame_time * SAMPLE_RATE)
    inicio = max(0, centro - JANELA_SAMPLES // 2)
    fim    = inicio + JANELA_SAMPLES
    if fim > len(waveform):
        fim    = len(waveform)
        inicio = max(0, fim - JANELA_SAMPLES)
    trecho = waveform[inicio:fim]
    if len(trecho) < JANELA_SAMPLES:
        trecho = np.pad(trecho, (0, JANELA_SAMPLES - len(trecho)))
    return trecho



def carregar_feature_tri(music_id, frame_time, audio_map):
    """Retorna tensor 3×TRI_TARGET_F×T com [STFT, log-Mel, MFCC] empilhados."""
    waveform, _ = librosa.load(audio_map[music_id], sr=SAMPLE_RATE, mono=True)
    trecho = _extrair_trecho(waveform, frame_time)
    wav    = torch.tensor(trecho).unsqueeze(0)
    specs  = [_resize_freq(transforms[m](wav), TRI_TARGET_F) for m in ("stft", "mel", "mfcc")]
    return torch.cat(specs, dim=0) 



def augmentar(feat):
    feat   = feat.clone()
    fill   = feat.mean().item()
    n_freq = feat.shape[1]
    n_time = feat.shape[2]

    f_mask  = max(1, int(n_freq * 0.15))
    f_start = torch.randint(0, max(1, n_freq - f_mask), (1,)).item()
    feat[:, f_start:f_start + f_mask, :] = fill

    t_mask  = max(1, int(n_time * 0.15))
    t_start = torch.randint(0, max(1, n_time - t_mask), (1,)).item()
    feat[:, :, t_start:t_start + t_mask] = fill

    feat += torch.randn_like(feat) * (feat.std().item() * 0.05)

    return feat


def _quadrante(a, v):
    if a >= 0.5 and v >= 0.5: return 0  
    if a >= 0.5 and v <  0.5: return 1  
    if a <  0.5 and v >= 0.5: return 2  
    return 3                             

SUBSAMPLE_STEPS = {0: 4, 1: 2, 2: 2, 3: 3}  


def carregar_feature(music_id, frame_time, audio_map, modo="mel"):
    waveform, _ = librosa.load(audio_map[music_id], sr=SAMPLE_RATE, mono=True)
    trecho      = _extrair_trecho(waveform, frame_time)
    return transforms[modo](torch.tensor(trecho).unsqueeze(0))


class DEAMDinamicoDataset(Dataset):
    def __init__(self, dataframe, audio_map=None, modo="mel",
                 processed_dir="data\\processed", augment=False, preload=True):
        self.df            = dataframe.reset_index(drop=True)
        self.modo          = modo
        self.audio_map     = audio_map
        self.processed_dir = processed_dir
        self.augment       = augment
        self.mean          = _stats.get(modo, {}).get("mean", 0.0)
        self.std           = _stats.get(modo, {}).get("std", 1.0)
        self.cache_dir     = os.path.join(processed_dir, modo)
        _first_mid = int(self.df['musicId'].iloc[0])
        self.use_cache = os.path.exists(
            os.path.join(self.cache_dir, f"{_first_mid}.pt")
        )
        self._mem            = {}
        self._mem_normalized = False
        
        mids = self.df['musicId'].astype(int).values
        fts  = self.df['frameTime'].astype(float).values
        self._arousal = self.df['Arousal(mean)'].to_numpy(dtype=np.float32)
        self._valence = self.df['Valence(mean)'].to_numpy(dtype=np.float32)
        self._keys = [f"{int(m)}_{str(float(f)).replace('.', '_')}"
                      for m, f in zip(mids, fts)]

        if self.use_cache and preload:
            music_ids = self.df['musicId'].unique()
            print(f"  Carregando cache '{modo}' na RAM ({len(music_ids)} músicas)...", flush=True)
            for mid in music_ids:
                path = os.path.join(self.cache_dir, f"{int(mid)}.pt")
                try:
                    song = torch.load(path, weights_only=True)
                    if self.modo == "stft":
                        song = {k: _resize_freq(v.float(), STFT_TARGET_F) for k, v in song.items()}
                    self._mem.update(song)
                except Exception:
                    pass
            print(f"  {len(self._mem)} frames em RAM.", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        shapes = {"stft": (1, STFT_TARGET_F, 173),
                  "mel":  (1, N_MELS, 173),
                  "mfcc": (1, N_MFCC, 173),
                  "tri":  (3, TRI_TARGET_F, 173)}
        key = self._keys[idx]
        try:
            if self._mem:
                feat = self._mem.get(key)
                feat = feat.float().clone() if feat is not None else torch.zeros(shapes[self.modo])
            elif self.use_cache:
                song_path = os.path.join(self.cache_dir,
                                         f"{key.split('_')[0]}.pt")
                song_dict = torch.load(song_path, weights_only=True)
                feat = song_dict[key].clone()
            else:
                row      = self.df.iloc[idx]
                music_id = int(row['musicId'])
                frame_t  = float(row['frameTime'])
                if self.modo == "tri":
                    feat = carregar_feature_tri(music_id, frame_t, self.audio_map)
                else:
                    feat = carregar_feature(music_id, frame_t, self.audio_map, self.modo)
        except Exception:
            feat = torch.zeros(shapes[self.modo])

        # Normaliza só se ainda não foi feito no preload
        if not self._mem_normalized:
            if self.modo == "tri":
                for i, m in enumerate(("stft", "mel", "mfcc")):
                    feat[i] = (feat[i] - _stats.get(m, {}).get("mean", 0.0)) / (_stats.get(m, {}).get("std", 1.0) + 1e-6)
            else:
                feat = (feat - self.mean) / (self.std + 1e-6)

        if self.augment:
            feat = augmentar(feat)

        label = torch.tensor(
            [self._arousal[idx], self._valence[idx]],
            dtype=torch.float32
        )
        return feat, label

