# compute_stats.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import DEAMDinamicoDataset
import pandas as pd
import glob
import os
import json
from sklearn.model_selection import train_test_split

AUDIO_DIR = "data/deam/audio"
CSV_PATH  = "data/dynamic_annotations.csv"
MODOS     = ["stft", "mel", "mfcc"]

audio_map = {}
for filepath in glob.glob(os.path.join(AUDIO_DIR, "*.mp3")):
    music_id = int(os.path.basename(filepath).replace(".mp3", ""))
    audio_map[music_id] = filepath

df = pd.read_csv(CSV_PATH)

# Split por musicId para evitar vazamento de dados (usa só treino para calcular stats)
music_ids = df["musicId"].unique()
train_ids, _ = train_test_split(music_ids, test_size=0.1, random_state=3)
df_train = df[df["musicId"].isin(train_ids)].reset_index(drop=True)

print(f"Total de frames para stats: {len(df_train)} (de {df_train['musicId'].nunique()} musicas)\n")

stats = {}
for modo in MODOS:
    print(f"Calculando stats para {modo}...")
    ds     = DEAMDinamicoDataset(df_train, audio_map, modo)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    mean, std, n = 0.0, 0.0, 0
    for features, _ in loader:
        mean += features.mean().item()
        n    += 1
    mean /= n

    for features, _ in loader:
        std += ((features - mean) ** 2).mean().item()
    std = (std / n) ** 0.5

    stats[modo] = {"mean": round(mean, 6), "std": round(std, 6)}
    print(f"  {modo}: mean={mean:.4f}  std={std:.4f}")

with open("data/stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("\nStats salvas em data/stats.json")
