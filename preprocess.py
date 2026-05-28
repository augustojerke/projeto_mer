# preprocess.py — salva 1 arquivo por música (dict de frames)
# 390k arquivos → 5.4k arquivos = sem thrashing de I/O no SSD

import os
import glob
import torch
import pandas as pd
import librosa
from multiprocessing import Pool, cpu_count

from dataset import (
    SAMPLE_RATE, TRI_TARGET_F,
    transforms, _extrair_trecho, _resize_freq,
)

AUDIO_DIR = "data/deam/audio"
CSV_PATH  = "data/dynamic_annotations.csv"
SAVE_DIR  = "data/processed"
MODOS     = ["stft", "mel", "mfcc"]
N_WORKERS = cpu_count()

os.makedirs(SAVE_DIR, exist_ok=True)

audio_map = {}
for _fp in glob.glob(os.path.join(AUDIO_DIR, "*.mp3")):
    audio_map[int(os.path.basename(_fp).replace(".mp3", ""))] = _fp


def _frame_key(music_id: int, frame_time: float) -> str:
    return f"{music_id}_{str(frame_time).replace('.', '_')}"


def processar_musica(args):
    """Gera data/processed/{modo}/{musicId}.pt com todos os frames da música."""
    music_id, frame_times, modo, save_dir = args

    save_path = os.path.join(save_dir, f"{music_id}.pt")
    if os.path.exists(save_path):
        return 1, 0   # já existe, pula

    try:
        waveform, _ = librosa.load(audio_map[music_id], sr=SAMPLE_RATE, mono=True)

        trechos = [
            torch.tensor(_extrair_trecho(waveform, ft), dtype=torch.float32)
            for ft in frame_times
        ]
        batch = torch.stack(trechos).unsqueeze(1)   # (N, 1, JANELA_SAMPLES)

        with torch.no_grad():
            if modo == "tri":
                specs = [_resize_freq(transforms[m](batch), TRI_TARGET_F)
                         for m in ("stft", "mel", "mfcc")]
                feats = torch.cat(specs, dim=1)                    # (N, 3, TRI_TARGET_F, T)
            else:
                feats = transforms[modo](batch)                    # (N, 1, F, T)

        # Um único torch.save com dict {frame_key: tensor}
        song_dict = {_frame_key(music_id, ft): feats[i]
                     for i, ft in enumerate(frame_times)}
        torch.save(song_dict, save_path)

        return 1, len(frame_times)

    except Exception as e:
        print(f"  ERRO musicId={music_id}: {e}")
        return 0, 0


if __name__ == "__main__":
    dataset = pd.read_csv(CSV_PATH)
    grupos  = [(mid, grp["frameTime"].tolist())
               for mid, grp in dataset.groupby("musicId")]

    print(f"CPUs: {cpu_count()} | Workers: {N_WORKERS}")
    print(f"Músicas: {len(grupos)} | Frames: {len(dataset)}")
    print(f"Arquivos a gerar: {len(grupos) * len(MODOS)} (1 por música por modo)\n")

    for modo in MODOS:
        modo_dir = os.path.join(SAVE_DIR, modo)
        os.makedirs(modo_dir, exist_ok=True)

        args_list = [(mid, fts, modo, modo_dir) for mid, fts in grupos]

        total_musicas = total_frames = 0
        print(f"[{modo.upper()}] {len(grupos)} músicas...")

        with Pool(processes=N_WORKERS) as pool:
            for i, (n_mus, n_frames) in enumerate(
                pool.imap_unordered(processar_musica, args_list, chunksize=16)
            ):
                total_musicas += n_mus
                total_frames  += n_frames
                if (i + 1) % 300 == 0 or (i + 1) == len(grupos):
                    pct = (i + 1) / len(grupos) * 100
                    print(f"  {i+1}/{len(grupos)} ({pct:.0f}%) | "
                          f"{total_frames} frames salvos", flush=True)

        print(f"  {modo.upper()} concluído\n")

    print("Pré-processamento completo!")
