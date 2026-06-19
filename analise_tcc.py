import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

from dataset import DEAMDinamicoDataset
from model import build_model

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIO_DIR = "data/deam/audio"
CSV_PATH  = "data/dynamic_annotations.csv"
CKPT_DIR  = "checkpoints"
OUT_DIR   = "analise_tcc"
os.makedirs(OUT_DIR, exist_ok=True)

MODOS = ["mel", "mfcc", "stft"]
ARCH  = "resnet18"
BATCH = 512

audio_map = {}
for fp in glob.glob(os.path.join(AUDIO_DIR, "*.mp3")):
    mid = int(os.path.basename(fp).replace(".mp3", ""))
    audio_map[mid] = fp

df        = pd.read_csv(CSV_PATH)
music_ids = df["musicId"].unique()
rng       = np.random.default_rng(3)
ids_shuf  = rng.permutation(music_ids)
n_test    = round(len(ids_shuf) * 0.10)
ids_test  = ids_shuf[:n_test]
ids_cv    = ids_shuf[n_test:]
df_test   = df[df["musicId"].isin(ids_test)].reset_index(drop=True)
df_cv     = df[df["musicId"].isin(ids_cv)].reset_index(drop=True)

def _ccc(p, l):
    p, l  = np.array(p), np.array(l)
    cov   = np.mean((p - p.mean()) * (l - l.mean()))
    return 2 * cov / (p.var() + l.var() + (p.mean() - l.mean()) ** 2 + 1e-8)

def _pearson(p, l):
    p, l = np.array(p), np.array(l)
    return float(np.corrcoef(p, l)[0, 1])

def _rmse(p, l):
    return float(np.sqrt(np.mean((np.array(p) - np.array(l)) ** 2)))

from sklearn.model_selection import KFold
kf       = KFold(n_splits=5, shuffle=True, random_state=3)
ids_cv_a = np.array(ids_cv)

print("=" * 60)
print("Carregando checkpoints e gerando predições...")

fold_cccs   = {m: [] for m in MODOS} 
test_preds  = {}                        

for modo in MODOS:
    print(f"\n  [{modo.upper()}]")
    preload = True

    for fold, (tr_idx, val_idx) in enumerate(kf.split(ids_cv_a)):
        ids_v  = ids_cv_a[val_idx]
        df_v   = df_cv[df_cv["musicId"].isin(ids_v)].reset_index(drop=True)
        ckpt   = os.path.join(CKPT_DIR, f"{ARCH}_{modo}_dinamico_fold{fold+1}_best.pt")
        if not os.path.exists(ckpt):
            print(f"    fold{fold+1}: checkpoint não encontrado, pulando")
            continue

        modelo = build_model(ARCH, modo).to(DEVICE)
        modelo.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        modelo.eval()

        ds  = DEAMDinamicoDataset(df_v, audio_map, modo, augment=False, preload=preload)
        dl  = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
        all_p, all_l = [], []
        with torch.no_grad():
            for feats, labels in dl:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = modelo(feats.to(DEVICE)).cpu().float()
                all_p.append(preds)
                all_l.append(labels)
        p = torch.cat(all_p).numpy()
        l = torch.cat(all_l).numpy()
        ccc_a = _ccc(p[:, 0], l[:, 0])
        ccc_v = _ccc(p[:, 1], l[:, 1])
        ccc_m = (ccc_a + ccc_v) / 2
        fold_cccs[modo].append(ccc_m)
        print(f"    fold{fold+1}: CCC A={ccc_a:.4f} V={ccc_v:.4f} media={ccc_m:.4f}")

    best_fold = int(np.argmax(fold_cccs[modo])) + 1
    ckpt_best = os.path.join(CKPT_DIR, f"{ARCH}_{modo}_dinamico_fold{best_fold}_best.pt")
    modelo = build_model(ARCH, modo).to(DEVICE)
    modelo.load_state_dict(torch.load(ckpt_best, map_location=DEVICE, weights_only=True))
    modelo.eval()

    ds_test = DEAMDinamicoDataset(df_test, audio_map, modo, augment=False, preload=preload)
    dl_test = DataLoader(ds_test, batch_size=BATCH, shuffle=False, num_workers=0)
    all_p, all_l = [], []
    with torch.no_grad():
        for feats, labels in dl_test:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = modelo(feats.to(DEVICE)).cpu().float()
            all_p.append(preds)
            all_l.append(labels)
    p = torch.cat(all_p).numpy()
    l = torch.cat(all_l).numpy()
    test_preds[modo] = {"pred": p, "true": l}
    print(f"    Teste (fold{best_fold}): CCC A={_ccc(p[:,0],l[:,0]):.4f} V={_ccc(p[:,1],l[:,1]):.4f}")

print("\n" + "=" * 60)
print("BASELINE TRIVIAL (predição = média do treino)")
mean_a = df_cv["Arousal(mean)"].mean()
mean_v = df_cv["Valence(mean)"].mean()
print(f"  Média treino — Arousal: {mean_a:.4f}  Valence: {mean_v:.4f}")

true_a = test_preds["mel"]["true"][:, 0]
true_v = test_preds["mel"]["true"][:, 1]
base_a = np.full_like(true_a, mean_a)
base_v = np.full_like(true_v, mean_v)

print(f"  CCC     A:{_ccc(base_a, true_a):.4f}  V:{_ccc(base_v, true_v):.4f}")
print(f"  Pearson A:{_pearson(base_a, true_a):.4f}  V:{_pearson(base_v, true_v):.4f}")
print(f"  RMSE    A:{_rmse(base_a, true_a):.4f}  V:{_rmse(base_v, true_v):.4f}")

print("\n  Ganho sobre baseline (CCC — quanto cada modo melhora):")
for modo in MODOS:
    p = test_preds[modo]["pred"]
    l = test_preds[modo]["true"]
    ccc_a = _ccc(p[:,0], l[:,0])
    ccc_v = _ccc(p[:,1], l[:,1])
    ganho_a = ccc_a - _ccc(base_a, true_a)
    ganho_v = ccc_v - _ccc(base_v, true_v)
    print(f"  {modo.upper():4s}: Arousal +{ganho_a:.4f}  Valence +{ganho_v:.4f}")

print("\n" + "=" * 60)
print("TESTE DE WILCOXON (CCC médio por fold, 5 amostras)")
pairs = [("mel", "mfcc"), ("mel", "stft"), ("mfcc", "stft")]
for a, b in pairs:
    va = fold_cccs[a]
    vb = fold_cccs[b]
    if len(va) < 2 or len(vb) < 2:
        print(f"  {a} vs {b}: dados insuficientes")
        continue
    diffs = np.array(va) - np.array(vb)
    if np.all(diffs == 0):
        print(f"  {a} vs {b}: diferenças nulas, não aplicável")
        continue
    stat, p = wilcoxon(va, vb, alternative="two-sided", zero_method="wilcox")
    sig = "*** SIGNIFICATIVO (p<0.05)" if p < 0.05 else "(não significativo)"
    med_a, med_b = np.mean(va), np.mean(vb)
    print(f"  {a.upper()} ({med_a:.4f}) vs {b.upper()} ({med_b:.4f}): W={stat:.1f} p={p:.4f} {sig}")

print("\n" + "=" * 60)
print("Gerando scatter plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Predito × Real — Arousal (linha sup.) e Valence (linha inf.)", fontsize=14)

CORES = {"mel": "#2196F3", "mfcc": "#FF9800", "stft": "#4CAF50"}
NOMES = {"mel": "MEL", "mfcc": "MFCC", "stft": "STFT"}

for col, modo in enumerate(MODOS):
    p = test_preds[modo]["pred"]
    l = test_preds[modo]["true"]
    for row, (dim, nome_dim) in enumerate([(0, "Arousal"), (1, "Valence")]):
        ax   = axes[row][col]
        pred = p[:, dim]
        true = l[:, dim]
        ccc  = _ccc(pred, true)
        r    = _pearson(pred, true)
        rmse = _rmse(pred, true)

        ax.scatter(true, pred, alpha=0.15, s=6, color=CORES[modo])
        ax.plot([0, 1], [0, 1], "r--", lw=1.2, label="ideal")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"Real ({nome_dim})", fontsize=9)
        ax.set_ylabel(f"Predito ({nome_dim})", fontsize=9)
        ax.set_title(f"{NOMES[modo]}\nCCC={ccc:.3f}  r={r:.3f}  RMSE={rmse:.3f}", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

plt.tight_layout()
path_scatter = os.path.join(OUT_DIR, "scatter_predito_real.png")
plt.savefig(path_scatter, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: {path_scatter}")

print("Gerando curvas de aprendizado...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Curvas de Aprendizado — Val CCC por Época (melhor fold)", fontsize=13)

for modo in MODOS:
    path_json = f"resultados_{ARCH}_{modo}_dinamico.json"
    if not os.path.exists(path_json):
        continue
    with open(path_json) as f:
        data = json.load(f)
    epocas = list(data.values())[0]["epocas"]
    ep     = [e["epoca"]      for e in epocas]
    ccc_a  = [e["ccc_arousal"] for e in epocas]
    ccc_v  = [e["ccc_valence"] for e in epocas]
    axes[0].plot(ep, ccc_a, label=NOMES[modo], color=CORES[modo], lw=1.8)
    axes[1].plot(ep, ccc_v, label=NOMES[modo], color=CORES[modo], lw=1.8)

for ax, title in zip(axes, ["CCC Arousal (val)", "CCC Valence (val)"]):
    ax.set_xlabel("Época"); ax.set_ylabel("CCC")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

plt.tight_layout()
path_curvas = os.path.join(OUT_DIR, "curvas_aprendizado.png")
plt.savefig(path_curvas, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Salvo: {path_curvas}")

print("\n" + "=" * 60)
print("TABELA RESUMO FINAL")
print(f"{'Modo':<6} {'CCC_A':>7} {'CCC_V':>7} {'CCC_med':>8} {'Pearson_A':>10} {'RMSE_A':>7} {'RMSE_V':>7}")
print("-" * 60)
for modo in MODOS:
    p = test_preds[modo]["pred"]
    l = test_preds[modo]["true"]
    ccc_a = _ccc(p[:,0], l[:,0])
    ccc_v = _ccc(p[:,1], l[:,1])
    pear_a = _pearson(p[:,0], l[:,0])
    rmse_a = _rmse(p[:,0], l[:,0])
    rmse_v = _rmse(p[:,1], l[:,1])
    print(f"{modo.upper():<6} {ccc_a:>7.4f} {ccc_v:>7.4f} {(ccc_a+ccc_v)/2:>8.4f} {pear_a:>10.4f} {rmse_a:>7.4f} {rmse_v:>7.4f}")

print(f"\nArquivos gerados em: {OUT_DIR}/")
print("  - scatter_predito_real.png")
print("  - curvas_aprendizado.png")
