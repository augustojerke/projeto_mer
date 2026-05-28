# predict.py
"""
Prevê a emoção de qualquer música usando os 3 modelos treinados (MEL, STFT, MFCC).

Uso:
    python predict.py --audio musica.mp3
    python predict.py --audio musica.mp3 --plot
    python predict.py --audio musica.mp3 --arch resnet18
"""

import argparse
import glob
import os
import sys
from collections import Counter

import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset import transforms, SAMPLE_RATE, JANELA_SAMPLES, TRI_TARGET_F, _stats, _resize_freq, STFT_TARGET_F
from model import build_model

# ─────────────────────────────────────────
# Argumentos
# ─────────────────────────────────────────
parser = argparse.ArgumentParser(description="MER Predict — prevê emoção com os 3 espectrogramas")
parser.add_argument("--audio", type=str, required=True, help="Caminho do arquivo (.mp3/.wav)")
parser.add_argument("--arch",  type=str, default="resnet18", choices=["cnn", "resnet18"])
parser.add_argument("--plot",  action="store_true", help="Gera gráfico e salva PNG")
args = parser.parse_args()

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOP_SAMPLES = int(0.5 * SAMPLE_RATE)
MODOS       = ["mel", "stft", "mfcc"]

QUADRANTES = {
    (True,  True):  "Exaltado (A+V+)",
    (True,  False): "Irritado (A+V-)",
    (False, True):  "Calmo (A-V+)",
    (False, False): "Triste (A-V-)",
}
DESCRICOES = {
    "Exaltado (A+V+)": "Alta energia, tom positivo — alegre, animado, eufórico",
    "Irritado (A+V-)":  "Alta energia, tom negativo — tenso, agressivo, agitado",
    "Calmo (A-V+)":     "Baixa energia, tom positivo — relaxado, sereno, tranquilo",
    "Triste (A-V-)":    "Baixa energia, tom negativo — melancólico, triste, depressivo",
}
CORES_Q = {
    "Exaltado (A+V+)": "#FFD600",
    "Irritado (A+V-)":  "#F44336",
    "Calmo (A-V+)":     "#4CAF50",
    "Triste (A-V-)":    "#2196F3",
}
CORES_MODO = {"mel": "#2196F3", "stft": "#4CAF50", "mfcc": "#FF9800"}

# ─────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────
def encontrar_checkpoint(arch, modo):
    for padrao in [f"checkpoints/{arch}_{modo}_dinamico_*_best.pt",
                   f"checkpoints/{arch}_{modo}_*_best.pt"]:
        candidatos = sorted(glob.glob(padrao))
        if candidatos:
            for c in candidatos:
                if "fold1" in c:
                    return c
            return candidatos[0]
    return None

# ─────────────────────────────────────────
# Feature de um trecho
# ─────────────────────────────────────────
def extrair_feature(trecho, modo):
    if len(trecho) < JANELA_SAMPLES:
        trecho = np.pad(trecho, (0, JANELA_SAMPLES - len(trecho)))
    wav  = torch.tensor(trecho[:JANELA_SAMPLES]).unsqueeze(0)
    feat = transforms[modo](wav)
    if modo == "stft":
        feat = _resize_freq(feat, STFT_TARGET_F)
    feat = (feat - _stats.get(modo, {}).get("mean", 0.0)) / (_stats.get(modo, {}).get("std", 1.0) + 1e-6)
    return feat

# ─────────────────────────────────────────
# Carregar áudio
# ─────────────────────────────────────────
if not os.path.exists(args.audio):
    print(f"Arquivo não encontrado: {args.audio}")
    sys.exit(1)

print(f"\nCarregando: {os.path.basename(args.audio)}")
waveform, _ = librosa.load(args.audio, sr=SAMPLE_RATE, mono=True)
duracao_s   = len(waveform) / SAMPLE_RATE

inícios  = list(range(0, max(1, len(waveform) - JANELA_SAMPLES + 1), HOP_SAMPLES))
tempos_s = [i / SAMPLE_RATE for i in inícios]
if not inícios:
    inícios = [0]; tempos_s = [0.0]

print(f"Duração: {duracao_s:.1f}s  |  {len(inícios)} frames analisados")

# ─────────────────────────────────────────
# Inferência — 3 modelos
# ─────────────────────────────────────────
resultados = {}   # modo → {"arousals", "valences"}

for modo in MODOS:
    ckpt = encontrar_checkpoint(args.arch, modo)
    if ckpt is None:
        print(f"  [{modo.upper()}] checkpoint não encontrado — pulando")
        continue

    modelo = build_model(args.arch, modo).to(DEVICE)
    modelo.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    modelo.eval()

    feats = [extrair_feature(waveform[i:i + JANELA_SAMPLES], modo) for i in inícios]
    arousals, valences = [], []
    with torch.no_grad():
        for i in range(0, len(feats), 64):
            batch = torch.stack(feats[i:i + 64]).to(DEVICE)
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    preds = modelo(batch).cpu().float().numpy()
            else:
                preds = modelo(batch).cpu().numpy()
            arousals.extend(preds[:, 0].tolist())
            valences.extend(preds[:, 1].tolist())

    resultados[modo] = {
        "arousals": np.array(arousals),
        "valences":  np.array(valences),
        "ckpt":      os.path.basename(ckpt),
    }
    print(f"  [{modo.upper()}] OK — {ckpt}")

if not resultados:
    print("Nenhum checkpoint encontrado. Execute o treino primeiro.")
    sys.exit(1)

# ─────────────────────────────────────────
# Resultado — 1 bloco por modo + consenso
# ─────────────────────────────────────────
nome_arq = os.path.basename(args.audio)
print(f"\n{'='*60}")
print(f"  {nome_arq}")
print(f"{'='*60}")

all_a, all_v = [], []
for modo in MODOS:
    if modo not in resultados:
        continue
    r = resultados[modo]
    a = float(r["arousals"].mean())
    v = float(r["valences"].mean())
    q = QUADRANTES[(a >= 0.5, v >= 0.5)]
    all_a.append(a); all_v.append(v)

    print(f"\n  ── {modo.upper()} ({r['ckpt']})")
    print(f"     Arousal : {a:.3f}  (±{r['arousals'].std():.3f})")
    print(f"     Valence : {v:.3f}  (±{r['valences'].std():.3f})")
    print(f"     Emoção  : {q}")

    quad_frames = [QUADRANTES[(ar >= 0.5, va >= 0.5)]
                   for ar, va in zip(r["arousals"], r["valences"])]
    contagem = Counter(quad_frames)
    for qq in ["Exaltado (A+V+)", "Irritado (A+V-)", "Calmo (A-V+)", "Triste (A-V-)"]:
        c   = contagem.get(qq, 0)
        pct = 100 * c / len(quad_frames)
        bar = "█" * int(pct / 5)
        print(f"       {qq:<25} {bar:<20} {pct:5.1f}%")

# Consenso (média dos modelos disponíveis)
a_cons = float(np.mean(all_a))
v_cons = float(np.mean(all_v))
q_cons = QUADRANTES[(a_cons >= 0.5, v_cons >= 0.5)]
print(f"\n{'='*60}")
print(f"  CONSENSO (média dos {len(all_a)} modelos)")
print(f"  Arousal : {a_cons:.3f}   Valence : {v_cons:.3f}")
print(f"  Emoção  : {q_cons}")
print(f"  Descrição: {DESCRICOES[q_cons]}")
print(f"{'='*60}")

# ─────────────────────────────────────────
# Plot
# ─────────────────────────────────────────
if args.plot:
    n_modos = len(resultados)
    fig = plt.figure(figsize=(16, 4 + 3 * n_modos))
    gs  = fig.add_gridspec(n_modos + 1, 2, hspace=0.45, wspace=0.35)

    # Curvas temporais — 1 linha por modo
    for row, modo in enumerate([m for m in MODOS if m in resultados]):
        r  = resultados[modo]
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(tempos_s, r["arousals"], color="#E53935", lw=1.4, label="Arousal")
        ax.plot(tempos_s, r["valences"],  color="#1E88E5", lw=1.4, label="Valence")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.fill_between(tempos_s, r["arousals"], 0.5, alpha=0.10, color="#E53935")
        ax.fill_between(tempos_s, r["valences"],  0.5, alpha=0.10, color="#1E88E5")
        ax.set_xlim(tempos_s[0], tempos_s[-1])
        ax.set_ylim(0, 1)
        a_m = r["arousals"].mean(); v_m = r["valences"].mean()
        ax.set_title(f"{modo.upper()} — Arousal: {a_m:.3f}  Valence: {v_m:.3f}", fontweight="bold")
        ax.set_ylabel("Valor [0,1]"); ax.legend(fontsize=8)
        if row == n_modos - 1:
            ax.set_xlabel("Tempo (s)")

        # Plano Russell
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.scatter(r["valences"], r["arousals"],
                    c=tempos_s, cmap="plasma", s=12, alpha=0.6)
        ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax2.axvline(0.5, color="gray", ls="--", lw=0.8)
        ax2.scatter([v_m], [a_m], color="red", s=150, zorder=5, marker="*",
                    label=f"Média  A={a_m:.2f}  V={v_m:.2f}")
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.set_xlabel("Valence →"); ax2.set_ylabel("Arousal →")
        ax2.set_title(f"{modo.upper()} — Plano Russell", fontweight="bold")
        for (af, vf), nome in QUADRANTES.items():
            ax2.text(0.72 if vf else 0.02, 0.72 if af else 0.02,
                     nome.split("(")[0].strip(), fontsize=8, color="gray")
        ax2.legend(fontsize=8)

    # Consenso — comparação de barras
    ax_bar = fig.add_subplot(gs[n_modos, :])
    modos_ok  = [m for m in MODOS if m in resultados]
    a_vals    = [resultados[m]["arousals"].mean() for m in modos_ok]
    v_vals    = [resultados[m]["valences"].mean()  for m in modos_ok]
    x         = np.arange(len(modos_ok))
    w         = 0.35
    bars_a    = ax_bar.bar(x - w/2, a_vals, w, label="Arousal", color="#E53935", alpha=0.85)
    bars_v    = ax_bar.bar(x + w/2, v_vals, w, label="Valence",  color="#1E88E5", alpha=0.85)
    ax_bar.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels([m.upper() for m in modos_ok], fontsize=12)
    ax_bar.set_ylim(0, 1); ax_bar.set_ylabel("Valor médio")
    ax_bar.set_title(f"Consenso dos modelos — {q_cons}  |  A={a_cons:.3f}  V={v_cons:.3f}",
                     fontweight="bold")
    ax_bar.legend()
    for bar, v in zip(bars_a, a_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    for bar, v in zip(bars_v, v_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    fig.suptitle(f"Análise Emocional — {nome_arq}", fontsize=14, fontweight="bold", y=1.01)
    out_png = f"predict_{os.path.splitext(nome_arq)[0]}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGráfico salvo em: {out_png}")
