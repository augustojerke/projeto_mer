# evaluate_chorus.py
"""
Testa o modelo em todas as músicas da pasta chorus usando o CSV de anotações.

Uso:
    python evaluate_chorus.py
    python evaluate_chorus.py --modo stft
    python evaluate_chorus.py --modo mel --arch resnet18 --plot
    python evaluate_chorus.py --modo mel --checkpoint checkpoints/resnet18_mel_fold1_best.pt
"""

import argparse
import glob
import os
import sys

import librosa
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

from dataset import transforms, SAMPLE_RATE, JANELA_SAMPLES, _stats, _resize_freq, TRI_TARGET_F
from model import build_model

# ─────────────────────────────────────────
# Argumentos
# ─────────────────────────────────────────
parser = argparse.ArgumentParser(description="Avaliação em massa — pasta chorus")
parser.add_argument("--audio_dir",  type=str, default="data/chorus",   help="Pasta com os .mp3")
parser.add_argument("--csv",        type=str, default="data/static_annotations.csv",
                    help="CSV de anotações (default: DEAM static, mesma escala do treino)")
parser.add_argument("--arch",       type=str, default="resnet18",       choices=["cnn","cnn3spec","resnet18"])
parser.add_argument("--modo",       type=str, default="mel",            choices=["stft","mel","mfcc","tri"])
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--plot",       action="store_true")
parser.add_argument("--out",        type=str, default=None,             help="Salva resultados em CSV (opcional)")
parser.add_argument("--calibrar",   action="store_true",
                    help="Aplica calibração linear para corrigir viés e compressão de range")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QUADRANTES = {
    (True,  True):  "Exaltado (A+V+)",
    (True,  False): "Irritado (A+V-)",
    (False, True):  "Calmo (A-V+)",
    (False, False): "Triste (A-V-)",
}
CORES_Q = {
    "Exaltado (A+V+)": "#FFD600",
    "Irritado (A+V-)":  "#F44336",
    "Calmo (A-V+)":     "#4CAF50",
    "Triste (A-V-)":    "#2196F3",
}

# ─────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────
def encontrar_checkpoint(arch, modo, tarefa):
    padroes = [
        f"checkpoints/{arch}_{modo}_{tarefa}_*_best.pt",
        f"checkpoints/{arch}_{modo}_*_best.pt",
    ]
    for padrao in padroes:
        candidatos = sorted(glob.glob(padrao))
        if candidatos:
            for c in candidatos:
                if "fold1" in c:
                    return c
            return candidatos[0]
    print(f"Nenhum checkpoint encontrado para arch={arch} modo={modo} tarefa={tarefa}")
    for f in sorted(glob.glob("checkpoints/*.pt")):
        print(f"  {f}")
    sys.exit(1)

ckpt = args.checkpoint or encontrar_checkpoint(args.arch, args.modo, "dinamico")
print(f"Checkpoint : {ckpt}")
print(f"Dispositivo: {DEVICE}")

# ─────────────────────────────────────────
# Modelo
# ─────────────────────────────────────────
modelo = build_model(args.arch, args.modo).to(DEVICE)
modelo.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
modelo.eval()
print(f"Modelo     : {args.arch} | modo: {args.modo}\n")

# ─────────────────────────────────────────
# Dados
# ─────────────────────────────────────────
df_gt = pd.read_csv(args.csv)
df_gt.columns = df_gt.columns.str.strip()

audio_map = {}
for fp in glob.glob(os.path.join(args.audio_dir, "*.mp3")):
    mid = int(os.path.basename(fp).replace(".mp3", ""))
    audio_map[mid] = fp

ids_validos = set(df_gt["musicId"].values) & set(audio_map.keys())
df_gt = df_gt[df_gt["musicId"].isin(ids_validos)].reset_index(drop=True)
print(f"Músicas com anotação e áudio: {len(df_gt)}")

# ─────────────────────────────────────────
# Extração de feature de um trecho
# ─────────────────────────────────────────
HOP_SAMPLES = int(0.5 * SAMPLE_RATE)

def extrair_feature(trecho):
    if len(trecho) < JANELA_SAMPLES:
        trecho = np.pad(trecho, (0, JANELA_SAMPLES - len(trecho)))
    wav = torch.tensor(trecho[:JANELA_SAMPLES]).unsqueeze(0)
    if args.modo == "tri":
        specs = [_resize_freq(transforms[m](wav), TRI_TARGET_F) for m in ("stft", "mel", "mfcc")]
        feat  = torch.cat(specs, dim=0)
        for i, m in enumerate(("stft", "mel", "mfcc")):
            mean = _stats.get(m, {}).get("mean", 0.0)
            std  = _stats.get(m, {}).get("std",  1.0)
            feat[i] = (feat[i] - mean) / (std + 1e-6)
    else:
        feat = transforms[args.modo](wav)
        mean = _stats.get(args.modo, {}).get("mean", 0.0)
        std  = _stats.get(args.modo, {}).get("std",  1.0)
        feat = (feat - mean) / (std + 1e-6)
    return feat

def predizer_musica(filepath):
    waveform, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    inícios = list(range(0, max(1, len(waveform) - JANELA_SAMPLES + 1), HOP_SAMPLES))
    if not inícios:
        inícios = [0]
    feats = [extrair_feature(waveform[i:i+JANELA_SAMPLES]) for i in inícios]
    batch = torch.stack(feats).to(DEVICE)
    with torch.no_grad():
        preds = modelo(batch).cpu().numpy()
    return float(preds[:, 0].mean()), float(preds[:, 1].mean())

# ─────────────────────────────────────────
# Inferência em massa
# ─────────────────────────────────────────
resultados = []
total = len(df_gt)

for i, row in df_gt.iterrows():
    mid      = int(row["musicId"])
    a_true   = float(row["Arousal(mean)"])
    v_true   = float(row["Valence(mean)"])
    filepath = audio_map[mid]

    try:
        a_pred, v_pred = predizer_musica(filepath)
    except Exception as e:
        print(f"  Erro musicId={mid}: {e}")
        continue

    quad_real = next(n for (af,vf),n in QUADRANTES.items() if (a_true>=0.5)==af and (v_true>=0.5)==vf)
    quad_pred = next(n for (af,vf),n in QUADRANTES.items() if (a_pred>=0.5)==af and (v_pred>=0.5)==vf)
    resultados.append({
        "musicId":         mid,
        "arousal_real":    round(a_true, 4),
        "valence_real":    round(v_true, 4),
        "arousal_pred":    round(a_pred, 4),
        "valence_pred":    round(v_pred, 4),
        "erro_arousal":    round(abs(a_true - a_pred), 4),
        "erro_valence":    round(abs(v_true - v_pred), 4),
        "quadrante_real":  quad_real,
        "quadrante_pred":  quad_pred,
        "acerto_quadrante": quad_real == quad_pred,
    })

    if (i + 1) % 50 == 0 or (i + 1) == total:
        print(f"  {i+1}/{total} processadas...", flush=True)

df_res = pd.DataFrame(resultados)

# ─────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────
def pearson(a, b):
    a, b = np.array(a), np.array(b)
    return np.corrcoef(a, b)[0, 1]

def ccc(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    cov  = np.cov(a, b, ddof=0)[0, 1]
    return 2 * cov / (a.var() + b.var() + (a.mean() - b.mean())**2 + 1e-8)

def rmse(a, b):
    return float(np.sqrt(np.mean((np.array(a) - np.array(b))**2)))

def mae(a, b):
    return float(np.mean(np.abs(np.array(a) - np.array(b))))

def r2(true, pred):
    true, pred = np.array(true), np.array(pred)
    ss_res = ((true - pred)**2).sum()
    ss_tot = ((true - true.mean())**2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))

# ── Calibração pós-treino (opcional) ──────────────────────────────────────
# Escala as predições para ter mesma média e std do DEAM de treino.
# Fórmula: pred_cal = (pred - pred_mean) / pred_std * deam_std + deam_mean
# Estatísticas do DEAM dynamic (calculadas no compute_stats.py)
DEAM_STATS = {
    "mel":  {"arousal": {"mean": 0.5637, "std": 0.1415}, "valence": {"mean": 0.5386, "std": 0.1256}},
    "stft": {"arousal": {"mean": 0.5637, "std": 0.1415}, "valence": {"mean": 0.5386, "std": 0.1256}},
    "mfcc": {"arousal": {"mean": 0.5637, "std": 0.1415}, "valence": {"mean": 0.5386, "std": 0.1256}},
}

if args.calibrar:
    ref = DEAM_STATS.get(args.modo, DEAM_STATS["mel"])
    for col, dim in [("arousal_pred", "arousal"), ("valence_pred", "valence")]:
        p  = df_res[col].values
        p_cal = (p - p.mean()) / (p.std() + 1e-8) * ref[dim]["std"] + ref[dim]["mean"]
        p_cal = np.clip(p_cal, 0.0, 1.0)
        df_res[col] = p_cal.round(4)
    df_res["erro_arousal"] = (df_res["arousal_real"] - df_res["arousal_pred"]).abs().round(4)
    df_res["erro_valence"] = (df_res["valence_real"] - df_res["valence_pred"]).abs().round(4)
    df_res["erro_total"]   = df_res["erro_arousal"] + df_res["erro_valence"]
    # Recalcula quadrante predito
    df_res["quadrante_pred"]    = df_res.apply(
        lambda r: next(n for (af,vf),n in QUADRANTES.items()
                       if (r["arousal_pred"]>=0.5)==af and (r["valence_pred"]>=0.5)==vf), axis=1)
    df_res["acerto_quadrante"]  = df_res["quadrante_real"] == df_res["quadrante_pred"]
    print("  [calibracao aplicada]")

a_true = df_res["arousal_real"].values
v_true = df_res["valence_real"].values
a_pred = df_res["arousal_pred"].values
v_pred = df_res["valence_pred"].values

q_true = df_res["quadrante_real"].values
q_pred = df_res["quadrante_pred"].values
nomes_q = list(QUADRANTES.values())

f1_mac = f1_score(q_true, q_pred, labels=nomes_q, average="macro", zero_division=0)
acc    = accuracy_score(q_true, q_pred)
report = classification_report(q_true, q_pred, labels=nomes_q, zero_division=0)

print(f"\n{'='*55}")
print(f"  RESULTADO — {args.arch} | {args.modo.upper()} | chorus ({len(df_res)} músicas)")
print(f"{'='*55}")
print(f"  Pearson  A: {pearson(a_true, a_pred):.4f}   V: {pearson(v_true, v_pred):.4f}")
print(f"  CCC      A: {ccc(a_true, a_pred):.4f}   V: {ccc(v_true, v_pred):.4f}")
print(f"  RMSE     A: {rmse(a_true, a_pred):.4f}   V: {rmse(v_true, v_pred):.4f}")
print(f"  MAE      A: {mae(a_true, a_pred):.4f}   V: {mae(v_true, v_pred):.4f}")
print(f"  R²       A: {r2(a_true, a_pred):.4f}   V: {r2(v_true, v_pred):.4f}")
print(f"  Acurácia quadrante: {acc:.4f}  |  F1-macro: {f1_mac:.4f}")
print(f"\n{report}")

# ── Distribuição das predições ──────────────────────────────────────────────
print(f"  Distribuição predições (min/media/max):")
print(f"    Arousal pred: {a_pred.min():.3f} / {a_pred.mean():.3f} / {a_pred.max():.3f}")
print(f"    Valence pred: {v_pred.min():.3f} / {v_pred.mean():.3f} / {v_pred.max():.3f}")
print(f"  Distribuição labels reais:")
print(f"    Arousal real: {a_true.min():.3f} / {a_true.mean():.3f} / {a_true.max():.3f}")
print(f"    Valence real: {v_true.min():.3f} / {v_true.mean():.3f} / {v_true.max():.3f}")

# ── Piores e melhores predições ─────────────────────────────────────────────
df_res["erro_total"] = df_res["erro_arousal"] + df_res["erro_valence"]
print(f"\n  5 músicas com MAIOR erro:")
print(df_res.nlargest(5, "erro_total")[
    ["musicId","arousal_real","valence_real","arousal_pred","valence_pred","quadrante_real","quadrante_pred"]
].to_string(index=False))
print(f"\n  5 músicas com MENOR erro:")
print(df_res.nsmallest(5, "erro_total")[
    ["musicId","arousal_real","valence_real","arousal_pred","valence_pred","quadrante_real","quadrante_pred"]
].to_string(index=False))

# ── Salvar CSV sempre ───────────────────────────────────────────────────────
csv_path = args.out or f"comparacao_{args.arch}_{args.modo}.csv"
df_res.sort_values("musicId").to_csv(csv_path, index=False)
print(f"\nCSV de comparação salvo em: {csv_path}")

# ─────────────────────────────────────────
# Plots (opcional)
# ─────────────────────────────────────────
if args.plot:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Scatter Arousal: pred vs true
    axes[0][0].scatter(a_true, a_pred, alpha=0.4, s=15, color="#E53935")
    axes[0][0].plot([0,1],[0,1], 'k--', linewidth=1)
    axes[0][0].set_xlabel("Arousal Real")
    axes[0][0].set_ylabel("Arousal Predito")
    axes[0][0].set_title(f"Arousal  Pearson={pearson(a_true,a_pred):.3f}  CCC={ccc(a_true,a_pred):.3f}")

    # 2. Scatter Valence: pred vs true
    axes[0][1].scatter(v_true, v_pred, alpha=0.4, s=15, color="#1E88E5")
    axes[0][1].plot([0,1],[0,1], 'k--', linewidth=1)
    axes[0][1].set_xlabel("Valence Real")
    axes[0][1].set_ylabel("Valence Predito")
    axes[0][1].set_title(f"Valence  Pearson={pearson(v_true,v_pred):.3f}  CCC={ccc(v_true,v_pred):.3f}")

    # 3. Plano Russell — Real vs Predito
    axes[1][0].scatter(v_true, a_true, alpha=0.4, s=15, label="Real",    color="#43A047")
    axes[1][0].scatter(v_pred, a_pred, alpha=0.4, s=15, label="Predito", color="#FB8C00", marker="^")
    axes[1][0].axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    axes[1][0].axvline(0.5, color='gray', linestyle='--', linewidth=0.8)
    axes[1][0].set_xlabel("Valence →")
    axes[1][0].set_ylabel("Arousal →")
    axes[1][0].set_xlim(0,1); axes[1][0].set_ylim(0,1)
    axes[1][0].set_title("Plano Russell Circumplex")
    axes[1][0].legend(fontsize=9)
    for (af, vf), nome in QUADRANTES.items():
        axes[1][0].text(0.75 if vf else 0.03, 0.75 if af else 0.03,
                        nome.split('(')[0].strip(), fontsize=8, color='gray', alpha=0.8)

    # 4. Confusion matrix quadrantes
    cm = confusion_matrix(q_true, q_pred, labels=nomes_q)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[q[:10] for q in nomes_q],
                yticklabels=[q[:10] for q in nomes_q],
                ax=axes[1][1], linewidths=0.5)
    axes[1][1].set_title(f"Confusão Quadrantes  Acc={acc:.3f}  F1={f1_mac:.3f}")
    axes[1][1].set_xlabel("Predito")
    axes[1][1].set_ylabel("Real")

    plt.suptitle(f"Avaliação chorus — {args.arch} | {args.modo.upper()}  ({len(df_res)} músicas)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_png = f"evaluate_chorus_{args.arch}_{args.modo}.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Gráfico salvo em: {out_png}")
