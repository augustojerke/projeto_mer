import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import glob
import os
import time
from sklearn.model_selection import KFold
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score)

from dataset import DEAMDinamicoDataset, SUBSAMPLE_STEPS, _quadrante
from model import build_model
from benchmark import BenchmarkMonitor, ExperimentoMetrics, EpocaMetrics, salvar_resultados

parser = argparse.ArgumentParser(description="Treino MER — DEAM")
parser.add_argument("--modo",   type=str, default="mel",
                    choices=["stft", "mel", "mfcc", "tri"],
                    help="Espectrograma de entrada")
parser.add_argument("--arch",   type=str, default="resnet18",
                    choices=["resnet18"],
                    help="Arquitetura do modelo")
parser.add_argument("--epochs", type=int, default=100,
                    help="Número máximo de épocas por fold")
args = parser.parse_args()

BATCH    = {"mfcc": 1024, "stft": 256, "mel": 256, "tri": 64}.get(args.modo, 256)
LR       = 3e-4 if args.arch == "resnet18" else 1e-3
PATIENCE = 15
FOLDS    = 5
WORKERS  = 0 
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP  = torch.cuda.is_available() 
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

QUADRANTES = {0: "Exaltado (A+V+)", 1: "Irritado (A+V-)",
              2: "Calmo (A-V+)",    3: "Triste (A-V-)"}

def para_quadrante(arousal: torch.Tensor, valence: torch.Tensor) -> torch.Tensor:
    q = torch.full((len(arousal),), 3, dtype=torch.long)
    q[(arousal >= 0.5) & (valence >= 0.5)] = 0
    q[(arousal >= 0.5) & (valence <  0.5)] = 1
    q[(arousal <  0.5) & (valence >= 0.5)] = 2
    return q

AUDIO_DIR = "data/deam/audio"
CSV_PATH  = "data/dynamic_annotations.csv"

audio_map = {}
for fp in glob.glob(os.path.join(AUDIO_DIR, "*.mp3")):
    mid = int(os.path.basename(fp).replace(".mp3", ""))
    audio_map[mid] = fp

df        = pd.read_csv(CSV_PATH)
music_ids = df["musicId"].unique()

rng      = np.random.default_rng(3)
ids_shuf = rng.permutation(music_ids)
n_test   = round(len(ids_shuf) * 0.10)
ids_test = ids_shuf[:n_test]
ids_cv   = ids_shuf[n_test:]

df_test = df[df["musicId"].isin(ids_test)].reset_index(drop=True)
df_cv   = df[df["musicId"].isin(ids_cv)].reset_index(drop=True)

print(f"\nModo: {args.modo} | Épocas: {args.epochs}")
print(f"Músicas — CV: {len(ids_cv)} | Teste: {len(ids_test)}")
print(f"Amostras— CV: {len(df_cv)} | Teste: {len(df_test)}")
print(f"Usando: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

_smooth_l1    = nn.SmoothL1Loss()
MIXUP_ALPHA   = 0.4
WARMUP_EPOCHS = 5

def mixup_assimetrico(feats, labels):
    """Mixup preferencial entre minoritários: par (minoritário, qualquer) com prob 0.7."""
    lam     = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    a, v    = labels[:, 0].cpu().numpy(), labels[:, 1].cpu().numpy()
    quads   = np.array([_quadrante(float(ai), float(vi)) for ai, vi in zip(a, v)])
    is_min  = torch.tensor(quads != 0, dtype=torch.bool)   # tudo exceto Exaltado é minoritário

    # Para amostras minoritárias: par com outra minoritária (se disponível)
    idx = torch.randperm(feats.size(0), device=feats.device)
    min_idx = torch.where(is_min)[0]
    if len(min_idx) >= 2 and np.random.random() < 0.7:
        min_perm = min_idx[torch.randperm(len(min_idx))]
        for i, j in zip(min_idx, min_perm):
            idx[i] = j

    return lam * feats + (1 - lam) * feats[idx], lam * labels + (1 - lam) * labels[idx]

def subsamplear_frames(df):
    """Step de subsample por quadrante: minoritários usam step menor (mais frames)."""
    grupos = []
    for _, g in df.sort_values(["musicId", "frameTime"]).groupby("musicId"):
        # Determina quadrante dominante da música pelo valor médio
        a_med = g['Arousal(mean)'].mean()
        v_med = g['Valence(mean)'].mean()
        step  = SUBSAMPLE_STEPS[_quadrante(a_med, v_med)]
        grupos.append(g.iloc[::step])
    return pd.concat(grupos).reset_index(drop=True)

def make_balanced_sampler(df):
    """WeightedRandomSampler com pesos inversamente proporcionais ao quadrante emocional."""
    a = df['Arousal(mean)'].values
    v = df['Valence(mean)'].values
    quads = np.where((a >= 0.5) & (v >= 0.5), 0,
            np.where((a >= 0.5) & (v <  0.5), 1,
            np.where((a <  0.5) & (v >= 0.5), 2, 3))).astype(int)
    counts  = np.bincount(quads, minlength=4).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)
    weights = torch.tensor(1.0 / counts[quads], dtype=torch.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def _pearson(p, l):
    px, py = p - p.mean(), l - l.mean()
    return (px * py).sum() / (px.norm() * py.norm() + 1e-8)

def _ccc(p, l):
    cov = ((p - p.mean()) * (l - l.mean())).mean()
    return 2 * cov / (p.var() + l.var() + (p.mean() - l.mean()) ** 2 + 1e-8)

def criterion(preds, labels):
    """Loss misto: 50% SmoothL1 + 25% (1-CCC_Arousal) + 25% (1-CCC_Valence)."""
    l1    = _smooth_l1(preds, labels)
    ccc_a = 1.0 - _ccc(preds[:, 0], labels[:, 0])
    ccc_v = 1.0 - _ccc(preds[:, 1], labels[:, 1])
    return 0.5 * l1 + 0.25 * ccc_a + 0.25 * ccc_v

def _r2(p, l):
    ss_res = ((l - p) ** 2).sum()
    ss_tot = ((l - l.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()

def calcular_metricas(preds, labels):
    return {
        "loss":      criterion(preds, labels).item(),
        "rmse_a":    torch.sqrt(nn.MSELoss()(preds[:, 0], labels[:, 0])).item(),
        "rmse_v":    torch.sqrt(nn.MSELoss()(preds[:, 1], labels[:, 1])).item(),
        "mae_a":     nn.L1Loss()(preds[:, 0], labels[:, 0]).item(),
        "mae_v":     nn.L1Loss()(preds[:, 1], labels[:, 1]).item(),
        "pearson_a": _pearson(preds[:, 0], labels[:, 0]).item(),
        "pearson_v": _pearson(preds[:, 1], labels[:, 1]).item(),
        "ccc_a":     _ccc(preds[:, 0], labels[:, 0]).item(),
        "ccc_v":     _ccc(preds[:, 1], labels[:, 1]).item(),
        "r2_a":      _r2(preds[:, 0], labels[:, 0]),
        "r2_v":      _r2(preds[:, 1], labels[:, 1]),
    }

# ─────────────────────────────────────────
# Treino / avaliação
# ─────────────────────────────────────────
def treinar_epoca(modelo, loader, optimizer, fold_tag, epoca, scaler):
    modelo.train()
    total = 0.0
    n     = len(loader)
    for i, (feats, labels) in enumerate(loader):
        feats  = feats.to(DEVICE,  non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        feats, labels = mixup_assimetrico(feats, labels)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(modelo(feats), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(modelo(feats), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step()
        total += loss.item()
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"\r  [{fold_tag}] E{epoca:03d} {i+1}/{n} batches  loss:{total/(i+1):.4f}", end="", flush=True)
    print()
    return total / n

def avaliar(modelo, loader):
    modelo.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for feats, labels in loader:
            if USE_AMP:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    all_p.append(modelo(feats.to(DEVICE, non_blocking=True)).cpu())
            else:
                all_p.append(modelo(feats.to(DEVICE, non_blocking=True)).cpu())
            all_l.append(labels)
    return torch.cat(all_p), torch.cat(all_l)

# ─────────────────────────────────────────
# Treino de um fold
# ─────────────────────────────────────────
def treinar_fold(df_train, df_val, fold_tag):
    modelo = build_model(args.arch, args.modo).to(DEVICE)

    if hasattr(modelo, 'backbone'):
        for p in modelo.backbone[4].parameters():
            p.requires_grad = True
        params_main   = [p for n, p in modelo.named_parameters()
                         if p.requires_grad and 'backbone.4' not in n]
        params_layer1 = list(modelo.backbone[4].parameters())
        optimizer = torch.optim.AdamW([
            {'params': params_main,   'lr': LR},
            {'params': params_layer1, 'lr': LR * 0.05},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in modelo.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
        )

    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - WARMUP_EPOCHS), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
    )

    preload      = True
    df_train_sub = subsamplear_frames(df_train)
    sampler      = make_balanced_sampler(df_train_sub)
    ds_train     = DEAMDinamicoDataset(df_train_sub, audio_map, args.modo, augment=True,  preload=preload)
    ds_val       = DEAMDinamicoDataset(df_val,       audio_map, args.modo, augment=False, preload=preload)
    n_treino     = f"{df_train_sub['musicId'].nunique()} músicas / {len(df_train_sub)} frames [subsampled]"

    dl_train = DataLoader(
        ds_train, batch_size=BATCH, sampler=sampler,
        num_workers=WORKERS,
    )
    dl_val = DataLoader(
        ds_val, batch_size=BATCH, shuffle=False,
        num_workers=WORKERS,
    )

    save_path       = f"{SAVE_DIR}/{args.arch}_{args.modo}_dinamico_{fold_tag}_best.pt"
    melhor_loss     = float("inf")
    paciencia_atual = 0
    epocas_log      = []

    print(f"\n{'─'*60}")
    print(f"  {fold_tag.upper()}  ({n_treino})")
    print(f"{'─'*60}")

    for epoca in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = treinar_epoca(modelo, dl_train, optimizer, fold_tag, epoca, scaler)
        p_val, l_val = avaliar(modelo, dl_val)
        m          = calcular_metricas(p_val, l_val)
        scheduler.step()
        tempo      = time.time() - t0

        epocas_log.append(EpocaMetrics(
            epoca=epoca,
            train_loss=round(train_loss, 6),
            val_loss=round(m["loss"], 6),
            rmse_arousal=round(m["rmse_a"], 6),
            rmse_valence=round(m["rmse_v"], 6),
            mae_arousal=round(m["mae_a"], 6),
            mae_valence=round(m["mae_v"], 6),
            pearson_arousal=round(m["pearson_a"], 6),
            pearson_valence=round(m["pearson_v"], 6),
            ccc_arousal=round(m["ccc_a"], 6),
            ccc_valence=round(m["ccc_v"], 6),
            tempo_seg=round(tempo, 2),
            gpu_mem_mb=0.0,
        ))

        print(f"  E{epoca:03d} | train:{train_loss:.4f} val:{m['loss']:.4f} | "
              f"RMSE A:{m['rmse_a']:.3f} V:{m['rmse_v']:.3f} | "
              f"Pearson A:{m['pearson_a']:.3f} V:{m['pearson_v']:.3f} | "
              f"CCC A:{m['ccc_a']:.3f} V:{m['ccc_v']:.3f} | {tempo:.1f}s")

        if m["loss"] < melhor_loss:
            melhor_loss     = m["loss"]
            paciencia_atual = 0
            torch.save(modelo.state_dict(), save_path)
            print(f"    → melhor salvo")
        else:
            paciencia_atual += 1
            if paciencia_atual >= PATIENCE:
                print(f"    → early stopping (época {epoca})")
                break

    modelo.load_state_dict(torch.load(save_path, weights_only=True))
    p_val, l_val = avaliar(modelo, dl_val)
    metricas = calcular_metricas(p_val, l_val)

    del dl_train, dl_val, ds_train, ds_val
    torch.cuda.empty_cache()

    return modelo, metricas, epocas_log, save_path

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    total_params = sum(p.numel() for p in build_model(args.arch, args.modo).parameters())
    print(f"Arquitetura: {args.arch} | Parâmetros: {total_params:,}")

    experimento  = f"{args.arch}_{args.modo}_dinamico"
    monitor      = BenchmarkMonitor()
    monitor.iniciar()
    tempo_inicio = time.time()

    # ── 5-fold CV por música ────────────────────────────────────────────
    kf       = KFold(n_splits=FOLDS, shuffle=True, random_state=3)
    ids_cv_a = np.array(ids_cv)

    fold_results    = []
    best_ccc_media  = -float("inf")
    best_model_path = None
    best_epocas_log = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(ids_cv_a)):
        ids_tr = ids_cv_a[tr_idx]
        ids_v  = ids_cv_a[val_idx]
        df_tr  = df_cv[df_cv["musicId"].isin(ids_tr)]
        df_v   = df_cv[df_cv["musicId"].isin(ids_v)]

        fold_tag = f"fold{fold + 1}"
        _, val_m, epocas_log, save_path = treinar_fold(df_tr, df_v, fold_tag)
        fold_results.append(val_m)

        ccc_media = (val_m["ccc_a"] + val_m["ccc_v"]) / 2
        if ccc_media > best_ccc_media:
            best_ccc_media  = ccc_media
            best_model_path = save_path
            best_epocas_log = epocas_log

        print(f"\n  [{fold_tag}] val → "
              f"RMSE A:{val_m['rmse_a']:.4f} V:{val_m['rmse_v']:.4f} | "
              f"Pearson A:{val_m['pearson_a']:.3f} V:{val_m['pearson_v']:.3f} | "
              f"CCC A:{val_m['ccc_a']:.3f} V:{val_m['ccc_v']:.3f}")

    # ── Resumo CV ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CV {FOLDS}-fold — médias ± std")
    cv_summary = {}
    for key in ("rmse_a", "rmse_v", "mae_a", "mae_v", "pearson_a", "pearson_v", "ccc_a", "ccc_v"):
        vals = [r[key] for r in fold_results]
        media, std = np.mean(vals), np.std(vals)
        cv_summary[key] = {"media": round(float(media), 4), "std": round(float(std), 4)}
        print(f"  {key:12s}: {media:.4f} ± {std:.4f}")

    # ── Avaliação no teste com melhor fold ──────────────────────────────
    best_model = build_model(args.arch, args.modo).to(DEVICE)
    best_model.load_state_dict(torch.load(best_model_path, weights_only=True))

    ds_test = DEAMDinamicoDataset(df_test, audio_map, args.modo, augment=False, preload=True)
    dl_test = DataLoader(
        ds_test, batch_size=BATCH, shuffle=False,
        num_workers=WORKERS,
    )
    p_test, l_test = avaliar(best_model, dl_test)
    test_m         = calcular_metricas(p_test, l_test)

    # ── Classificação emocional por quadrante ──────────────────────────
    q_pred = para_quadrante(p_test[:, 0], p_test[:, 1]).numpy()
    q_true = para_quadrante(l_test[:, 0], l_test[:, 1]).numpy()
    nomes  = [QUADRANTES[i] for i in sorted(QUADRANTES)]

    cm     = confusion_matrix(q_true, q_pred, labels=[0, 1, 2, 3])
    f1_mac = f1_score(q_true, q_pred, average="macro", zero_division=0)
    acc    = accuracy_score(q_true, q_pred)
    report = classification_report(q_true, q_pred,
                                   labels=[0, 1, 2, 3], target_names=nomes,
                                   zero_division=0, output_dict=True)

    # ── Impressão final ─────────────────────────────────────────────────
    monitor.parar()
    tempo_total = time.time() - tempo_inicio

    print(f"\n{'='*60}")
    print(f"TESTE FINAL — {experimento}  ({len(p_test)} músicas)")
    print(f"  RMSE    A:{test_m['rmse_a']:.4f}  V:{test_m['rmse_v']:.4f}")
    print(f"  MAE     A:{test_m['mae_a']:.4f}  V:{test_m['mae_v']:.4f}")
    print(f"  Pearson A:{test_m['pearson_a']:.4f}  V:{test_m['pearson_v']:.4f}")
    print(f"  CCC     A:{test_m['ccc_a']:.4f}  V:{test_m['ccc_v']:.4f}")
    print(f"  R²      A:{test_m['r2_a']:.4f}  V:{test_m['r2_v']:.4f}")

    print(f"\n  Classificação emocional (quadrantes):")
    print(f"  Acurácia: {acc:.4f}  |  F1-macro: {f1_mac:.4f}")
    print(f"\n  Matriz de Confusão (linhas=real, colunas=pred):")
    header = "               " + "  ".join(f"{n[:9]:>9}" for n in nomes)
    print(f"  {header}")
    for i, row in enumerate(cm):
        print(f"  {nomes[i][:14]:14s} " + "  ".join(f"{v:>9d}" for v in row))
    print(f"\n  Relatório por quadrante:")
    for nome in nomes:
        r = report[nome]
        print(f"  {nome[:20]:20s}  prec:{r['precision']:.3f}  "
              f"rec:{r['recall']:.3f}  f1:{r['f1-score']:.3f}  n:{int(r['support'])}")

    print(f"\n  Tempo total: {tempo_total:.1f}s | "
          f"CPU: {monitor.cpu_medio():.1f}% | GPU pico: {monitor.gpu_pico():.0f}MB")

    # ── Salvar ──────────────────────────────────────────────────────────
    exp_metrics = ExperimentoMetrics(
        nome=experimento, modo=args.modo,
        device=str(DEVICE), total_params=total_params,
        epocas=best_epocas_log,
        test_rmse_arousal=round(test_m["rmse_a"], 6),
        test_rmse_valence=round(test_m["rmse_v"], 6),
        test_mae_arousal=round(test_m["mae_a"], 6),
        test_mae_valence=round(test_m["mae_v"], 6),
        test_pearson_arousal=round(test_m["pearson_a"], 6),
        test_pearson_valence=round(test_m["pearson_v"], 6),
        test_ccc_arousal=round(test_m["ccc_a"], 6),
        test_ccc_valence=round(test_m["ccc_v"], 6),
        test_r2_arousal=round(test_m["r2_a"], 6),
        test_r2_valence=round(test_m["r2_v"], 6),
        test_rmse_songs_arousal=round(test_m["rmse_a"], 6),
        test_rmse_songs_valence=round(test_m["rmse_v"], 6),
        quadrant_accuracy=round(float(acc), 6),
        quadrant_f1_macro=round(float(f1_mac), 6),
        quadrant_report=report,
        cv_metricas=cv_summary,
        tempo_total_seg=round(tempo_total, 2),
        cpu_percent_medio=round(monitor.cpu_medio(), 1),
        gpu_mem_pico_mb=round(monitor.gpu_pico(), 1),
    )

    resultado_path = f"resultados_{experimento}.json"
    salvar_resultados({experimento: exp_metrics}, path=resultado_path)
    print(f"Resultados salvos em {resultado_path}")
