import os
import glob
import pandas as pd

DEAM_DIR        = os.path.join("data", "deam")
AUDIO_DIR       = os.path.join(DEAM_DIR, "audio")
ANNOT_DIR       = os.path.join(DEAM_DIR, "annotations",
                               "annotations")
STATIC_RAW      = os.path.join(ANNOT_DIR, "song_level",
                               "static_annotations_averaged_songs_1_2000.csv")
STATIC_RAW_2    = os.path.join(ANNOT_DIR, "song_level",
                               "static_annotations_averaged_songs_2000_2058.csv")
DYNAMIC_AR_CSV  = os.path.join(ANNOT_DIR, "dynamic", "arousal.csv")
DYNAMIC_VA_CSV  = os.path.join(ANNOT_DIR, "dynamic", "valence.csv")

OUT_STATIC  = os.path.join("data", "static_annotations.csv")
OUT_DYNAMIC = os.path.join("data", "dynamic_annotations.csv")

SCALE = lambda x: (x - 1.0) / 8.0   # [1,9] -> [0,1]


def _ler_static(path):
    """Lê um CSV estático do DEAM e retorna DataFrame normalizado com 3 colunas."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    col_id = next(c for c in df.columns if "song" in c.lower() or "id" in c.lower())
    # arousal_mean ou mean_arousal — qualquer ordem de palavras
    col_ar = next(c for c in df.columns
                  if "arousal" in c.lower() and "mean" in c.lower()
                  and "std" not in c.lower() and "max" not in c.lower() and "min" not in c.lower())
    col_va = next(c for c in df.columns
                  if "valence" in c.lower() and "mean" in c.lower()
                  and "std" not in c.lower() and "max" not in c.lower() and "min" not in c.lower())
    return pd.DataFrame({
        "musicId":       df[col_id].astype(int),
        "Arousal(mean)": SCALE(df[col_ar].astype(float)).round(6),
        "Valence(mean)": SCALE(df[col_va].astype(float)).round(6),
    })


def processar_static():
    partes = [_ler_static(STATIC_RAW)]
    print(f"  {STATIC_RAW}: {len(partes[0])} músicas")

    if os.path.exists(STATIC_RAW_2):
        p2 = _ler_static(STATIC_RAW_2)
        partes.append(p2)
        print(f"  {STATIC_RAW_2}: {len(p2)} músicas")

    out = pd.concat(partes, ignore_index=True).drop_duplicates(subset="musicId")

    # Mantém só músicas que têm arquivo de áudio
    audio_ids = {int(os.path.basename(f).replace(".mp3", ""))
                 for f in glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))}
    before = len(out)
    out = out[out["musicId"].isin(audio_ids)].reset_index(drop=True)
    print(f"  Total: {before} -> {len(out)} com áudio disponível")

    out.to_csv(OUT_STATIC, index=False)
    print(f"  Salvo: {OUT_STATIC}\n")
    return out["musicId"].tolist()


def processar_dynamic(valid_ids):
    """
    Formato wide do DEAM: uma linha por música, colunas = sample_15000ms, sample_15500ms ...
    Valor em [-1, 1] (já normalizado pelos autores — NÃO aplicar SCALE).
    Convertemos para [0,1] com  (x + 1) / 2.
    """
    print(f"Lendo: {DYNAMIC_AR_CSV}")
    ar_df = pd.read_csv(DYNAMIC_AR_CSV)
    va_df = pd.read_csv(DYNAMIC_VA_CSV)
    ar_df.columns = ar_df.columns.str.strip()
    va_df.columns = va_df.columns.str.strip()

    # Coluna de id da música
    id_col = next(c for c in ar_df.columns if "song" in c.lower() or "id" in c.lower())

    # Colunas de timestamp comuns aos dois arquivos
    ar_time_cols = set(c for c in ar_df.columns if c.startswith("sample_"))
    va_time_cols = set(c for c in va_df.columns if c.startswith("sample_"))
    time_cols = sorted(ar_time_cols & va_time_cols,
                       key=lambda c: int(c.replace("sample_", "").replace("ms", "")))
    times_sec = [int(c.replace("sample_", "").replace("ms", "")) / 1000.0 for c in time_cols]

    valid_set = set(valid_ids)
    rows = []

    for _, ar_row in ar_df.iterrows():
        mid = int(ar_row[id_col])
        if mid not in valid_set:
            continue

        va_row = va_df[va_df[id_col] == mid]
        if va_row.empty:
            continue
        va_row = va_row.iloc[0]

        for col, t in zip(time_cols, times_sec):
            a = ar_row[col]
            v = va_row[col]
            if pd.isna(a) or pd.isna(v):
                continue
            rows.append({
                "musicId":       mid,
                "frameTime":     round(t, 3),
                "Arousal(mean)": round((float(a) + 1.0) / 2.0, 6),
                "Valence(mean)": round((float(v) + 1.0) / 2.0, 6),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DYNAMIC, index=False)
    print(f"  {len(out)} frames de {out['musicId'].nunique()} músicas salvos em: {OUT_DYNAMIC}\n")


if __name__ == "__main__":
    for path in [AUDIO_DIR, ANNOT_DIR, STATIC_RAW]:
        if not os.path.exists(path):
            print(f"\nERRO: caminho não encontrado:\n  {os.path.abspath(path)}")
            print("\nEstrutura esperada:")
            print("  data/deam/audio/              ← arquivos .mp3 do DEAM")
            print("  data/deam/annotations/        ← pasta annotations do DEAM")
            raise SystemExit(1)

    os.makedirs("data", exist_ok=True)

    valid_ids = processar_static()

    if os.path.exists(DYNAMIC_AR_CSV) and os.path.exists(DYNAMIC_VA_CSV):
        processar_dynamic(valid_ids)
    else:
        print("Anotações dinâmicas não encontradas — apenas estático gerado.")
        print(f"  Esperado em: {DYNAMIC_AR_CSV}")

    print("Pré-processamento DEAM concluído!")
    print(f"  Static : {OUT_STATIC}")
    if os.path.exists(OUT_DYNAMIC):
        print(f"  Dynamic: {OUT_DYNAMIC}")
