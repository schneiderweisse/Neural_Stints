"""Transforma a tabela de voltas de corrida em sequências e janelas para a rede.

Entrada: data/processed/race_laps.parquet (race_laps.py). Saídas:

- data/processed/dataset.npz: matrizes por volta (variáveis observadas, conhecidas no futuro, alvo e
  máscara), concatenadas por sequência (piloto-corrida), com os offsets de cada sequência, as
  variáveis estáticas de cada sequência e o índice das janelas;
- data/processed/dataset_meta.json: nomes das variáveis, vocabulários, estatísticas de
  normalização (ajustadas só no treino) e a divisão das corridas em treino/validação/teste.

Uma janela é (sequência, volta t): o encoder lê as voltas 1..t e o decoder recebe as variáveis
conhecidas das voltas t+1..t+H e prevê o ritmo relativo nelas. O decoder vê o futuro real do pneu,
inclusive paradas nos boxes e trocas de composto, que é o que ensina a curva do próximo stint.

Uso, a partir da pasta Neural_Stints:  .venv/Scripts/python -m pipeline.sequences
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RACE_LAPS = ROOT / "data" / "processed" / "race_laps.parquet"
OUT_NPZ = ROOT / "data" / "processed" / "dataset.npz"
OUT_META = ROOT / "data" / "processed" / "dataset_meta.json"

HORIZON = 10          # H: voltas previstas por janela
MIN_CONTEXT = 2       # a primeira previsão é feita ao fim da volta 2
N_VAL, N_TEST = 4, 5  # últimas corridas (em ordem cronológica) para validação e teste

# Observadas: só existem para as voltas já disputadas (entram no encoder)
OBSERVED_NUM = ["pace_in", "s1_pace", "s2_pace", "s3_pace", "speed_i1_rel", "speed_i2_rel",
                "speed_fl_rel", "speed_st_rel", "gap_ahead", "gap_leader", "position",
                "track_temp", "air_temp", "humidity"]
OBSERVED_BOOL = ["clean", "neutralised", "yellow", "traffic", "rain"]
# Conhecidas no futuro: entram no encoder e no decoder (no decoder, descrevem o cenário de pneu)
KNOWN_NUM = ["race_progress", "laps_remaining", "tyre_life", "stint_lap", "stint"]
KNOWN_BOOL = ["fresh_tyre", "pit_in", "pit_out", "is_soft", "is_medium", "is_hard"]
# Estáticas: uma vez por sequência
STATIC_CAT = ["circuit", "driver", "team_year", "regulation"]
STATIC_NUM = ["laps_total"]
STATIC_BOOL = ["ref_from_quali"]
GAP_CLIP = {"gap_ahead": (0, 30), "gap_leader": (0, 120)}
Z_CLIP = 10.0         # voltas de SC e de pit ficam a ~20 desvios-padrão das voltas limpas


def prepare(race: pd.DataFrame) -> pd.DataFrame:
    race = race[~race["race_wet"] & race["ref_lap"].notna()].copy()
    race = race.sort_values(["year", "round", "driver", "lap"]).reset_index(drop=True)
    for c, (lo, hi) in GAP_CLIP.items():
        race[c] = race[c].clip(lo, hi)
    race.loc[race["position"] == 1, "gap_ahead"] = GAP_CLIP["gap_ahead"][1]   # líder: pista livre
    for comp in ("soft", "medium", "hard"):
        race[f"is_{comp}"] = race["compound"] == comp.upper()
    race["team_year"] = race["team"].astype(str) + " " + race["year"].astype(str)
    race["seq_id"] = race["race_id"] + "_" + race["driver"].astype(str)
    return race


def split_races(race: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Divisão cronológica das corridas e o rótulo de cada uma (<ano> <circuito>)."""
    order = race[["race_id", "race_label", "year", "round"]].drop_duplicates().sort_values(["year", "round"])
    ids = order["race_id"].tolist()
    splits = {"train": ids[:-(N_VAL + N_TEST)], "val": ids[-(N_VAL + N_TEST):-N_TEST], "test": ids[-N_TEST:]}
    return splits, dict(zip(order["race_id"], order["race_label"]))


def fit_stats(train: pd.DataFrame, cols: list[str]) -> dict[str, list[float]]:
    """Média e desvio-padrão de cada variável numérica, só com as voltas limpas do treino."""
    ref = train[train["clean"]]
    return {c: [float(ref[c].mean()), float(ref[c].std() or 1.0)] for c in cols}


def vocab(train: pd.DataFrame, col: str) -> list[str]:
    """Vocabulário do treino; o índice 0 é reservado para valores não vistos (<unk>)."""
    return ["<unk>"] + sorted(train[col].astype(str).unique().tolist())


def encode(df: pd.DataFrame, num: list[str], boolean: list[str], stats: dict) -> np.ndarray:
    parts = [((df[c].astype(float) - stats[c][0]) / stats[c][1]).clip(-Z_CLIP, Z_CLIP).fillna(0.0).to_numpy()
             for c in num]
    parts += [df[c].fillna(False).astype(float).to_numpy() for c in boolean]
    return np.stack(parts, axis=1).astype(np.float32)


def build() -> tuple[dict, dict]:
    race = prepare(pd.read_parquet(RACE_LAPS))
    splits, labels = split_races(race)
    train = race[race["race_id"].isin(splits["train"])]
    stats = fit_stats(train, OBSERVED_NUM + KNOWN_NUM + STATIC_NUM)
    vocabs = {c: vocab(train, c) for c in STATIC_CAT}

    x_obs = encode(race, OBSERVED_NUM, OBSERVED_BOOL, stats)
    x_known = encode(race, KNOWN_NUM, KNOWN_BOOL, stats)
    y = race["pace"].astype(float).fillna(0.0).to_numpy(np.float32)
    y_mask = race["clean"].to_numpy(bool)

    seqs = race.groupby("seq_id", sort=False)
    first = seqs.head(1).reset_index(drop=True)
    offsets = np.concatenate([[0], np.cumsum(seqs.size().to_numpy())]).astype(np.int64)
    static_cat = np.stack([first[c].astype(str).map({v: i for i, v in enumerate(vocabs[c])}).fillna(0)
                           .astype(int).to_numpy() for c in STATIC_CAT], axis=1).astype(np.int64)
    static_x = encode(first, STATIC_NUM, STATIC_BOOL, stats)
    split_of_race = {r: s for s, ids in splits.items() for r in ids}
    seq_split = first["race_id"].map(split_of_race).to_numpy()

    # Janelas: (sequência, t) com ao menos uma volta limpa entre t+1 e t+H
    win_seq, win_t, win_split = [], [], []
    for s in range(len(first)):
        lo, hi = offsets[s], offsets[s + 1]
        n = hi - lo
        for t in range(MIN_CONTEXT, n):
            if y_mask[lo + t: lo + min(t + HORIZON, n)].any():
                win_seq.append(s)
                win_t.append(t)
                win_split.append(seq_split[s])
    windows = {"win_seq": np.array(win_seq, np.int64), "win_t": np.array(win_t, np.int64),
               "win_split": np.array(win_split)}

    arrays = {"x_obs": x_obs, "x_known": x_known, "y": y, "y_mask": y_mask, "offsets": offsets,
              "static_cat": static_cat, "static_x": static_x, "seq_split": seq_split,
              "seq_id": first["seq_id"].to_numpy(), "lap": race["lap"].to_numpy(np.int64),
              "ref_lap": race["ref_lap"].to_numpy(np.float32), **windows}
    meta = {
        "horizon": HORIZON, "min_context": MIN_CONTEXT, "splits": splits, "race_labels": labels,
        "observed": OBSERVED_NUM + OBSERVED_BOOL, "known": KNOWN_NUM + KNOWN_BOOL,
        "static_cat": STATIC_CAT, "static_x": STATIC_NUM + STATIC_BOOL,
        "target": "pace = lap_time / ref_lap - 1", "stats": stats, "vocabs": vocabs,
    }
    return arrays, meta


def load(npz: Path = OUT_NPZ, meta: Path = OUT_META) -> tuple[dict, dict]:
    data = np.load(npz, allow_pickle=True)
    return {k: data[k] for k in data.files}, json.loads(meta.read_text(encoding="utf-8"))


def get_window(d: dict, w: int, horizon: int = HORIZON) -> dict[str, np.ndarray]:
    """Arrays de uma janela: encoder (voltas 1..t), decoder (t+1..t+H, com padding) e alvo."""
    s, t = int(d["win_seq"][w]), int(d["win_t"][w])
    lo, hi = d["offsets"][s], d["offsets"][s + 1]
    enc = np.concatenate([d["x_obs"][lo:lo + t], d["x_known"][lo:lo + t]], axis=1)
    fut = slice(lo + t, min(lo + t + horizon, hi))
    h = fut.stop - fut.start
    dec = np.zeros((horizon, d["x_known"].shape[1]), np.float32)
    y = np.zeros(horizon, np.float32)
    mask = np.zeros(horizon, bool)
    dec[:h], y[:h], mask[:h] = d["x_known"][fut], d["y"][fut], d["y_mask"][fut]
    return {"enc": enc, "dec": dec, "y": y, "y_mask": mask,
            "static_cat": d["static_cat"][s], "static_x": d["static_x"][s]}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    arrays, meta = build()
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **arrays)
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(arrays['seq_id']):,} sequências, {len(arrays['y']):,} voltas, "
          f"{len(arrays['win_seq']):,} janelas -> {OUT_NPZ.relative_to(ROOT)}")
    for name in ("train", "val", "test"):
        ids = meta["splits"][name]
        n_win = int((arrays["win_split"] == name).sum())
        n_seq = int((arrays["seq_split"] == name).sum())
        print(f"  {name:5}: {len(ids):3} corridas, {n_seq:5} sequências, {n_win:6} janelas")
        if name != "train":
            print("         " + ", ".join(meta["race_labels"][r] for r in ids))
    print(f"  entradas: encoder {len(meta['observed']) + len(meta['known'])}, decoder {len(meta['known'])}, "
          f"estáticas {len(meta['static_cat'])} categóricas + {len(meta['static_x'])} numéricas")


if __name__ == "__main__":
    main()
