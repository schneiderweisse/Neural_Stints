"""Tabela de alvos comum a todos os modelos.

Cada linha é uma volta prevista: a janela (sequência, volta t) do dataset.npz e um horizonte h, com
os metadados e o valor real da volta t + h. Os modelos só preenchem as colunas de previsão
(pace_p10, pace_p50, pace_p90); results.py completa o resto do formato de docs/resultados.md.
"""
import numpy as np
import pandas as pd

from pipeline.sequences import RACE_LAPS, dataset_paths, load, prepare

META_COLUMNS = ["race_id", "race_label", "circuit", "year", "regulation", "driver", "team"]


def prepared_laps() -> pd.DataFrame:
    """Voltas na mesma ordem e com o mesmo filtro das matrizes do dataset.npz."""
    return prepare(pd.read_parquet(RACE_LAPS))


def target_frame(splits: tuple[str, ...] = ("val", "test"), dataset: str = "dataset"
                 ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Devolve (alvos, voltas, arrays, meta).

    alvos tem uma linha por (janela, h) das divisões pedidas, com a coluna `row` (índice da volta
    prevista em `voltas`) e `anchor` (índice da volta t, a última conhecida).
    """
    d, meta = load(*dataset_paths(dataset))
    laps = prepared_laps()
    if not np.array_equal(laps["lap"].to_numpy(), d["lap"]):
        raise RuntimeError("race_laps.parquet e dataset.npz estão dessincronizados; rode pipeline.sequences")

    H = meta["horizon"]
    keep = np.isin(d["win_split"], splits)
    seq, t, split = d["win_seq"][keep], d["win_t"][keep], d["win_split"][keep]
    lo, n = d["offsets"][seq], np.diff(d["offsets"])[seq]
    h = np.arange(1, H + 1)
    valid = (t[:, None] + h[None, :]) <= n[:, None]          # a corrida pode acabar antes de t + H
    w_idx, h_idx = np.nonzero(valid)

    frame = pd.DataFrame({
        "split": split[w_idx],
        "t": t[w_idx],
        "h": h[h_idx],
        "row": lo[w_idx] + t[w_idx] + h[h_idx] - 1,           # volta t + h (as voltas começam em 1)
        "anchor": lo[w_idx] + t[w_idx] - 1,                   # volta t
    })
    target = laps.iloc[frame["row"]].reset_index(drop=True)
    anchor = laps.iloc[frame["anchor"]].reset_index(drop=True)
    for c in META_COLUMNS:
        frame[c] = target[c].to_numpy()
    frame["lap"] = target["lap"].to_numpy()
    frame["compound"] = target["compound"].astype(str).to_numpy()
    frame["tyre_life"] = target["tyre_life"].to_numpy()
    frame["stint"] = target["stint"].to_numpy()
    frame["same_stint"] = (target["stint"].to_numpy() == anchor["stint"].to_numpy())
    frame["clean"] = target["clean"].to_numpy(bool)
    frame["ref_lap"] = target["ref_lap"].to_numpy()
    frame["pace_real"] = target["pace"].to_numpy()
    return frame, laps, d, meta
