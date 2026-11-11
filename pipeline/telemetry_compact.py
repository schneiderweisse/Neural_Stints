"""Converte a telemetria do TracingInsights (JSON por volta) num formato compacto, reamostrado por distância.

Entrada: telemetry/tracinginsights/<ano>/<GP>/<Race|Sprint>/<piloto>/<volta>_tel.json (fetch_telemetry.py),
~19 GB de JSON em ~135 mil voltas, com ~7,5 amostras por segundo e um número de amostras diferente em
cada volta. Saída em telemetry/compact/:

    index.parquet   uma linha por volta: `row` (posição nos arrays), ano, rodada, `race_id` (o mesmo de
                    data/processed/race_laps.parquet), GP, circuito, sessão (R ou S), piloto, volta e
                    medidas de qualidade da volta original (amostras, duração, distância, maior intervalo
                    sem amostra, recuos de distância, parte sem posição ou sem carro da frente)
    tel_f16.npy     (voltas, 512, 10) float16: velocidade, rpm, marcha, acelerador, freio, DRS,
                    acc_x, acc_y, acc_z e distância ao carro da frente
    tel_f32.npy     (voltas, 512, 4) float32: tempo desde o início da volta, x, y e z
    meta.json       canais, unidades, grade, commit do TracingInsights de cada ano e data da conversão
    parts/          um .npz por sessão (cache: rodar de novo só converte as sessões que mudaram)

Reamostragem: cada volta vira 512 pontos igualmente espaçados na distância da própria volta, da primeira
à última amostra (grade de `rel_distance`, ~10 m por ponto num circuito de 5 km, perto da resolução
original). Assim o ponto k é o mesmo trecho da pista em todas as voltas de um circuito. Os canais
contínuos são interpolados linearmente; marcha, freio e DRS pegam a amostra mais próxima. O tempo e a
posição ficam em float32, porque em float16 o tempo de uma volta de 90 s teria passo de 0,06 s e as
coordenadas, de até 16 m.

Tratamentos:
- "None" no JSON vira NaN; um canal sem nenhuma amostra válida na volta fica todo em NaN;
- a distância às vezes recua alguns milímetros com o carro parado (largada, pit stop): usa-se o
  máximo acumulado, e amostras sem avanço de distância ficam fora da interpolação (vale a primeira);
- o DRS não existe a partir de 2026 (sempre 0 no JSON);
- x, y e z estão em décimos de metro, como no FastF1, embora o DATA_REFERENCE.md diga metros: o z do
  México fica em ~22.270, e a pista está a ~2.240 m de altitude.

Uso, a partir da pasta do repositório:
    .venv/Scripts/python -m pipeline.telemetry_compact               # tudo, 2022 a 2026
    .venv/Scripts/python -m pipeline.telemetry_compact --years 2026  # só alguns anos (o índice junta
                                                                     # todas as partes já convertidas)
    .venv/Scripts/python -m pipeline.telemetry_compact --rebuild     # ignora o cache de parts/

Leitura no treino: np.load("telemetry/compact/tel_f16.npy", mmap_mode="r") e index.parquet.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import orjson
    loads = orjson.loads
except ImportError:                      # orjson é opcional: só deixa a leitura ~3× mais rápida
    loads = json.loads

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "telemetry" / "tracinginsights"
OUT = ROOT / "telemetry" / "compact"
RACE_LAPS = ROOT / "data" / "processed" / "race_laps.parquet"
YEARS = [2022, 2023, 2024, 2025, 2026]
SESSIONS = {"Race": "R", "Sprint": "S"}
GRID = 512

# (canal, campo do JSON, unidade, interpolação)
F16 = [("speed", "speed", "km/h", "linear"), ("rpm", "rpm", "rpm", "linear"), ("gear", "gear", "marcha", "nearest"),
       ("throttle", "throttle", "%", "linear"), ("brake", "brake", "0/1", "nearest"), ("drs", "drs", "código", "nearest"),
       ("acc_x", "acc_x", "m/s²", "linear"), ("acc_y", "acc_y", "m/s²", "linear"), ("acc_z", "acc_z", "m/s²", "linear"),
       ("dist_ahead", "DistanceToDriverAhead", "m", "linear")]
F32 = [("time", "time", "s desde o início da volta", "linear"), ("x", "x", "1/10 m", "linear"),
       ("y", "y", "1/10 m", "linear"), ("z", "z", "1/10 m", "linear")]


def as_float(values: list) -> np.ndarray:
    """Lista do JSON -> float64, com "None" (texto) e None como NaN."""
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return np.array([v if isinstance(v, (int, float)) and not isinstance(v, bool) else
                         (float(v) if isinstance(v, bool) else np.nan) for v in values], dtype=np.float64)


def resample(tel: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Uma volta -> (512 × 10 float16, 512 × 4 float32, medidas de qualidade)."""
    t = as_float(tel["time"])
    d = as_float(tel["distance"])
    n = len(t)
    back = int((np.diff(d) < 0).sum()) if n > 1 else 0
    d = np.maximum.accumulate(np.nan_to_num(d, nan=-np.inf))
    keep = np.concatenate([[True], np.diff(d) > 0]) if n > 1 else np.ones(n, bool)
    dk = d[keep]
    grid = np.linspace(dk[0], dk[-1], GRID) if len(dk) > 1 else np.full(GRID, dk[0] if len(dk) else np.nan)
    nearest = np.clip(np.searchsorted(dk, grid), 0, max(len(dk) - 1, 0))

    def channel(field: str, how: str) -> np.ndarray:
        v = as_float(tel.get(field, []))
        if len(v) != n:
            return np.full(GRID, np.nan)
        v = v[keep]
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            return np.full(GRID, np.nan)
        if how == "nearest":
            return v[nearest]
        if ok.sum() == 1:
            return np.full(GRID, v[ok][0])
        return np.interp(grid, dk[ok], v[ok])

    # float16 vai até 65.504: com o carro parado por muito tempo, a distância ao da frente passa disso
    f16_max = float(np.finfo(np.float16).max)
    a16 = np.clip(np.stack([channel(f, how) for _, f, _, how in F16], axis=1), -f16_max, f16_max).astype(np.float16)
    a32 = np.stack([channel(f, how) for _, f, _, how in F32], axis=1).astype(np.float32)
    x = as_float(tel.get("x", []))
    ahead = as_float(tel.get("DistanceToDriverAhead", []))
    quality = {
        "n_samples": n,
        "duration_s": float(t[-1]) if n else np.nan,
        "distance_m": float(dk[-1] - dk[0]) if len(dk) > 1 else np.nan,
        "start_distance_m": float(as_float(tel["distance"])[0]) if n else np.nan,
        "max_gap_s": float(np.diff(t).max()) if n > 1 else np.nan,
        "n_backward": back,
        "missing_pos": float(np.isnan(x).mean()) if len(x) == n and n else 1.0,
        "missing_ahead": float(np.isnan(ahead).mean()) if len(ahead) == n and n else 1.0,
    }
    return a16, a32, quality


def convert_session(year: int, event: str, session: str, out_path: str) -> tuple[str, int, int]:
    """Converte todas as voltas de uma sessão e grava um .npz em parts/. Devolve (arquivo, voltas, falhas)."""
    folder = SRC / str(year) / event / session
    rows, a16s, a32s, failed = [], [], [], 0
    for drv in sorted(p for p in folder.iterdir() if p.is_dir()):
        for f in drv.glob("*_tel.json"):
            try:
                tel = loads(f.read_bytes())["tel"]
                a16, a32, q = resample(tel)
            except Exception:                         # arquivo corrompido ou sem os campos básicos
                failed += 1
                continue
            rows.append({"driver": drv.name, "lap": int(f.name.split("_")[0]), **q})
            a16s.append(a16)
            a32s.append(a32)
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["driver"], rows[i]["lap"]))
    idx = pd.DataFrame([rows[i] for i in order])
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, tel_f16=np.stack([a16s[i] for i in order]) if rows else np.zeros((0, GRID, len(F16)), np.float16),
             tel_f32=np.stack([a32s[i] for i in order]) if rows else np.zeros((0, GRID, len(F32)), np.float32),
             index=idx.to_json(orient="split"))
    os.replace(tmp, out_path)
    return out_path, len(rows), failed


def source_commit(year: int) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=SRC / str(year), capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sessions(years: list[int]) -> list[tuple[int, str, str]]:
    out = []
    for y in years:
        for gp in sorted(p for p in (SRC / str(y)).iterdir() if p.is_dir() and not p.name.startswith(".")):
            out += [(y, gp.name, s) for s in SESSIONS if (gp / s).is_dir()]
    return out


def part_name(year: int, event: str, session: str) -> str:
    return f"{year}_{event.replace(' ', '_')}_{SESSIONS[session]}.npz"


def newest_source(year: int, event: str, session: str) -> float:
    folder = SRC / str(year) / event / session
    return max((f.stat().st_mtime for f in folder.glob("*/*_tel.json")), default=0.0)


def assemble(races: pd.DataFrame, commits: dict) -> None:
    """Junta as partes num índice e em dois arrays .npy contínuos (leitura por memmap no treino)."""
    parts = sorted((OUT / "parts").glob("*.npz"))
    frames, n = [], 0
    for p in parts:
        year, session = int(p.stem[:4]), p.stem[-1]
        with np.load(p) as z:
            idx = pd.read_json(io.StringIO(z["index"].item()), orient="split")
        if idx.empty:
            continue
        event = p.stem[5:-2].replace("_", " ")
        frames.append(idx.assign(year=year, event=event, session=session, part=p.name, part_row=np.arange(len(idx))))
        n += len(idx)
    index = pd.concat(frames, ignore_index=True)
    index = index.merge(races, on=["year", "event"], how="left")
    index = index.sort_values(["year", "round", "session", "driver", "lap"], na_position="last").reset_index(drop=True)
    index["row"] = np.arange(len(index), dtype=np.int64)

    f16 = np.lib.format.open_memmap(OUT / "tel_f16.npy.tmp", mode="w+", dtype=np.float16, shape=(n, GRID, len(F16)))
    f32 = np.lib.format.open_memmap(OUT / "tel_f32.npy.tmp", mode="w+", dtype=np.float32, shape=(n, GRID, len(F32)))
    for part, g in index.groupby("part", sort=False):
        with np.load(OUT / "parts" / part) as z:
            f16[g["row"].to_numpy()] = z["tel_f16"][g["part_row"].to_numpy()]
            f32[g["row"].to_numpy()] = z["tel_f32"][g["part_row"].to_numpy()]
    f16.flush(), f32.flush()
    del f16, f32
    os.replace(OUT / "tel_f16.npy.tmp", OUT / "tel_f16.npy")
    os.replace(OUT / "tel_f32.npy.tmp", OUT / "tel_f32.npy")

    cols = ["row", "year", "round", "race_id", "event", "circuit", "session", "driver", "lap", "n_samples",
            "duration_s", "distance_m", "start_distance_m", "max_gap_s", "n_backward", "missing_pos", "missing_ahead"]
    index[cols].to_parquet(OUT / "index.parquet.tmp", index=False)
    os.replace(OUT / "index.parquet.tmp", OUT / "index.parquet")
    meta = {
        "grid": GRID, "grid_definition": "pontos igualmente espaçados na distância da volta, da primeira à última amostra",
        "tel_f16": [{"channel": c, "json_field": f, "unit": u, "interpolation": h} for c, f, u, h in F16],
        "tel_f32": [{"channel": c, "json_field": f, "unit": u, "interpolation": h} for c, f, u, h in F32],
        "n_laps": int(n), "source": "github.com/TracingInsights/<ano>, arquivos <volta>_tel.json",
        "source_commits": commits, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    unmatched = index.loc[index["race_id"].isna(), ["year", "event"]].drop_duplicates()
    print(f"{n:,} voltas -> {OUT.relative_to(ROOT)} (tel_f16.npy {(OUT / 'tel_f16.npy').stat().st_size / 1e9:.2f} GB, "
          f"tel_f32.npy {(OUT / 'tel_f32.npy').stat().st_size / 1e9:.2f} GB)")
    if len(unmatched):
        print(f"  GPs sem race_id em race_laps.parquet: {unmatched.values.tolist()}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=YEARS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--rebuild", action="store_true", help="converte de novo mesmo as sessões já em parts/")
    args = ap.parse_args()

    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    todo = []
    for y, ev, s in sessions(args.years):
        out = OUT / "parts" / part_name(y, ev, s)
        if args.rebuild or not out.exists() or out.stat().st_mtime < newest_source(y, ev, s):
            todo.append((y, ev, s, str(out)))
    print(f"{len(todo)} sessões para converter ({args.workers} processos)", flush=True)
    t0, laps, failed = time.perf_counter(), 0, 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(convert_session, *job) for job in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            path, n, bad = fut.result()
            laps += n
            failed += bad
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} sessões, {laps:,} voltas, {time.perf_counter() - t0:.0f} s", flush=True)
    if failed:
        print(f"  {failed} arquivos não puderam ser lidos (ficaram de fora)")

    races = pd.read_parquet(RACE_LAPS, columns=["year", "event", "round", "race_id", "circuit"]).drop_duplicates()
    assemble(races, {str(y): source_commit(y) for y in YEARS if (SRC / str(y)).exists()})


if __name__ == "__main__":
    main()
