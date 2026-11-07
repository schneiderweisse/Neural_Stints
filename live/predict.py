"""Previsões da GRU sobre o histórico ao vivo.

Acompanha data/live/<sessão>/laps.parquet (gravado pelo live.reader). A cada volta nova de um piloto,
monta as variáveis com live/history.py (as mesmas funções do pipeline), codifica com a normalização e
os vocabulários do treino guardados no model.pt e prevê as próximas 10 voltas no cenário A (o piloto
segue na pista com o pneu atual). Se o run for um ensemble (modeling.ensemble), a previsão é a média
dos quantis de todos os membros, como nas previsões de validação e teste do run. Grava data/live/<sessão>/predictions.parquet no formato padrão de
docs/resultados.md, com split = live; o valor real de cada volta prevista é preenchido quando ela
termina, e aí dá para medir o erro ao vivo.

Só faz sentido em corridas (e sprints): os treinos livres não têm total de voltas.

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/python -m live.predict                           # sessão de corrida mais recente
    .venv/Scripts/python -m live.predict --session 2026_interlagos_race
    .venv/Scripts/python -m live.predict --session <pasta> --once  # todas as voltas de uma vez (avaliação)
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from live import history
from live.history import LIVE
from live.power import keep_awake
from modeling.gru import Seq2SeqGRU, latest_run
from modeling.results import finish_predictions, write_atomic
from pipeline import sequences as sq
from pipeline.race_laps import CIRCUITS, regulation

SESSION_TYPES = ("race", "sprint")


class LivePredictor:
    def __init__(self, session_dir: Path, run_dir: Path | None = None, reference: pd.DataFrame | None = None):
        self.dir = Path(session_dir)
        self.run_dir = run_dir or latest_run("gru")
        ck = torch.load(self.run_dir / "model.pt", weights_only=False)
        # um ensemble (modeling.ensemble) guarda vários membros; um run comum é um membro só
        self.models = []
        for m in ck.get("members", [ck]):
            net = Seq2SeqGRU(**m["model_args"])
            net.load_state_dict(m["state_dict"])
            net.eval()
            norm = m["config"]["target_normalization"]
            self.models.append((net, norm["mean"], norm["std"]))
        self.meta, self.config = ck["dataset_meta"], ck["config"]
        self.H, self.min_context = self.meta["horizon"], self.meta["min_context"]

        info = json.loads((self.dir / "session.json").read_text(encoding="utf-8"))
        self.year = int(str(info["StartDate"])[:4])
        self.event = info["Meeting"]["Name"]
        circuits = pd.read_csv(CIRCUITS)
        hit = circuits[(circuits["year"] == self.year) & (circuits["event"] == self.event)]
        self.circuit = hit["circuit"].iloc[0] if len(hit) else info["Meeting"]["Circuit"]["ShortName"]
        # pasta <ano>_<circuito>_<sessão> -> circuito, para achar as outras sessões do fim de semana
        slug = self.dir.name.removeprefix(f"{self.year}_")
        slug = slug.removesuffix("_sprint") if slug.endswith("_sprint") else slug.removesuffix("_race")
        if reference is None:
            reference, self.ref_source = history.event_reference(self.year, self.event, slug)
        else:
            self.ref_source = "dada"
        self.reference = reference
        self.run_id = f"live_{self.dir.name}"
        self.predictions = pd.DataFrame()
        self.last_anchor: dict[str, int] = {}

    # -------------------------------------------------------------- variáveis
    def features(self, laps: pd.DataFrame) -> pd.DataFrame:
        feats = history.race_features(laps, self.reference, self.year, self.event, self.circuit, "live")
        return sq.model_columns(feats)

    def encode_known(self, df: pd.DataFrame) -> np.ndarray:
        return sq.encode(df, sq.KNOWN_NUM, sq.KNOWN_BOOL, self.meta["stats"])

    def future(self, last: pd.Series, n: int) -> pd.DataFrame:
        """Cenário A: mesmo pneu, sem parada, nas próximas n voltas."""
        k = np.arange(1, n + 1)
        lap = last["lap"] + k
        total = float(last["laps_total"])
        return pd.DataFrame({
            "lap": lap, "race_progress": lap / total, "laps_remaining": total - lap,
            "tyre_life": last["tyre_life"] + k, "stint_lap": last["stint_lap"] + k, "stint": last["stint"],
            "fresh_tyre": bool(last["fresh_tyre"]), "pit_in": False, "pit_out": False,
            "is_soft": bool(last["is_soft"]), "is_medium": bool(last["is_medium"]), "is_hard": bool(last["is_hard"]),
            "compound": last["compound"],
        })

    def static(self, g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        first = g.head(1)
        cat = [self.meta["vocabs"][c].index(str(first[c].iloc[0])) if str(first[c].iloc[0]) in self.meta["vocabs"][c]
               else 0 for c in sq.STATIC_CAT]
        return np.array([cat], np.int64), sq.encode(first, sq.STATIC_NUM, sq.STATIC_BOOL, self.meta["stats"])

    # -------------------------------------------------------------- previsão
    def forecast_driver(self, g: pd.DataFrame, t: int) -> pd.DataFrame | None:
        """Previsão feita ao fim da volta t de um piloto (g: todas as voltas dele, em ordem)."""
        hist = g[g["lap"] <= t]
        last = hist.iloc[-1]
        n = int(min(self.H, float(last["laps_total"]) - t))
        if t < self.min_context or n <= 0 or bool(last["pit_in"]) or pd.isna(last["ref_lap"]):
            return None                                   # na volta de entrada nos boxes o pneu ainda vai mudar
        fut = self.future(last, n)
        enc = np.concatenate([sq.encode(hist, sq.OBSERVED_NUM, sq.OBSERVED_BOOL, self.meta["stats"]),
                              self.encode_known(hist)], axis=1)
        cat, sx = self.static(g)
        inputs = (torch.from_numpy(enc[None]), torch.from_numpy(self.encode_known(fut)[None]),
                  torch.from_numpy(cat), torch.from_numpy(sx))
        with torch.no_grad():                             # ensemble: média dos quantis dos membros
            q = np.mean([net.forecast(*inputs)[0].numpy() * std + mean for net, mean, std in self.models], axis=0)
        return pd.DataFrame({
            "split": "live", "race_id": "live", "race_label": f"{self.year} {self.circuit}", "circuit": self.circuit,
            "year": self.year, "regulation": regulation(self.year), "driver": last["driver"], "team": last["team"],
            "t": t, "h": np.arange(1, n + 1), "lap": fut["lap"].to_numpy(), "compound": last["compound"],
            "tyre_life": fut["tyre_life"].to_numpy(), "stint": last["stint"], "same_stint": True,
            "ref_lap": last["ref_lap"], "pace_p10": q[:, 0], "pace_p50": q[:, 1], "pace_p90": q[:, 2],
            "pace_real": np.nan, "clean": False,
        })

    def fill_real(self, feats: pd.DataFrame) -> None:
        """Valor real das voltas previstas que já terminaram."""
        if self.predictions.empty:
            return
        real = feats[["driver", "lap", "pace", "clean"]].rename(columns={"pace": "real_pace", "clean": "real_clean"})
        p = self.predictions.drop(columns=["real_pace", "real_clean"], errors="ignore").merge(real, on=["driver", "lap"],
                                                                                           how="left")
        p["pace_real"] = p["real_pace"]
        p["clean"] = p["real_clean"].fillna(False).astype(bool)
        p["time_real"] = (1 + p["pace_real"]) * p["ref_lap"]
        self.predictions = p.drop(columns=["real_pace", "real_clean"])

    def add(self, frames: list[pd.DataFrame]) -> None:
        frames = [f for f in frames if f is not None]
        if not frames:
            return
        new = finish_predictions(pd.concat(frames, ignore_index=True), self.run_id, "gru")
        new["predicted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        new["scenario"], new["scenario_compound"], new["scenario_pit_lap"] = "A", None, np.nan
        new["gru_run"] = self.run_dir.name
        self.predictions = pd.concat([self.predictions, new], ignore_index=True)

    def step(self, laps: pd.DataFrame) -> int:
        """Prevê a partir da última volta de cada piloto que avançou. Devolve quantas previsões fez."""
        feats = self.features(laps)
        frames = []
        for drv, g in feats.groupby("driver"):
            g = g.sort_values("lap")
            t = int(g["lap"].max())
            if t > self.last_anchor.get(drv, 0):
                frames.append(self.forecast_driver(g, t))
                self.last_anchor[drv] = t
        n = sum(f is not None for f in frames)
        self.add(frames)
        self.fill_real(feats)
        return n

    def replay_all(self, laps: pd.DataFrame) -> int:
        """Todas as âncoras de todos os pilotos, como se a sessão estivesse acontecendo (avaliação)."""
        feats = self.features(laps)
        frames = []
        for drv, g in feats.groupby("driver"):
            g = g.sort_values("lap")
            frames += [self.forecast_driver(g, t) for t in range(self.min_context, int(g["lap"].max()) + 1)]
        self.add(frames)
        self.fill_real(feats)
        return sum(f is not None for f in frames)

    def save(self) -> Path:
        path = self.dir / "predictions.parquet"
        write_atomic(path, lambda p: self.predictions.to_parquet(p, index=False))
        return path


def latest_race_session(root: Path = LIVE) -> Path | None:
    dirs = [d for d in root.iterdir() if d.is_dir() and d.name.endswith(SESSION_TYPES) and (d / "session.json").exists()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="pasta em data/live (padrão: a corrida mais recente)")
    ap.add_argument("--run", help="pasta em runs/ do modelo (padrão: a GRU mais recente)")
    ap.add_argument("--once", action="store_true", help="prevê todas as voltas do histórico e sai")
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()

    session_dir = LIVE / args.session if args.session else None
    while session_dir is None:
        session_dir = latest_race_session()
        if session_dir is None:
            print(f"{datetime.now():%H:%M:%S} nenhuma corrida em data/live ainda; aguardando o live.reader…")
            time.sleep(args.interval * 5)
    run_dir = Path("runs") / args.run if args.run else None
    pred = LivePredictor(session_dir, run_dir)
    print(f"{session_dir.name}: modelo {pred.run_dir.name} · circuito {pred.circuit} · referência: {pred.ref_source} "
          f"({len(pred.reference)} pilotos)")
    laps_path = session_dir / "laps.parquet"
    if args.once:
        n = pred.replay_all(pd.read_parquet(laps_path))
        print(f"{n} previsões -> {pred.save()}")
        return
    if keep_awake(display=True):
        print("modo acordado: o Windows não suspende nem apaga a tela enquanto o previsor roda (não feche a tampa)")
    seen = None
    try:
        while True:
            if laps_path.exists() and laps_path.stat().st_mtime != seen:
                seen = laps_path.stat().st_mtime
                n = pred.step(pd.read_parquet(laps_path))
                pred.save()
                if n:
                    p = pred.predictions
                    last = p[p["t"] == p.groupby("driver")["t"].transform("max")]
                    nxt = last[last["h"] == 1].sort_values("time_p50").head(3)
                    print(f"{datetime.now():%H:%M:%S} {n} pilotos previstos · próxima volta: " +
                          " ".join(f"{r.driver} {r.time_p50:.2f}s" for r in nxt.itertuples()))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("previsões encerradas")


if __name__ == "__main__":
    main()
