"""Duelos previstos pela rede numa corrida, para o modo "Duelos previstos" da aba Replay.

A regra é a do painel ao vivo e do backtest (modeling/duels.py, modeling/duels_backtest.py). Ao fim da
volta t, para um carro (B) de 1 a 5 s atrás do carro logo à frente (A) na linha, a rede projeta o gap das
10 voltas seguintes somando as diferenças de tempo previstas. O duelo é previsto quando a chance de B
chegar a menos de 1 s de A passa de 50% dentro do horizonte. A janela de 80% vai da volta em que essa
chance passa de 10% à volta em que passa de 90%.

O que aconteceu vem dos tempos oficiais (TracingInsights): o gap na linha, volta a volta. A conta para
na primeira volta com parada, SC/VSC/bandeira vermelha, pneu de chuva ou sem tempo de algum dos dois,
porque a projeção supõe pista livre e ninguém parando.

Duelos em curso: quando o carro de trás já está a menos de 1 s, a pergunta passa a ser se ele passa. Com
a mesma projeção (e a mesma normal), a chance de ultrapassagem até a volta k é a de o gap na linha ficar
abaixo de 0; a ultrapassagem é prevista quando ela passa de 50%, e a janela vai de 10% a 90%. Atenção:
a rede prevê o ritmo de cada piloto sem saber do outro (ar sujo, DRS), e essa chance não foi calibrada
para ultrapassagens. Em Las Vegas, ela foi pior que prever sempre a taxa média (Brier no resumo).

Previsões seguidas do mesmo par (volta t, t+1, ...) formam um episódio. O resultado do episódio é o da
primeira previsão, o aviso mais cedo. O episódio também guarda as previsões das voltas seguintes, para o
player mostrar como a rede foi ajustando a conta.

Previsões: as feitas ao vivo (data/live/<sessão>/predictions.parquet) e, nas origens que faltam (as
perdidas numa queda do previsor), as refeitas pelo live.predict --once (predictions_replay.parquet). Em
Las Vegas, as refeitas são idênticas às do ao vivo em 97% das origens em comum; nas outras, a diferença
chega a 0,11 s, por voltas que a cronometragem corrigiu depois. Como no previsor ao vivo, só valem
origens em voltas gravadas dos dois pilotos.

Saída: data/processed/replay/<ano>_<gp>_duels.json, ao lado do replay da corrida.

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/python -m modeling.duel_replay 2026 "Las Vegas Grand Prix" --live 2026_las_vegas_race
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import duels

ROOT = Path(__file__).resolve().parents[1]
TEL = ROOT / "telemetry" / "tracinginsights"
LIVE = ROOT / "data" / "live"
OUT = ROOT / "data" / "processed" / "replay"
H = 10
MIN_GAP, MAX_GAP = duels.BATTLE_GAP, 5.0
WET = {"INTERMEDIATE", "WET"}


def num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower().replace("grand prix", "")).strip("_")


def official_laps(folder: Path) -> pd.DataFrame:
    """Uma linha por piloto e volta: fim da volta (tempo de sessão), tempo, posição e o que encerra a conta."""
    rows = []
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        lt = json.loads((d / "laptimes.json").read_text(encoding="utf-8-sig"))
        for i, lap in enumerate(lt["lap"]):
            status = str(lt["status"][i])
            rows.append({"driver": d.name, "lap": int(lap), "end": num(lt["sesT"][i]), "time": num(lt["time"][i]),
                         "pos": num(lt["pos"][i]), "pit": num(lt["pin"][i]) is not None or num(lt["pout"][i]) is not None,
                         "neutral": any(c in status for c in "4567"), "wet": str(lt["compound"][i]).upper() in WET})
    return pd.DataFrame(rows)


def forecast_arrays(preds: pd.DataFrame) -> dict:
    """{(piloto, t): matriz (H, 3) com p10, p50 e p90 do tempo de volta, em segundos}."""
    out = {}
    for (drv, t), g in preds.groupby(["driver", "t"]):
        m = np.full((H, 3), np.nan)
        m[g["h"].to_numpy(int) - 1] = g[["time_p10", "time_p50", "time_p90"]].to_numpy(float)
        out[(drv, int(t))] = m
    return out


def outcome(laps: pd.DataFrame, behind: str, ahead: str, t: int, gap0: float) -> dict:
    """Gap real na linha nas H voltas depois de t, como no backtest, seguindo depois da disputa até a
    ultrapassagem (ou o fim do horizonte) para o player mostrar o desfecho."""
    path, gap, catch, passed, stop = [round(gap0, 3)], gap0, None, None, None
    for k in range(1, H + 1):
        kb, ka = (behind, t + k), (ahead, t + k)
        if kb not in laps.index or ka not in laps.index:
            stop = "fim da corrida" if k > 1 else "sem dados"
            break
        for key in (kb, ka):
            row = laps.loc[key]
            if row["pit"]:
                stop = f"parada de {key[0]} na volta {t + k}"
            elif row["neutral"]:
                stop = f"safety car ou VSC na volta {t + k}"
            elif row["wet"]:
                stop = f"pneu de chuva na volta {t + k}"
            elif pd.isna(row["time"]):
                stop = f"volta {t + k} sem tempo"
            if stop:
                break
        if stop:
            break
        gap += laps.at[kb, "time"] - laps.at[ka, "time"]
        path.append(round(float(gap), 3))
        if catch is None and gap <= duels.BATTLE_GAP:
            catch = k
        if passed is None and gap < 0:
            passed = k
    return {"k": catch, "lap": t + catch if catch else None, "pass_lap": t + passed if passed else None,
            "path": path, "stop": stop}


def pass_window(pr: dict) -> dict:
    """Duelo em curso: chance de o carro de trás já ter passado (gap na linha abaixo de 0) até a volta k,
    com a mesma normal da projeção, e a volta provável e a janela (10%–90%) da ultrapassagem."""
    prob = duels.normal_cdf((0.0 - pr["mu"]) / pr["sd"])
    return duels.catch_window(np.maximum.accumulate(prob) if len(prob) else prob)


def pass_text(pr: dict, w: dict, t: int) -> str:
    n = len(pr["mu"])
    pct = "≥99%" if w["p_horizon"] >= 0.995 else f"{w['p_horizon']:.0%}"
    if w["likely"] is None:
        return f"Sem ultrapassagem prevista nas próximas {n} voltas ({pct} de chance)"
    window = (f"janela {t + w['early']}–{t + w['late']}" if not w["late_open"]
              else f"janela da volta {t + w['early']} em diante")
    return f"Ultrapassagem prevista na volta {t + w['likely']} ({window}; {pct} de chance nas próximas {n} voltas)"


def judge(kind: str, t: int, w: dict, o: dict) -> dict:
    """Resultado de uma previsão. Duelo que começa: o carro de trás chegou a menos de 1 s? Duelo em curso:
    a rede previu ultrapassagem (ou não) e o carro de trás passou (ou não)?"""
    def inside(k):
        return k is not None and w["early"] is not None and w["early"] <= k and (w["late_open"] or k <= w["late"])
    cut = o["stop"] and o["stop"] != "fim da corrida"
    if kind == "inicio":
        cls = "aconteceu" if o["k"] else "interrompido" if cut else "nao_aconteceu"
        return {"inside": inside(o["k"]), "cls": cls, "correct": None if cls == "interrompido" else cls == "aconteceu"}
    k_pass = o["pass_lap"] - t if o["pass_lap"] else None
    cls = "passou" if k_pass else "interrompido" if cut else "segurou"
    predicted = w["likely"] is not None
    return {"inside": inside(k_pass), "cls": cls,
            "correct": None if cls == "interrompido" else (cls == "passou") == predicted}


def build(year: int, event: str, session: str) -> Path:
    folder = TEL / str(year) / event / "Race"
    laps = official_laps(folder).set_index(["driver", "lap"]).sort_index()
    live = LIVE / session
    preds = pd.read_parquet(live / "predictions.parquet").assign(source="ao vivo")
    n_live = preds[["driver", "t"]].drop_duplicates().shape[0]
    if (live / "predictions_replay.parquet").exists():           # completa as origens que faltam
        rep = pd.read_parquet(live / "predictions_replay.parquet").assign(source="refeita")
        have = set(map(tuple, preds[["driver", "t"]].drop_duplicates().itertuples(index=False)))
        rep = rep[[(d, t) not in have for d, t in zip(rep["driver"], rep["t"])]]
        preds = pd.concat([preds, rep], ignore_index=True)
    fc = forecast_arrays(preds)
    src = preds.drop_duplicates(["driver", "t"]).set_index(["driver", "t"])["source"]
    recorded = set(map(tuple, pd.read_parquet(live / "laps.parquet")[["driver", "lap"]].itertuples(index=False)))

    # situações: B logo atrás de A na linha ao fim da volta t, os dois com previsão em t. A 1–5 s, o duelo
    # ainda pode começar ("inicio"); a menos de 1 s, ele já está em curso ("em_curso")
    proj = {}
    by_lap = laps.reset_index()
    for t, same in by_lap.groupby("lap"):
        t = int(t)
        same = same.dropna(subset=["pos", "end"]).set_index("pos")
        for p, b in same.iterrows():
            if p < 2 or (p - 1) not in same.index:
                continue
            a = same.loc[p - 1]
            B, A = b["driver"], a["driver"]
            gap0 = float(b["end"] - a["end"])
            if not (0 < gap0 <= MAX_GAP) or (B, t) not in recorded or (A, t) not in recorded:
                continue
            if (B, t) not in fc or (A, t) not in fc or b["pit"] or a["pit"] or b["neutral"] or b["wet"] or a["wet"]:
                continue
            pr = duels.project(gap0, fc[(B, t)], fc[(A, t)])
            if not len(pr["caught"]):
                continue
            if gap0 > MIN_GAP:
                kind, w, text = "inicio", duels.catch_window(pr["caught"]), duels.describe(gap0, pr, t)
            else:
                kind = "em_curso"
                w = pass_window(pr)
                text = pass_text(pr, w, t)
            proj[(B, A, t)] = {"gap0": gap0, "pr": pr, "w": w, "kind": kind, "text": text,
                               "live": src.get((B, t)) == "ao vivo" and src.get((A, t)) == "ao vivo"}

    def pack(t: int, s: dict) -> dict:
        w, pr = s["w"], s["pr"]
        return {"t": t, "kind": s["kind"], "gap0": round(s["gap0"], 3), "mu": np.round(pr["mu"], 3).tolist(),
                "sd": np.round(pr["sd"], 3).tolist(), "early": w["early"], "likely": w["likely"], "late": w["late"],
                "late_open": w["late_open"], "p": round(w["p_horizon"], 3), "text": s["text"], "live": bool(s["live"])}

    # episódios: previsões seguidas do mesmo par e do mesmo tipo. Duelo que começa: só as que preveem a
    # disputa. Duelo em curso: todas as voltas do par a menos de 1 s; o resultado do episódio é o da
    # primeira previsão, e as das voltas seguintes (que podem mudar de ideia) vão em "updates"
    def flag(s):
        return s["w"]["likely"] is not None
    keys = sorted((k for k, s in proj.items() if s["kind"] == "em_curso" or flag(s)), key=lambda k: (k[0], k[1], k[2]))
    episodes, cur = [], None
    for B, A, t in keys:
        s = proj[(B, A, t)]
        if (cur and cur["behind"] == B and cur["ahead"] == A and t == cur["origins"][-1] + 1
                and cur["kind"] == s["kind"]):
            cur["origins"].append(t)
            continue
        cur = {"behind": B, "ahead": A, "origins": [t], "kind": s["kind"]}
        episodes.append(cur)
    out_eps = []
    for e in episodes:
        B, A, t0 = e["behind"], e["ahead"], e["origins"][0]
        s0 = proj[(B, A, t0)]
        o = outcome(laps, B, A, t0, s0["gap0"])
        # previsões das voltas seguintes para o mesmo par, de qualquer tipo, dentro do horizonte
        updates = [pack(t, proj[(B, A, t)]) for t in range(t0, t0 + H) if (B, A, t) in proj]
        out_eps.append({"kind": e["kind"], "behind": B, "ahead": A, "t0": t0, "origins": e["origins"],
                        "start": float(laps.at[(B, t0), "end"]), "forecast": pack(t0, s0), "updates": updates,
                        "outcome": {**o, **judge(e["kind"], t0, s0["w"], o)}})
    out_eps.sort(key=lambda e: (e["t0"], e["behind"]))

    # resumos: por previsão (cada volta) e por episódio (a primeira previsão de cada um)
    def start_summary(items: list[dict]) -> dict:
        done = [i for i in items if i["cls"] != "interrompido"]
        hit = [i for i in done if i["cls"] == "aconteceu"]
        return {"n": len(items), "interrompidos": len(items) - len(done), "aconteceram": len(hit),
                "precisao": round(len(hit) / max(len(done), 1), 3),
                "na_janela": round(sum(i["inside"] for i in hit) / max(len(hit), 1), 3)}

    def ongoing_summary(items: list[tuple[bool, dict]]) -> dict:
        done = [(p, i) for p, i in items if i["cls"] != "interrompido"]
        pp = [i for p, i in done if p]
        hp = [i for p, i in done if not p]
        return {"n": len(items), "interrompidos": len(items) - len(done),
                "previu_ultrapassagem": len(pp), "ultrapassagem_confirmada": sum(i["cls"] == "passou" for i in pp),
                "na_janela": sum(i["inside"] for i in pp if i["cls"] == "passou"),
                "previu_que_segura": len(hp), "segurou": sum(i["cls"] == "segurou" for i in hp),
                "ultrapassagens_reais": sum(i["cls"] == "passou" for _, i in done),
                "acerto": round(sum(i["correct"] for _, i in done) / max(len(done), 1), 3)}

    judged = {k: judge(proj[k]["kind"], k[2], proj[k]["w"], outcome(laps, k[0], k[1], k[2], proj[k]["gap0"])) for k in keys}
    # calibração da chance de ultrapassagem, por previsão, contra a taxa base (prever sempre a média)
    pairs = [(proj[k]["w"]["p_horizon"], judged[k]["cls"] == "passou") for k in keys
             if proj[k]["kind"] == "em_curso" and judged[k]["cls"] != "interrompido"]
    p_hat, y = np.array([p for p, _ in pairs], float), np.array([v for _, v in pairs], float)
    brier = {"rede": round(float(np.mean((p_hat - y) ** 2)), 3) if len(y) else None,
             "taxa_base": round(float(np.mean((y.mean() - y) ** 2)), 3) if len(y) else None,
             "taxa_de_ultrapassagem": round(float(y.mean()), 3) if len(y) else None}
    res = {"year": year, "event": event, "session": session, "live_origins": n_live,
           "origins": int(preds[["driver", "t"]].drop_duplicates().shape[0]),
           "run": str(preds["gru_run"].iloc[0]) if "gru_run" in preds else "",
           "first_origin": min((t for _, _, t in proj), default=None),
           "situations": {"inicio": sum(s["kind"] == "inicio" for s in proj.values()),
                          "em_curso": sum(s["kind"] == "em_curso" for s in proj.values())},
           "summary_origins": start_summary([judged[k] for k in keys if proj[k]["kind"] == "inicio"]),
           "summary_episodes": start_summary([e["outcome"] for e in out_eps if e["kind"] == "inicio"]),
           "summary_ongoing_origins": ongoing_summary([(flag(proj[k]), judged[k]) for k in keys if proj[k]["kind"] == "em_curso"]),
           "summary_ongoing": ongoing_summary([(e["forecast"]["likely"] is not None, e["outcome"])
                                               for e in out_eps if e["kind"] == "em_curso"]),
           "brier_ongoing": brier,
           "episodes": out_eps}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{year}_{slug(event)}_duels.json"
    path.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("event", help='pasta do GP no TracingInsights, ex.: "Las Vegas Grand Prix"')
    ap.add_argument("--live", required=True, help="pasta da corrida em data/live, com as previsões")
    args = ap.parse_args()
    path = build(args.year, args.event, args.live)
    r = json.loads(path.read_text(encoding="utf-8"))
    st = r["situations"]
    print(f"{path}: desde a volta {r['first_origin']}, {st['inicio']} situações de par a 1–5 s e {st['em_curso']} de "
          f"duelo em curso; previsões de {r['origins']} origens, {r['live_origins']} feitas ao vivo e o resto refeito com "
          f"live.predict --once")
    for label, s in (("por previsão", r["summary_origins"]), ("por episódio", r["summary_episodes"])):
        print(f"duelos que começam, {label}: {s['n']} ({s['interrompidos']} sem desfecho); aconteceram {s['precisao']:.0%} "
              f"dos outros, {s['na_janela']:.0%} deles dentro da janela")
    for label, s in (("por previsão", r["summary_ongoing_origins"]), ("por episódio", r["summary_ongoing"])):
        print(f"duelos em curso, {label}: {s['n']} ({s['interrompidos']} sem desfecho); ultrapassagem prevista em "
              f"{s['previu_ultrapassagem']}, confirmada em {s['ultrapassagem_confirmada']} ({s['na_janela']} na janela); "
              f"'segura' previsto em {s['previu_que_segura']}, confirmado em {s['segurou']}; ultrapassagens reais "
              f"{s['ultrapassagens_reais']}; acerto {s['acerto']:.0%}")
    b = r["brier_ongoing"]
    print(f"chance de ultrapassagem: Brier {b['rede']} da rede × {b['taxa_base']} prevendo sempre a taxa de "
          f"{b['taxa_de_ultrapassagem']:.0%}")


if __name__ == "__main__":
    main()
