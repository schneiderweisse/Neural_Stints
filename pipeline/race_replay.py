"""Replay de uma corrida pela telemetria do TracingInsights, para a aba Replay do painel.

A telemetria traz, para cada carro e volta, a distância percorrida e a posição x/y (~7,5 amostras por
segundo). Com ela, este script monta, numa grade de tempo da sessão:

- o progresso na corrida (voltas completas + distância na volta), a ordem e o intervalo para o carro
  da frente (a cada 0,5 s). O intervalo é o tempo que o carro de trás leva para chegar aonde o da
  frente está. Na linha de chegada, ele fica a 0,03 s (mediana) do gap oficial gravado ao vivo;
- a posição de cada carro no mapa (a cada 0,25 s): o ponto do traçado de referência (a volta mais
  rápida do vencedor) na fração da volta em que ele está, e o x/y real no pit lane. O x/y real não é
  usado na pista porque às vezes congela por vários segundos;
- as janelas de box, os pneus por volta, as bandeiras (track status) e as mensagens do race control.

O progresso de cada carro também vai no arquivo (float32), para o player calcular o gap entre dois carros
quaisquer (modo "Duelos previstos", com modeling/duel_replay.py).

Os duelos (menos de 1 s para o carro da frente ou de trás, com bandeira verde, fora do box) são achados
no player (dashboard/replay_player.html), que deixa mudar o corte.

Entrada: telemetry/tracinginsights/<ano>/<GP>/Race/ (pipeline/fetch_telemetry.py <ano> --event "<GP>").
Saída: data/processed/replay/<ano>_<gp>.json (~4 MB; os arrays vão em base64).

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/python -m pipeline.race_replay 2026 "Las Vegas Grand Prix"
"""
import argparse
import base64
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
TEL = ROOT / "telemetry" / "tracinginsights"
OUT = ROOT / "data" / "processed" / "replay"
DT_XY, DT = 0.25, 0.5        # grade do mapa e da ordem/intervalos, em segundos de sessão
DUEL_GAP = 1.0               # duelo: menos de 1 s para o carro da frente (padrão do player)
LEAD_HOLD = 5.0              # troca de líder: a nova liderança dura pelo menos 5 s
GREEN = {"1", "2"}           # verde e amarela local; SC (4), vermelha (5) e VSC (6, 7) ficam de fora
NO_GAP = 32767
HIDE = 65535


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower().replace("grand prix", "")).strip("_")


# ---------------------------------------------------------------- leitura
def driver_laps(folder: Path) -> tuple[dict, list[dict]]:
    """laptimes.json e as amostras de telemetria de cada volta, já em tempo de sessão."""
    lt = load(folder / "laptimes.json")
    laps = []
    for i, lap in enumerate(lt["lap"]):
        f, start = folder / f"{lap}_tel.json", num(lt["lST"][i])
        if not f.exists() or start is None:
            continue
        t = load(f)["tel"]
        s, x, y, rel = (arr(t[k]) for k in ("time", "x", "y", "rel_distance"))
        ok = ~(np.isnan(s) | np.isnan(x) | np.isnan(y) | ((x == 0) & (y == 0)))   # (0, 0): posição que faltou
        # volta "parada" (o carro não saiu do lugar): é preenchimento, como de quem não largou
        if ok.sum() < 10 or np.ptp(x[ok]) + np.ptp(y[ok]) < 100:
            continue
        s, x, y, rel = s[ok], x[ok], y[ok], rel[ok]
        idx, good = np.arange(len(rel)), ~np.isnan(rel)
        rel = np.interp(idx, idx[good], rel[good]) if good.any() else np.linspace(0, 1, len(rel))
        clean = str(lt["status"][i]) == "1" and num(lt["pin"][i]) is None and num(lt["pout"][i]) is None
        laps.append({"lap": int(lap), "s": start + s, "x": x, "y": y, "rel": rel, "clean": clean and lap > 1,
                     "s1": num(lt["s1T"][i]), "s2": num(lt["s2T"][i])})
    return lt, laps


def rel_at_sectors(laps: list[dict]) -> dict:
    """Distância percorrida (fração da volta) no instante oficial do fim dos setores 1 e 2, nas voltas
    limpas (bandeira verde, sem box, a partir da 2)."""
    return {k: [float(np.interp(l[k], l["s"], l["rel"])) for l in laps
                if l["clean"] and l[k] and l["s"][0] < l[k] < l["s"][-1]] for k in ("s1", "s2")}


def distance_bias(laps: list[dict], sectors: dict) -> tuple[list[float], list[float]]:
    """Viés da distância de um carro: quanto a fração da volta dele no fim de cada setor difere, na
    mediana, da fração desse ponto da pista (a do vencedor). Vira uma correção linear por trechos."""
    at = rel_at_sectors(laps)
    knots, bias = [0.0], [0.0]
    for k in ("s1", "s2"):
        if len(at[k]) >= 5:
            knots.append(sectors[k])
            bias.append(float(np.median(at[k])) - sectors[k])
    return knots + [1.0], bias + [0.0]


def arr(values) -> np.ndarray:
    """Lista do JSON (com "None" nos buracos) como array de floats com NaN."""
    return np.array([np.nan if v is None or v == "None" else v for v in values], float)


def reference(laps: list[dict], lt: dict) -> dict:
    """Traçado de referência: a volta mais rápida sem box e com bandeira verde, em ~1 m."""
    ok = {int(l): num(t) for l, t, s, pi, po in zip(lt["lap"], lt["time"], lt["status"], lt["pin"], lt["pout"])
          if num(t) and str(s) == "1" and num(pi) is None and num(po) is None}
    best = min(ok, key=ok.get)
    lap = next(l for l in laps if l["lap"] == best)
    rel = np.maximum.accumulate(np.clip(lap["rel"], 0, 1))
    grid = np.linspace(0, 1, 6000, endpoint=False)
    x, y = np.interp(grid, rel, lap["x"]), np.interp(grid, rel, lap["y"])
    return {"f": grid, "x": x, "y": y, "tree": cKDTree(np.c_[x, y]), "lap": best}


def progress(laps: list[dict], ref: dict, sectors: dict) -> dict:
    """Amostras de um piloto: tempo, x, y e progresso na corrida (voltas completas + fração da volta).

    A fração da volta é a distância percorrida nela (rel_distance), que vale exatamente 1 na linha.
    O x/y não serve para isso: há trechos em que ele congela por vários segundos (o do VER na volta 1,
    por exemplo) enquanto a distância segue certa.

    A distância vem da velocidade integrada e, no meio da volta, erra um pouco e de um jeito diferente
    em cada carro. No fim do setor 2, o erro é de 0,13 s (mediana, contra o tempo oficial). Parte dele é
    um viés fixo do carro, corrigido por distance_bias: estimado nas voltas ímpares, ele baixa o erro nas
    pares de 0,13 para 0,10 s (setor 2) e de 0,085 para 0,064 s (setor 1). Ancorar cada volta nos fins de
    setor não ajuda: com âncora no setor 1, o erro no setor 2 sobe para 0,14 s.

    Na volta 1, a distância começa em 0 no lugar de cada carro no grid; a projeção do primeiro ponto no
    traçado de referência dá esse ponto de partida."""
    knots, bias = distance_bias(laps, sectors)
    ps = []
    for l in laps:
        rel = l["rel"] - np.interp(l["rel"], knots, bias)
        p = l["lap"] - 1 + rel
        if l["lap"] == 1:
            _, k = ref["tree"].query([l["x"][0], l["y"][0]])
            f0 = ref["f"][k] - round(ref["f"][k])  # o grid fica em volta da linha: entre -0,5 e 0,5
            p = f0 + rel * (1 - f0)
        ps.append(p)
    s = np.concatenate([l["s"] for l in laps])
    x = np.concatenate([l["x"] for l in laps])
    y = np.concatenate([l["y"] for l in laps])
    p = np.concatenate(ps)
    order = np.argsort(s, kind="stable")
    s, x, y, p = s[order], x[order], y[order], np.maximum.accumulate(p[order])
    keep = np.r_[True, np.diff(s) > 1e-6]
    return {"s": s[keep], "x": x[keep], "y": y[keep], "p": p[keep]}


def pit_windows(lt: dict, start: float, last: float) -> list[list[float]]:
    """[entrada, saída] no pit lane, em tempo de sessão. Largada do pit lane: da largada até a saída;
    abandono no box: da entrada até a última amostra."""
    events = sorted([(v, "in") for v in map(num, lt["pin"]) if v] + [(v, "out") for v in map(num, lt["pout"]) if v])
    out, open_at = [], None
    for t, kind in events:
        if kind == "in":
            open_at = t
        else:
            out.append([open_at if open_at is not None else start, t])
            open_at = None
    if open_at is not None:
        out.append([open_at, last])
    return [[round(a, 2), round(b, 2)] for a, b in out]


# ---------------------------------------------------------------- montagem
def build(year: int, event: str) -> Path:
    folder = TEL / str(year) / event / "Race"
    if not folder.exists():
        raise SystemExit(f"{folder} não existe: rode  py pipeline/fetch_telemetry.py {year} --event \"{event}\"")
    info, corners = load(folder / "session_info.json"), load(folder / "corners.json")
    roster = {d["driver"]: d for d in load(folder / "drivers.json")["drivers"]}
    start = float(info["session_start_time"])
    total = int(info["total_laps"])
    t0_utc = datetime.fromisoformat(info["t0_date"])

    raw, dns = {}, []
    for d in sorted(p.name for p in folder.iterdir() if p.is_dir()):
        lt, laps = driver_laps(folder / d)
        if laps:
            raw[d] = (lt, laps)
        else:
            dns.append(d)
    winner = next(d for d, (lt, _) in raw.items() if lt["pos"][-1] == 1)
    ref = reference(raw[winner][1], raw[winner][0])
    # fração da volta no fim dos setores 1 e 2: pontos fixos da pista, pela mediana do vencedor
    sectors = {k: float(np.median(v)) for k, v in rel_at_sectors(raw[winner][1]).items()}

    cars = {d: progress(laps, ref, sectors) for d, (_, laps) in raw.items()}
    # quem terminou: a última amostra vem depois da bandeirada do vencedor
    chequered = float(cars[winner]["s"][-1])
    finished = {d for d, c in cars.items() if c["s"][-1] >= chequered - 1}
    # classificação: voltas completas e, entre iguais, quem cruzou a linha antes
    done = {d: sum(1 for t in raw[d][0]["sesT"] if num(t)) for d in cars}     # voltas com fim registrado
    final = sorted(cars, key=lambda d: (-(d in finished), -done[d], cars[d]["s"][-1] if d in finished
                                        else -cars[d]["p"][-1]))
    names = final                                  # a ordem dos índices é a da classificação
    n = len(names)

    t_first, t_end = start - 8.0, max(float(c["s"][-1]) for c in cars.values()) + 5.0
    gxy = np.arange(t_first, t_end, DT_XY)
    g = np.arange(t_first, t_end, DT)

    # posição no mapa: o ponto do traçado de referência na fração da volta (coerente com a ordem e os
    # gaps); no pit lane, o x/y da telemetria. Escondida depois da última amostra.
    pits = {d: pit_windows(raw[d][0], start, float(cars[d]["s"][-1])) for d in names}
    ref_f, ref_x, ref_y = np.r_[ref["f"], 1.0], np.r_[ref["x"], ref["x"][0]], np.r_[ref["y"], ref["y"][0]]
    xy = np.full((len(gxy), n, 2), HIDE, np.int32)
    raw_xy = np.full((len(gxy), n, 2), np.nan)
    for i, d in enumerate(names):
        c = cars[d]
        f = np.mod(np.interp(gxy, c["s"], c["p"]), 1.0)
        x, y = np.interp(f, ref_f, ref_x), np.interp(f, ref_f, ref_y)
        for a, b in pits[d]:
            box = (gxy >= a) & (gxy <= b)
            x[box], y[box] = np.interp(gxy[box], c["s"], c["x"]), np.interp(gxy[box], c["s"], c["y"])
        show = gxy <= c["s"][-1] + 2.0
        raw_xy[show, i, 0], raw_xy[show, i, 1] = x[show], y[show]

    # progresso e ordem
    P = np.zeros((len(g), n))
    key = np.zeros((len(g), n))
    for i, d in enumerate(names):
        c = cars[d]
        P[:, i] = np.interp(g, c["s"], c["p"])
        after = g > c["s"][-1]
        if d in finished:                          # parado depois da linha, na ordem de chegada
            key[:, i] = np.where(after, done[d] + 0.5 - c["s"][-1] * 1e-6, P[:, i])
        else:                                      # abandono: vai para o fim da lista
            key[:, i] = np.where(after, -1000 + P[:, i], P[:, i])
    order = np.argsort(-key, axis=1, kind="stable").astype(np.uint8)
    retired = np.zeros((len(g), n), bool)
    for i, d in enumerate(names):
        if d not in finished:
            retired[:, i] = g > cars[d]["s"][-1]

    # intervalo para o carro da frente: quanto tempo o de trás leva até onde o da frente está agora
    interval = np.full((len(g), n), NO_GAP, np.int32)
    for p in range(1, n):
        a, b = order[:, p - 1], order[:, p]
        for ia in np.unique(a):
            rows = np.where(a == ia)[0]
            ca = cars[names[ia]]
            pb = P[rows, b[rows]]
            t_a = np.interp(pb, ca["p"], ca["s"])
            gap = g[rows] - t_a
            down = np.floor(P[rows, ia] - pb + 1e-6)
            val = np.where(down >= 1, -down, np.clip(np.round(gap * 100), 0, 30000))
            valid = (g[rows] >= start) & ~retired[rows, b[rows]] & ~retired[rows, ia]
            interval[rows[valid], b[rows][valid]] = val[valid]

    # bandeiras e box
    ts = info["track_status"]
    status_t = np.array(ts["Time"], float)
    status_at = np.array(ts["Status"], str)[np.clip(np.searchsorted(status_t, g, side="right") - 1, 0, None)]

    # traçado (o mesmo do MultiViewer, que vem no corners.json) e o pit lane, pela parada do vencedor
    track = {"x": corners["TrackX"], "y": corners["TrackY"], "time": corners["TrackTime"],
             "rotation": float(corners["Rotation"][0] if isinstance(corners["Rotation"], list) else corners["Rotation"]),
             "corners": [[int(nm), float(x), float(y), float(a)] for nm, x, y, a in
                         zip(corners["CornerNumber"], corners["X"], corners["Y"], corners["Angle"])],
             "finish": [[float(ref["x"][0]), float(ref["y"][0])], [float(ref["x"][8]), float(ref["y"][8])]]}
    wc = cars[winner]
    if pits[winner]:
        a, b = pits[winner][0]
        m = (wc["s"] >= a - 3) & (wc["s"] <= b + 3)
        track["pit"] = np.c_[wc["x"][m], wc["y"][m]][::2].round(1).tolist()

    for i in range(n):
        ok = ~np.isnan(raw_xy[:, i, 0])
        xy[ok, i, 0], xy[ok, i, 1] = np.round(raw_xy[ok, i, 0]), np.round(raw_xy[ok, i, 1])

    # pneus e início de cada volta
    laps_info = {}
    for d in names:
        lt = raw[d][0]
        laps_info[d] = {"lap": [int(v) for v in lt["lap"]], "start": [num(v) for v in lt["lST"]],
                        "compound": [str(v) for v in lt["compound"]], "life": [num(v) for v in lt["life"]],
                        "time": [num(v) for v in lt["time"]]}

    # race control: o que importa para quem assiste (sem a sequência de amarelas e liberações por setor)
    rcm = load(folder / "rcm.json") if (folder / "rcm.json").exists() else None
    messages = []
    if rcm:
        for t, cat, msg, flag, scope in zip(rcm["time"], rcm["cat"], rcm["msg"], rcm["flag"], rcm["scope"]):
            when = (datetime.fromisoformat(str(t)[:26]) - t0_utc).total_seconds()
            if when < start - 60 or "BLUE" in str(flag):
                continue
            if cat == "Flag" and str(scope) == "Sector":
                continue
            messages.append([round(when, 1), str(msg)])

    racing = (g >= start) & (g <= chequered)
    at_line = []                                    # líder ao fim de cada volta (pela cronometragem)
    for lap in range(1, total + 1):
        first = [d for d in names if lap in laps_info[d]["lap"] and raw[d][0]["pos"][laps_info[d]["lap"].index(lap)] == 1]
        at_line.append(first[0] if first else None)
    green = racing & np.isin(status_at, list(GREEN))
    p2 = order[:, 1]
    p1_fight = green & (interval[np.arange(len(g)), p2] >= 0) & (interval[np.arange(len(g)), p2] < DUEL_GAP * 100)

    out = {
        "meta": {"year": year, "event": event, "circuit": info["session_info"]["Meeting"]["Circuit"]["ShortName"],
                 "total_laps": total, "t0": round(float(g[0]), 3), "dt": DT, "dt_xy": DT_XY,
                 "n": len(g), "n_xy": len(gxy), "start": start, "chequered": chequered,
                 "lap_ref": float(np.median([t for t in laps_info[winner]["time"] if t and t < 1.15 * min(
                     t for t in laps_info[winner]["time"] if t)])),
                 "duel_gap": DUEL_GAP, "reference_lap": ref["lap"], "dns": dns,
                 "source": "TracingInsights (telemetria)"},
        "drivers": [{"tla": d, "name": f"{roster.get(d, {}).get('fn', '')} {roster.get(d, {}).get('ln', '')}".strip(),
                     "team": roster.get(d, {}).get("team", raw[d][0]["team"][0]), "color": "#" + roster.get(d, {}).get("tc", "888888"),
                     "number": roster.get(d, {}).get("dn", ""), "finished": d in finished,
                     "laps": done[d], "last": round(float(cars[d]["s"][-1]), 2)}
                    for d in names],
        "track": track, "pits": pits, "laps": laps_info, "rcm": messages,
        "status": [[float(t), str(s)] for t, s in zip(ts["Time"], ts["Status"])],
        "stats": {"lead_changes": lead_changes(order[:, 0], g, racing, LEAD_HOLD),
                  "lead_changes_at_line": sum(1 for a, b in zip(at_line, at_line[1:]) if a and b and a != b),
                  "p1_fight_share": round(float(p1_fight.sum() / max(green.sum(), 1)), 3)},
        "xy": b64(xy.astype(np.uint16)), "order": b64(order), "interval": b64(interval.astype(np.int16)),
        # progresso (voltas) de cada carro a cada 0,5 s: dá o gap entre quaisquer dois carros no player
        "progress": b64(P.astype(np.float32)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{year}_{slug(event)}.json"
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return path


def lead_changes(leader: np.ndarray, g: np.ndarray, mask: np.ndarray, hold: float) -> int:
    """Trocas de líder, sem contar lideranças que duram menos de hold segundos (carros lado a lado
    trocam de ordem várias vezes num trecho, e a projeção tem alguns metros de erro)."""
    idx = np.flatnonzero(mask)
    reigns = []                                    # (início, líder)
    for j in idx:
        if not reigns or leader[j] != reigns[-1][1]:
            reigns.append((g[j], leader[j]))
    kept = []
    for k, (t, who) in enumerate(reigns):
        end = reigns[k + 1][0] if k + 1 < len(reigns) else g[idx[-1]]
        if (end - t >= hold or k == 0) and (not kept or who != kept[-1]):
            kept.append(who)
    return len(kept) - 1


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("event", help='pasta do GP no TracingInsights, ex.: "Las Vegas Grand Prix"')
    args = ap.parse_args()
    path = build(args.year, args.event)
    out = json.loads(path.read_text(encoding="utf-8"))
    st, m = out["stats"], out["meta"]
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB): {len(out['drivers'])} pilotos, {m['n']} instantes, "
          f"volta de referência {m['lap_ref']:.1f} s")
    print(f"trocas de líder (lideranças de {LEAD_HOLD:.0f} s ou mais): {st['lead_changes']}; na linha, volta a volta: "
          f"{st['lead_changes_at_line']}. "
          f"P1 a menos de {DUEL_GAP:.0f} s em {st['p1_fight_share']:.0%} do tempo com bandeira verde")


if __name__ == "__main__":
    main()
