"""Duelos: o gap entre dois carros projetado pela GRU e a janela de voltas em que a disputa deve começar.

A GRU prevê o tempo das próximas voltas de cada piloto (p10, p50 e p90). Para um par, o carro de trás
(B) e o da frente (A), o gap projetado depois de k voltas é

    gap_k = gap_0 + soma_{h<=k} (tempo_B,h − tempo_A,h).

A incerteza vem dos quantis: o desvio de uma volta é (p90 − p10) / 2,563. Dentro de um mesmo piloto os
erros das voltas seguintes andam juntos (o erro da rede é persistente), então os desvios se somam;
entre os dois pilotos, os erros são tratados como independentes. Assim gap_k ~ Normal(mu_k, sd_k) e
P(gap_k <= 1 s) sai da normal. `SIGMA_SCALE` ajusta essa largura e é calibrado na validação
(modeling.duels_backtest), para que a janela de 80% contenha de fato ~80% das disputas.

A disputa é o carro de trás chegar a menos de 1 s do da frente na linha de chegada, a zona de ataque.
A probabilidade de já ter chegado até a volta k é o máximo das probabilidades até ela, o que dá:
- volta provável: a primeira em que essa probabilidade passa de 50%;
- janela: da primeira volta em que passa de 10% à primeira em que passa de 90% (aberta se não passar).
"""
from math import erf, sqrt

import numpy as np

BATTLE_GAP = 1.0      # zona de ataque: menos de 1 s na linha de chegada
WIDTH_80 = 2.5631     # (p90 − p10) em desvios-padrão de uma normal
# Calibrado na validação (modeling.duels_backtest): somar os desvios das voltas como se os erros fossem
# totalmente persistentes deixa a janela larga demais (98% de cobertura com 1,0); com 0,5 a janela de
# 80% cobre 80% das disputas na validação, 93% no teste e 80% em São Paulo, ao vivo.
SIGMA_SCALE = 0.5
WINDOW = (0.10, 0.90)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(erf)(np.asarray(x, dtype=float) / sqrt(2)))


def project(gap0: float, behind: np.ndarray, ahead: np.ndarray, scale: float = SIGMA_SCALE) -> dict:
    """Gap projetado volta a volta. `behind` e `ahead`: matrizes (n, 3) com p10, p50 e p90 em segundos.

    Devolve mu, sd (gap médio e desvio depois de k = 1..n voltas), prob (P(gap_k <= 1 s)) e caught
    (probabilidade de já ter chegado à zona de ataque até a volta k). Voltas sem previsão de um dos dois
    encurtam o horizonte.
    """
    b, a = np.asarray(behind, dtype=float), np.asarray(ahead, dtype=float)
    n = min(len(b), len(a))
    b, a = b[:n], a[:n]
    ok = ~(np.isnan(b).any(axis=1) | np.isnan(a).any(axis=1))
    n = int(np.argmin(ok)) if not ok.all() else n             # para na primeira volta sem previsão
    b, a = b[:n], a[:n]
    mu = gap0 + np.cumsum(b[:, 1] - a[:, 1])
    sd_b = np.cumsum((b[:, 2] - b[:, 0]) / WIDTH_80)
    sd_a = np.cumsum((a[:, 2] - a[:, 0]) / WIDTH_80)
    sd = scale * np.sqrt(sd_a ** 2 + sd_b ** 2) + 1e-6
    prob = normal_cdf((BATTLE_GAP - mu) / sd)
    caught = np.maximum.accumulate(prob) if n else prob
    return {"mu": mu, "sd": sd, "prob": prob, "caught": caught}


def catch_window(caught: np.ndarray) -> dict:
    """Volta provável (50%) e janela (10%–90%) da chegada à zona de ataque, em voltas a partir de agora.

    `None` quando o limite não é atingido dentro do horizonte; `late_open` indica que a janela pode
    terminar depois do horizonte.
    """
    def first(level):
        hit = np.flatnonzero(caught >= level)
        return int(hit[0]) + 1 if len(hit) else None
    lo, hi = WINDOW
    return {"likely": first(0.5), "early": first(lo), "late": first(hi), "late_open": first(hi) is None,
            "p_horizon": float(caught[-1]) if len(caught) else 0.0}


def describe(gap0: float, proj: dict, t: int) -> str:
    """Frase curta para o painel, em número de volta (t: a última volta completada pelo carro de trás)."""
    if gap0 <= BATTLE_GAP:
        return "Já em disputa (menos de 1 s)"
    w = catch_window(proj["caught"])
    n = len(proj["caught"])
    pct = "≥99%" if w["p_horizon"] >= 0.995 else f"{w['p_horizon']:.0%}"
    chance = f"{pct} de chance nas próximas {n} voltas"
    if w["likely"] is None:
        if w["early"] is None:
            return f"Sem disputa prevista nas próximas {n} voltas ({pct} de chance)"
        return f"Disputa possível a partir da volta {t + w['early']} ({chance})"
    window = (f"janela {t + w['early']}–{t + w['late']}" if not w["late_open"]
              else f"janela da volta {t + w['early']} em diante")
    return f"Disputa prevista na volta {t + w['likely']} ({window}; {chance})"
