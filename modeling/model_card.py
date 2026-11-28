"""Ficha de um modelo recorrente para o painel (aba LSTM): camadas, parâmetros, pesos por portão e embeddings.

O painel não abre o model.pt (a demo roda no navegador, sem PyTorch). Este script lê os pesos de cada
membro do ensemble e grava runs/<run>/model_card.json com:

- as camadas, na ordem do cálculo, com entrada, saída e número de parâmetros (conferido com o total);
- os portões do encoder e do decoder: o peso médio absoluto de cada entrada em cada portão (a média
  sobre as unidades do estado e os membros), o peso recorrente médio e o viés de cada portão. As
  entradas numéricas chegam padronizadas, então a ordem de grandeza dos pesos é comparável entre elas;
  as booleanas chegam como 0/1;
- os embeddings (circuito, piloto, equipe-ano, regulamento) de um membro, projetados em 2 dimensões
  por PCA. Cada semente aprende o seu próprio sistema de eixos, por isso não dá para tirar a média.

Uso, a partir da pasta Neural_Stints:
    .venv/Scripts/python -m modeling.model_card runs/lstm_20261120_ens5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

GATES = {"lstm": ["entrada (i)", "esquecimento (f)", "candidato (g)", "saída (o)"],   # ordem do PyTorch
         "gru": ["reset (r)", "atualização (z)", "candidato (n)"]}
CAT_PT = {"circuit": "circuito", "driver": "piloto", "team_year": "equipe-ano", "regulation": "regulamento"}


def gate_summary(sds: list[dict], prefix: str, inputs: list[str], cell: str, hidden: int) -> dict:
    """Peso médio absoluto de cada entrada em cada portão, peso recorrente e viés por portão."""
    ng = len(GATES[cell])
    w_ih = np.stack([sd[f"{prefix}.weight_ih_l0"].numpy().reshape(ng, hidden, -1) for sd in sds])   # (M, G, H, in)
    w_hh = np.stack([sd[f"{prefix}.weight_hh_l0"].numpy().reshape(ng, hidden, hidden) for sd in sds])
    bias = np.stack([(sd[f"{prefix}.bias_ih_l0"] + sd[f"{prefix}.bias_hh_l0"]).numpy().reshape(ng, hidden) for sd in sds])
    return {"gates": GATES[cell], "inputs": inputs,
            "input_weight": np.abs(w_ih).mean(axis=(0, 2)).T.round(4).tolist(),            # (in, G)
            "recurrent_weight": np.abs(w_hh).mean(axis=(0, 2, 3)).round(4).tolist(),       # (G,)
            "bias_mean": bias.mean(axis=(0, 2)).round(4).tolist(),
            "bias_std": bias.std(axis=(0, 2)).round(4).tolist()}


def pca2(x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    xc = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = (s ** 2) / max((s ** 2).sum(), 1e-12)
    return xc @ vt[:2].T, var[:2].round(3).tolist()


def build(run_dir: Path) -> Path:
    ck = torch.load(run_dir / "model.pt", weights_only=False)
    meta = ck["dataset_meta"]
    members = ck.get("members", [ck])
    sds = [m["state_dict"] for m in members]
    args = members[0]["model_args"]
    cell, hidden, context = args["cell"], args["hidden"], args["context"]
    sd = sds[0]
    n = lambda *keys: int(sum(sd[k].numel() for k in keys if k in sd))
    ng = len(GATES[cell])
    cats = meta["static_cat"]
    enc_in = args["n_obs"] + args["n_known"] + context
    dec_in = args["n_known"] + context
    layers = [{"camada": f"Embedding de {CAT_PT.get(c, c)}", "tipo": "Embedding", "entrada": f"{v} categorias",
               "saida": f"{d} dimensões", "parametros": n(f"embs.{i}.weight")}
              for i, (c, v, d) in enumerate(zip(cats, args["vocab_sizes"], args["emb_dims"]))]
    layers += [
        {"camada": "Contexto estático", "tipo": "Linear + tanh",
         "entrada": f"{sum(args['emb_dims'])} dos embeddings + {args['n_static_x']} numéricas", "saida": f"{context}",
         "parametros": n("context.0.weight", "context.0.bias")},
        {"camada": "Estado inicial h₀" + (" e c₀" if cell == "lstm" else ""), "tipo": "Linear (tanh no h₀)",
         "entrada": f"contexto ({context})", "saida": f"{hidden}" + (" + " + str(hidden) if cell == "lstm" else ""),
         "parametros": n("h0.weight", "h0.bias", "c0.weight", "c0.bias")},
        {"camada": f"Encoder {cell.upper()}", "tipo": f"{cell.upper()} ({ng} blocos de pesos × {hidden} unidades)",
         "entrada": f"{args['n_obs']} observadas + {args['n_known']} conhecidas + {context} do contexto = {enc_in}",
         "saida": f"estado {'(h, c)' if cell == 'lstm' else 'h'} de {hidden} ao fim de cada volta",
         "parametros": n("encoder.weight_ih_l0", "encoder.weight_hh_l0", "encoder.bias_ih_l0", "encoder.bias_hh_l0")},
        {"camada": f"Decoder {cell.upper()}", "tipo": f"{cell.upper()} ({ng} blocos de pesos × {hidden} unidades)",
         "entrada": f"{args['n_known']} conhecidas das voltas futuras + {context} do contexto = {dec_in}",
         "saida": f"{hidden} por volta prevista (10 voltas)",
         "parametros": n("decoder.weight_ih_l0", "decoder.weight_hh_l0", "decoder.bias_ih_l0", "decoder.bias_hh_l0")},
        {"camada": "Dropout", "tipo": f"p = {args['dropout']}", "entrada": "estado inicial do decoder e saída dele",
         "saida": "igual", "parametros": 0},
        {"camada": "Saída", "tipo": "Linear + softplus", "entrada": f"{hidden}",
         "saida": "p10 ≤ p50 ≤ p90 do ritmo relativo", "parametros": n("head.weight", "head.bias")},
    ]
    total = int(sum(p.numel() for p in sd.values()))
    assert sum(l["parametros"] for l in layers) == total, "as camadas não somam o total de parâmetros"

    ctx_names = [f"contexto {i + 1}" for i in range(context)]
    enc = gate_summary(sds, "encoder", meta["observed"] + meta["known"] + ctx_names, cell, hidden)
    dec = gate_summary(sds, "decoder", meta["known"] + ctx_names, cell, hidden)

    embeddings = {}
    for i, c in enumerate(cats):
        w = sd[f"embs.{i}.weight"].numpy()
        names = meta["vocabs"][c]
        xy, var = pca2(w) if w.shape[1] > 2 else (w, [None, None])
        embeddings[c] = {"names": names, "x": np.round(xy[:, 0], 4).tolist(), "y": np.round(xy[:, 1], 4).tolist(),
                         "explained": var, "dims": int(w.shape[1]), "pca": w.shape[1] > 2}

    out = {"run": run_dir.name, "cell": cell, "hidden": hidden, "context": context, "gates_order": GATES[cell],
           "members": len(sds), "params_member": total, "params_total": total * len(sds), "layers": layers,
           "features": {"observed": meta["observed"], "known": meta["known"], "static_cat": cats,
                        "static_x": meta["static_x"]},
           "horizon": meta["horizon"], "min_context": meta["min_context"],
           "target_normalization": members[0]["config"].get("target_normalization"),
           "encoder": enc, "decoder": dec, "embeddings": embeddings, "embedding_member": 1}
    path = run_dir / "model_card.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="pasta do run em runs/, com model.pt")
    path = build(ap.parse_args().run)
    card = json.loads(path.read_text(encoding="utf-8"))
    print(f"{path}: {card['members']} membros × {card['params_member']:,} parâmetros".replace(",", "."))
    for l in card["layers"]:
        print(f"  {l['camada']:28} {l['parametros']:>7,}".replace(",", "."))
    enc = card["encoder"]
    print("  viés médio por portão do encoder:", dict(zip(enc["gates"], enc["bias_mean"])))


if __name__ == "__main__":
    main()
