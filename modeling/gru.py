"""GRU encoder–decoder com saída quantílica (v0).

- Contexto estático: embeddings de circuito, piloto, equipe-ano e regulamento, mais as estáticas
  numéricas, passam por uma camada densa. O contexto inicializa o estado do encoder e entra no
  decoder a cada passo (seção 10.2.4 de Goodfellow et al.).
- Encoder: GRU sobre as voltas da corrida (variáveis observadas + conhecidas + contexto). O estado
  ao fim da volta t é o ponto de partida do decoder para a previsão feita naquela volta.
- Decoder: GRU sobre as voltas t+1..t+H, recebendo só as variáveis conhecidas no futuro (pneu,
  composto, paradas, progresso) e o contexto. Nunca recebe o ritmo previsto ou real.
- Saída: p10, p50 e p90 do ritmo relativo; p10 ≤ p50 ≤ p90 por construção. Perda pinball nas
  voltas limpas (máscara `clean`).

O encoder lê cada corrida uma vez e todas as previsões da sequência partem dos seus estados, então
cada época processa as ~1.800 sequências de treino, não as ~94 mil janelas.

Uso, a partir da pasta Neural_Stints:  .venv/Scripts/python -m modeling.gru [--hidden 64] [--seed 1]
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from modeling.frame import target_frame
from modeling.results import (RUNS, compute_metrics, finish_predictions, git_commit, now_utc, print_summary,
                              write_atomic, write_run)

QUANTILES = (0.1, 0.5, 0.9)
EMB_DIMS = {"circuit": 4, "driver": 4, "team_year": 4, "regulation": 2}


class Seq2SeqGRU(nn.Module):
    def __init__(self, n_obs, n_known, vocab_sizes, emb_dims, n_static_x, hidden=64, context=16, dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(v, e) for v, e in zip(vocab_sizes, emb_dims))
        self.context = nn.Sequential(nn.Linear(sum(emb_dims) + n_static_x, context), nn.Tanh())
        self.h0 = nn.Linear(context, hidden)
        self.encoder = nn.GRU(n_obs + n_known + context, hidden, batch_first=True)
        self.decoder = nn.GRU(n_known + context, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 3)

    def forward(self, enc_x, known, static_cat, static_x, horizon):
        """enc_x (B, L, obs+known), known (B, L+H, known) -> quantis (B, L-1, H, 3).

        A saída [:, t-1, h-1] é a previsão feita ao fim da volta t para a volta t+h.
        """
        B, L, _ = enc_x.shape
        emb = [e(static_cat[:, i]) for i, e in enumerate(self.embs)]
        ctx = self.context(torch.cat(emb + [static_x], dim=1))                     # (B, C)
        enc_in = torch.cat([enc_x, ctx[:, None].expand(B, L, -1)], dim=2)
        h_all, _ = self.encoder(enc_in, torch.tanh(self.h0(ctx))[None])            # (B, L, hidden)

        T = L - 1                                                                   # âncoras t = 1..L-1
        idx = torch.arange(1, L, device=enc_x.device)[:, None] + torch.arange(horizon, device=enc_x.device)
        dec_in = known[:, idx]                                                      # voltas t+1..t+H
        dec_in = torch.cat([dec_in, ctx[:, None, None].expand(B, T, horizon, -1)], dim=3)
        h_start = self.drop(h_all[:, :T]).reshape(1, B * T, -1)
        out, _ = self.decoder(dec_in.reshape(B * T, horizon, -1), h_start.contiguous())
        return self.quantiles(self.head(self.drop(out)).reshape(B, T, horizon, 3))

    @staticmethod
    def quantiles(o):
        p50 = o[..., 0]
        return torch.stack([p50 - nn.functional.softplus(o[..., 1]), p50,
                            p50 + nn.functional.softplus(o[..., 2])], dim=-1)

    def forecast(self, enc_x, future_known, static_cat, static_x):
        """Previsão ao vivo: a partir do fim da última volta de enc_x, para as voltas de future_known.

        enc_x (B, t, obs+known), future_known (B, H, known) -> quantis (B, H, 3). É o mesmo cálculo
        de forward para a âncora t, sem precisar das voltas futuras reais.
        """
        B, L, _ = enc_x.shape
        H = future_known.shape[1]
        emb = [e(static_cat[:, i]) for i, e in enumerate(self.embs)]
        ctx = self.context(torch.cat(emb + [static_x], dim=1))
        enc_in = torch.cat([enc_x, ctx[:, None].expand(B, L, -1)], dim=2)
        _, h_last = self.encoder(enc_in, torch.tanh(self.h0(ctx))[None])
        dec_in = torch.cat([future_known, ctx[:, None].expand(B, H, -1)], dim=2)
        out, _ = self.decoder(dec_in, self.drop(h_last))
        return self.quantiles(self.head(self.drop(out)))


def pinball(pred, y, mask):
    """Perda quantílica média nos três quantis, só onde mask é verdadeiro."""
    q = torch.tensor(QUANTILES, device=pred.device)
    err = y[..., None] - pred
    loss = torch.maximum(q * err, (q - 1) * err).mean(-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


class Batches:
    """Monta lotes de sequências com padding no fim (o encoder é causal, então o padding não vaza)."""

    def __init__(self, d, y_mean, y_std, horizon, min_context):
        self.d, self.H, self.min_context = d, horizon, min_context
        self.y_z = ((d["y"] - y_mean) / y_std).astype(np.float32)
        self.n = np.diff(d["offsets"])

    def make(self, seqs, cat_dropout=0.0, rng=None):
        d, H = self.d, self.H
        L = max(int(self.n[seqs].max()), 2)
        B = len(seqs)
        n_obs, n_known = d["x_obs"].shape[1], d["x_known"].shape[1]
        enc = np.zeros((B, L, n_obs + n_known), np.float32)
        known = np.zeros((B, L + H, n_known), np.float32)
        y = np.zeros((B, L + H), np.float32)
        ok = np.zeros((B, L + H), bool)
        ref = np.zeros((B, L + H), np.float32)
        for i, s in enumerate(seqs):
            lo, hi = d["offsets"][s], d["offsets"][s + 1]
            n = hi - lo
            enc[i, :n, :n_obs], enc[i, :n, n_obs:] = d["x_obs"][lo:hi], d["x_known"][lo:hi]
            known[i, :n] = d["x_known"][lo:hi]
            y[i, :n], ok[i, :n], ref[i, :n] = self.y_z[lo:hi], d["y_mask"][lo:hi], d["ref_lap"][lo:hi]
        cat = d["static_cat"][seqs].copy()
        if cat_dropout and rng is not None:          # ensina o embedding <unk> (circuitos e pilotos novos)
            cat[rng.random(cat.shape) < cat_dropout] = 0
        # alvo e máscara alinhados com a saída do modelo: [:, t-1, h-1] -> volta t+h
        idx = np.arange(1, L)[:, None] + np.arange(H)[None, :]
        t = np.arange(1, L)
        anchor_ok = (t[None, :] >= self.min_context) & (t[None, :] < self.n[seqs][:, None])
        mask = ok[:, idx] & anchor_ok[:, :, None]
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return {"enc": to(enc), "known": to(known), "cat": to(cat), "sx": to(d["static_x"][seqs]),
                "y": to(y[:, idx]), "mask": to(mask.astype(np.float32)), "ref": to(ref[:, idx])}


def evaluate(model, batches, seqs, H, y_mean, y_std, device, batch=64):
    """Perda pinball (escala z) e MAE em segundos da próxima volta, num conjunto de sequências."""
    model.eval()
    loss_sum = w_sum = ae_sum = n1 = 0.0
    with torch.no_grad():
        for b in range(0, len(seqs), batch):
            bt = {k: v.to(device) for k, v in batches.make(seqs[b:b + batch]).items()}
            pred = model(bt["enc"], bt["known"], bt["cat"], bt["sx"], H)
            w = bt["mask"].sum().item()
            loss_sum += pinball(pred, bt["y"], bt["mask"]).item() * w
            w_sum += w
            m1 = bt["mask"][:, :, 0]
            ae = ((pred[:, :, 0, 1] - bt["y"][:, :, 0]).abs() * y_std * bt["ref"][:, :, 0] * m1).sum().item()
            ae_sum += ae
            n1 += m1.sum().item()
    return loss_sum / w_sum, ae_sum / n1


def predict(model, batches, seqs, H, y_mean, y_std, device, batch=64):
    """Quantis do ritmo por sequência: {s: array (n_s - 1, H, 3)} na escala original."""
    model.eval()
    out = {}
    with torch.no_grad():
        for b in range(0, len(seqs), batch):
            chunk = seqs[b:b + batch]
            bt = {k: v.to(device) for k, v in batches.make(chunk).items()}
            pred = model(bt["enc"], bt["known"], bt["cat"], bt["sx"], H).cpu().numpy() * y_std + y_mean
            for i, s in enumerate(chunk):
                out[int(s)] = pred[i, : batches.n[s] - 1]
    return out


def latest_run(prefix: str) -> Path | None:
    runs = sorted(RUNS.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--cat-dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dataset", default="dataset", help="dataset em data/processed (ex.: dataset_split5)")
    ap.add_argument("--out", type=Path, default=RUNS, help="pasta onde o run é gravado")
    ap.add_argument("--run-id", help="padrão: gru_<data>_s<semente>")
    ap.add_argument("--reference", type=Path, help="pasta do run do ingênuo para o skill (padrão: o mais recente)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = args.run_id or f"gru_{date.today():%Y%m%d}_s{args.seed}"

    frame, laps, d, meta = target_frame(("val", "test"), args.dataset)
    H = meta["horizon"]
    train_rows = np.repeat(d["seq_split"] == "train", np.diff(d["offsets"])) & d["y_mask"]
    y_mean, y_std = float(d["y"][train_rows].mean()), float(d["y"][train_rows].std())
    batches = Batches(d, y_mean, y_std, H, meta["min_context"])
    seqs = {s: np.where(d["seq_split"] == s)[0] for s in ("train", "val", "test")}

    vocab_sizes = [len(meta["vocabs"][c]) for c in meta["static_cat"]]
    emb_dims = [EMB_DIMS[c] for c in meta["static_cat"]]
    model = Seq2SeqGRU(d["x_obs"].shape[1], d["x_known"].shape[1], vocab_sizes, emb_dims,
                       d["static_x"].shape[1], args.hidden, args.context, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"{run_id}: {n_params:,} parâmetros, {len(seqs['train'])} sequências de treino, dispositivo {device}")

    history, best, best_state, bad, t_start = [], np.inf, None, 0, time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = rng.permutation(seqs["train"])
        loss_sum = w_sum = 0.0
        for b in range(0, len(perm), args.batch):
            bt = {k: v.to(device) for k, v in batches.make(perm[b:b + args.batch], args.cat_dropout, rng).items()}
            pred = model(bt["enc"], bt["known"], bt["cat"], bt["sx"], H)
            loss = pinball(pred, bt["y"], bt["mask"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            w = bt["mask"].sum().item()
            loss_sum += loss.item() * w
            w_sum += w
        val_loss, val_mae_h1 = evaluate(model, batches, seqs["val"], H, y_mean, y_std, device)
        is_best = val_loss < best
        if is_best:
            best, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        history.append({"epoch": epoch, "train_loss": loss_sum / w_sum, "val_loss": val_loss, "val_mae_h1": val_mae_h1,
                        "lr": opt.param_groups[0]["lr"], "seconds": time.perf_counter() - t0, "best": is_best})
        print(f"  época {epoch:2}: treino {loss_sum / w_sum:.4f} · validação {val_loss:.4f} · "
              f"MAE h1 {val_mae_h1:.3f} s{' *' if is_best else ''}")
        if bad >= args.patience:
            break
    train_seconds = time.perf_counter() - t_start
    model.load_state_dict(best_state)
    best_epoch = int(max(h["epoch"] for h in history if h["best"]))

    # Previsões na validação e no teste, no formato padrão
    preds = predict(model, batches, np.concatenate([seqs["val"], seqs["test"]]), H, y_mean, y_std, device)
    seq_of = np.searchsorted(d["offsets"], frame["anchor"].to_numpy(), side="right") - 1
    q = np.stack([preds[s][t - 1, h - 1] for s, t, h in zip(seq_of, frame["t"], frame["h"])])
    frame["pace_p10"], frame["pace_p50"], frame["pace_p90"] = q[:, 0], q[:, 1], q[:, 2]
    pred = finish_predictions(frame, run_id, "gru")

    naive_dir = args.reference or latest_run("naive")
    reference = pd.read_parquet(naive_dir / "predictions.parquet") if naive_dir else None
    metrics = compute_metrics(pred, reference)
    if naive_dir:
        metrics["reference_run"] = naive_dir.name

    config = {
        "run_id": run_id, "model": "gru", "seed": args.seed,
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("seed", "dataset", "out", "run_id", "reference")}
                           | {"emb_dims": EMB_DIMS},
        "dataset": args.dataset,
        "eras_strategy": "pooled", "features": {k: meta[k] for k in ("observed", "known", "static_cat", "static_x")},
        "horizon": H, "quantiles": list(QUANTILES), "data": meta["splits"],
        "target_normalization": {"mean": y_mean, "std": y_std},
        "n_params": n_params, "train_seconds": round(train_seconds, 1), "best_epoch": best_epoch,
        "device": str(device), "git_commit": git_commit(), "created_at": now_utc(),
    }
    out = write_run(run_id, config, pred, metrics, pd.DataFrame(history), root=args.out)
    checkpoint = {"state_dict": model.state_dict(), "config": config, "dataset_meta": meta,
                  "model_args": {"n_obs": d["x_obs"].shape[1], "n_known": d["x_known"].shape[1],
                                 "vocab_sizes": vocab_sizes, "emb_dims": emb_dims,
                                 "n_static_x": d["static_x"].shape[1], "hidden": args.hidden,
                                 "context": args.context, "dropout": args.dropout}}
    write_atomic(out / "model.pt", lambda p: torch.save(checkpoint, p))
    print(f"melhor época {best_epoch}, treino em {train_seconds:.0f} s -> {out}")
    print_summary(metrics)


if __name__ == "__main__":
    main()
