"""Measure how much the 4-model reward ensemble flatters itself.

Phase 3 instrument audit. NOT part of the published polyT5 method.

The paper's protocol is five INDEPENDENT random 80/20 splits, so each model's
own test set is genuinely unseen data and the reported per-split MAE is honest
*for a single model*. Our Phase 3 extension then reused splits 0-3 as an
ENSEMBLE, and that is where honesty breaks: for a given polymer, typically only
one of the four never trained on it, so the ensemble mean is mostly a memory
check.

This script quantifies the gap. For every polymer with at least one clean
member, it computes two predictions:

    contaminated  mean over ALL four reward models (what the reward actually uses)
    honest        mean over ONLY the members that never trained on that polymer

and compares both against experimental Tg. It reports the same contrast for the
disagreement sigma, because sigma feeds the reward's confidence weight -- a
sigma computed largely from models that memorised the answer understates real
uncertainty, which is the failure mode that matters for reward hacking.

Nothing here trains or writes a checkpoint. It is read-only over existing
artifacts, and defaults to CPU so it can run beside a training job.

Usage::

    python scripts/audit_ensemble_leakage.py
    python scripts/audit_ensemble_leakage.py --device cuda --out results/leakage.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from polyt5.data.prepare import prepare_labeled_corpus, read_lamalab_tg  # noqa: E402
from polyt5.inference.predictor import PolyT5PropertyPredictor  # noqa: E402
from polyt5.tokenization import PolyT5Tokenizer  # noqa: E402

logger = logging.getLogger("polyt5.audit_leakage")

#: Splits 0-3 are the reward ensemble. Split 4 is the held-out auditor and is
#: deliberately absent -- it must never be scored as part of a reward path.
REWARD_MEMBERS = (0, 1, 2, 3)

DEFAULT_SPLIT_ROOT = REPO / "results" / "tg_prediction_5splits_medium92m"
DEFAULT_CSV = REPO / "data" / "external" / "LAMALAB_CURATED_Tg.csv"
DEFAULT_TOKENIZER = REPO / "artifacts" / "tokenizer" / "polyt5_vocab.json"


def _mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson r, or None when either side is constant.

    A constant sigma (every group of size one, so every spread is exactly 0)
    makes the correlation undefined rather than zero. Returning None keeps that
    distinction visible instead of reporting a spurious 0.0.
    """
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", default="cpu",
                        help="cpu by default so this can run beside a training job")
    parser.add_argument("--num-beams", type=int, default=4, help="[PAPER] 4")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "ensemble_leakage.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    splits_payload = json.loads((args.split_root / "splits.json").read_text(encoding="utf-8"))
    splits = splits_payload["splits"]

    tokenizer = PolyT5Tokenizer.from_file(args.tokenizer)
    # Prepared exactly as scripts/run_splits.py does, so index i here is index i
    # in splits.json. The size assertion below is what actually guarantees that.
    pairs, _stats = prepare_labeled_corpus(
        read_lamalab_tg(args.csv), max_tokens=200, deduplicate=True, tokenizer=tokenizer
    )
    n = len(pairs)
    if n != splits_payload["n"]:
        raise SystemExit(
            f"corpus size {n} != splits.json n {splits_payload['n']}; indices would not align"
        )

    # A model is CLEAN for a polymer when that polymer was in neither its train
    # nor its val portion. val counts as seen: it drove checkpoint selection.
    seen = [set(s["train"]) | set(s["val"]) for s in splits]
    clean_mask = np.zeros((len(splits), n), dtype=bool)
    for m, seen_m in enumerate(seen):
        for i in range(n):
            clean_mask[m, i] = i not in seen_m

    truth = np.array([value for _pselfies, value in pairs], dtype=float)
    inputs = [pselfies for pselfies, _value in pairs]

    preds = np.full((len(splits), n), np.nan, dtype=float)
    for m in REWARD_MEMBERS:
        ckpt = args.split_root / f"split_{m}" / "checkpoints" / "best.pt"
        logger.info("scoring with split %d (%s)", m, ckpt.name)
        predictor = PolyT5PropertyPredictor.from_checkpoint(
            ckpt, args.tokenizer, device=args.device,
            num_beams=args.num_beams, batch_size=args.batch_size,
        )
        values = predictor.predict_values(inputs)
        preds[m] = [np.nan if v is None else float(v) for v in values]
        del predictor

    member = np.array(REWARD_MEMBERS)
    p = preds[member]                      # (4, n)
    c = clean_mask[member]                 # (4, n)
    usable = ~np.isnan(p)
    n_clean = (c & usable).sum(axis=0)

    report: dict[str, object] = {
        "n_polymers": int(n),
        "reward_members": list(REWARD_MEMBERS),
        "note": "split 4 is the held-out auditor and is deliberately not scored here",
        "clean_member_histogram": {
            str(k): int((n_clean == k).sum()) for k in range(len(REWARD_MEMBERS) + 1)
        },
    }

    for threshold in (1, 2):
        rows = np.where(n_clean >= threshold)[0]
        if rows.size == 0:
            continue
        cont_pred, hon_pred, cont_sd, hon_sd = [], [], [], []
        for i in rows:
            all_vals = p[usable[:, i], i]
            cln_vals = p[c[:, i] & usable[:, i], i]
            cont_pred.append(all_vals.mean())
            hon_pred.append(cln_vals.mean())
            cont_sd.append(all_vals.std())
            hon_sd.append(cln_vals.std())
        t = truth[rows]
        cont_pred_a, hon_pred_a = np.array(cont_pred), np.array(hon_pred)
        cont_sd_a, hon_sd_a = np.array(cont_sd), np.array(hon_sd)
        cont_mae, hon_mae = _mae(cont_pred_a, t), _mae(hon_pred_a, t)

        # THE decisive test for the reward's confidence weight. That weight is
        # 1/(1+sigma/sigma0), i.e. it assumes sigma PREDICTS error. If the
        # correlation is ~0 the weight is reweighting on noise; if it is
        # NEGATIVE the weight actively penalises the predictions that are more
        # accurate, which would be worse than having no weight at all.
        cont_r = _safe_corr(cont_sd_a, np.abs(cont_pred_a - t))
        hon_r = _safe_corr(hon_sd_a, np.abs(hon_pred_a - t))

        report[f"at_least_{threshold}_clean"] = {
            "n": int(rows.size),
            "contaminated_mae": round(cont_mae, 4),
            "honest_mae": round(hon_mae, 4),
            "optimism_k": round(hon_mae - cont_mae, 4),
            "optimism_pct": round(100.0 * (hon_mae - cont_mae) / max(cont_mae, 1e-9), 2),
            "contaminated_sigma_mean": round(float(np.mean(cont_sd_a)), 4),
            "honest_sigma_mean": round(float(np.mean(hon_sd_a)), 4),
            "corr_contaminated_sigma_vs_abs_error": cont_r,
            "corr_honest_sigma_vs_abs_error": hon_r,
        }
        logger.info(
            ">=%d clean (n=%d): contaminated MAE %.2f K, honest MAE %.2f K, optimism %.2f K",
            threshold, rows.size, cont_mae, hon_mae, hon_mae - cont_mae,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
