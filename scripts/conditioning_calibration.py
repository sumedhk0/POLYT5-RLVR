"""If the damage is an OFFSET and the slope survives, calibrate the prompt.

step 400 responds to the target with slope 0.914 (baseline 0.896) but sits ~51 K low.
That is an affine distortion, and an affine distortion is invertible: to obtain Tg T,
ask for the target that the distorted map sends to T. No retraining, no reward change.
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import torch

from polyt5.data.prepare import format_property_value
from polyt5.evaluation import evaluate_generation
from polyt5.generation import GenerationConfig, generate
from polyt5.inference import PolyT5PropertyPredictor
from polyt5.model import PolyT5Config, PolyT5ForConditionalGeneration
from polyt5.tokenization import PolyT5Tokenizer
from polyt5.training import load_checkpoint
from polyt5.utils import select_device

device = select_device("auto")
tok = PolyT5Tokenizer.from_file("artifacts/tokenizer/polyt5_vocab.json")
frozen = json.loads(Path("artifacts/baseline/frozen_baseline.json").read_text())
auditor = PolyT5PropertyPredictor.from_checkpoint(
    frozen["artifacts"]["tg_predictor_split4"]["path"].replace("\\", "/"),
    tokenizer_path="artifacts/tokenizer/polyt5_vocab.json", device=str(device))

def load(path):
    p = load_checkpoint(path, map_location="cpu")
    m = PolyT5ForConditionalGeneration(PolyT5Config.from_dict(p["model_config"]))
    m.load_state_dict(p["model_state"])
    return m.to(device).eval()

def sample(model, prompt_value, n, seed=0):
    enc = tok.batch_encode([format_property_value(prompt_value)] * n, add_eos=True,
                           max_length=200, padding=True, truncation=True)
    with torch.no_grad():
        out = generate(model, torch.tensor(enc["input_ids"], device=device),
                       torch.tensor(enc["attention_mask"], device=device),
                       config=GenerationConfig(max_length=200, temperature=0.7, top_p=0.95,
                                               do_sample=True, seed=seed, eos_token_id=tok.eos_id,
                                               pad_token_id=tok.pad_id,
                                               decoder_start_token_id=tok.decoder_start_token_id))
    return tok.batch_decode(out.sequences.tolist(), skip_special_tokens=True)

CKPT = "results/grpo_composite/checkpoints/step_000400.pt"
model = load(CKPT)

# Fit the affine map on a CALIBRATION set of targets, then test on the frozen protocol's
# 300/400/500. Fitting and testing on the same points would be circular.
CAL = [280.0, 350.0, 420.0, 470.0]
xs, ys = [], []
for t in CAL:
    vals = [v for v in auditor(sample(model, t, 120, seed=7)) if v == v]
    xs.append(t)
    ys.append(st.mean(vals))
n = len(xs)
mx, my = st.mean(xs), st.mean(ys)
cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
var = sum((x - mx) ** 2 for x in xs)
slope = cov / var
intercept = my - slope * mx
print(f"calibration fit on {CAL}: Tg = {slope:.4f} * target + {intercept:.1f}")
print(f"  inverse: to obtain T, prompt (T - {intercept:.1f}) / {slope:.4f}\n")

print(f"{'target':>7} {'prompt':>8} {'mean Tg':>9} {'TP±50':>7} {'PV':>7}")
print("-" * 42)
tp_raw = tp_cal = n_raw = n_cal = 0
pv_h = pv_n = 0
for target in (300.0, 400.0, 500.0):
    for mode in ("raw", "calibrated"):
        prompt = target if mode == "raw" else max(150.0, min(900.0, (target - intercept) / slope))
        texts = sample(model, prompt, 200, seed=0)
        vals = [v for v in auditor(texts) if v == v]
        hits = sum(1 for v in vals if abs(v - target) <= 50.0)
        rep = evaluate_generation(texts, target_property=target, tolerance=50.0,
                                  property_predictor=auditor)
        if mode == "raw":
            tp_raw += hits
            n_raw += len(vals)
        else:
            tp_cal += hits
            n_cal += len(vals)
            pv_h += rep.counts.n_pv
            pv_n += rep.counts.n_input
        print(f"{target:>7.0f} {prompt:>8.1f} {st.mean(vals):>9.1f} {hits/len(vals):>7.3f} "
              f"{rep.counts.n_pv/rep.counts.n_input:>7.3f}  {mode}")
print()
print(f"pooled TP  raw {tp_raw/n_raw:.3f}   calibrated {tp_cal/n_cal:.3f}   baseline 0.740")
print(f"pooled PV  calibrated {pv_h/pv_n:.3f}   baseline 0.658")
