"""Can prompt calibration close replay05's residual TP gap?

replay05 tracks the target almost perfectly (slope 0.973) but sits ~53 K low. An affine
distortion with an intact slope is invertible by prompting, and unlike composite
(slope 0.268) this model has a slope worth inverting.

The map is fitted on targets 280/350/420/470 and TESTED on the protocol's 300/400/500,
so it is not fitted to its own scoreboard. Round 1's attempt on composite failed partly
because asking for higher Tg cost validity; PV is reported here for the same reason.
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
    tokenizer_path="artifacts/tokenizer/polyt5_vocab.json",
    device=str(device),
)

payload = load_checkpoint(
    "results/grpo_composite_replay05/checkpoints/step_002000.pt", map_location="cpu"
)
model = PolyT5ForConditionalGeneration(PolyT5Config.from_dict(payload["model_config"]))
model.load_state_dict(payload["model_state"])
model = model.to(device).eval()


def sample(prompt_value, n, seed=0):
    enc = tok.batch_encode(
        [format_property_value(prompt_value)] * n,
        add_eos=True,
        max_length=200,
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        out = generate(
            model,
            torch.tensor(enc["input_ids"], device=device),
            torch.tensor(enc["attention_mask"], device=device),
            config=GenerationConfig(
                max_length=200,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                seed=seed,
                eos_token_id=tok.eos_id,
                pad_token_id=tok.pad_id,
                decoder_start_token_id=tok.decoder_start_token_id,
            ),
        )
    return tok.batch_decode(out.sequences.tolist(), skip_special_tokens=True)


CAL = [280.0, 350.0, 420.0, 470.0]
xs, ys = [], []
for t in CAL:
    values = [v for v in auditor(sample(t, 120, seed=7)) if v == v]
    xs.append(t)
    ys.append(st.mean(values))
mx, my = st.mean(xs), st.mean(ys)
cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
var = sum((x - mx) ** 2 for x in xs)
slope = cov / var
intercept = my - slope * mx
print(f"fit on {CAL}:  Tg = {slope:.4f} * target + {intercept:.1f}")
print(f"inverse: to obtain T, prompt (T - {intercept:.1f}) / {slope:.4f}\n")

print(f"{'target':>7} {'prompt':>8} {'mean Tg':>9} {'TP':>7} {'PV':>7}  mode")
print("-" * 50)
tp_raw = tp_cal = n_raw = n_cal = 0
pv_raw_h = pv_raw_n = pv_cal_h = pv_cal_n = 0
for target in (300.0, 400.0, 500.0):
    for mode in ("raw", "calibrated"):
        prompt = target if mode == "raw" else max(150.0, min(900.0, (target - intercept) / slope))
        texts = sample(prompt, 200, seed=0)
        values = [v for v in auditor(texts) if v == v]
        hits = sum(1 for v in values if abs(v - target) <= 50.0)
        report = evaluate_generation(
            texts, target_property=target, tolerance=50.0, property_predictor=auditor
        )
        if mode == "raw":
            tp_raw += hits
            n_raw += len(values)
            pv_raw_h += report.counts.n_pv
            pv_raw_n += report.counts.n_input
        else:
            tp_cal += hits
            n_cal += len(values)
            pv_cal_h += report.counts.n_pv
            pv_cal_n += report.counts.n_input
        print(
            f"{target:>7.0f} {prompt:>8.1f} {st.mean(values):>9.1f} "
            f"{hits / len(values):>7.3f} "
            f"{report.counts.n_pv / report.counts.n_input:>7.3f}  {mode}"
        )

print()
print(f"pooled TP   raw {tp_raw / n_raw:.3f}   calibrated {tp_cal / n_cal:.3f}   baseline 0.738")
print(
    f"pooled PV   raw {pv_raw_h / pv_raw_n:.3f}   calibrated {pv_cal_h / pv_cal_n:.3f}"
    f"   baseline 0.665"
)
