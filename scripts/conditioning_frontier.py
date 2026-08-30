"""Trade-off between composite's structural gain and its conditioning damage.

compare_arms scores only the FINAL checkpoint, so it cannot see whether an earlier
one keeps both. KL grew monotonically 0.003 -> 0.073 while most of the gate
improvement arrived by step 400, which is the shape of a usable operating point.
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

def sample(model, target, n):
    enc = tok.batch_encode([format_property_value(target)] * n, add_eos=True,
                           max_length=200, padding=True, truncation=True)
    with torch.no_grad():
        out = generate(model, torch.tensor(enc["input_ids"], device=device),
                       torch.tensor(enc["attention_mask"], device=device),
                       config=GenerationConfig(max_length=200, temperature=0.7, top_p=0.95,
                                               do_sample=True, seed=0, eos_token_id=tok.eos_id,
                                               pad_token_id=tok.pad_id,
                                               decoder_start_token_id=tok.decoder_start_token_id))
    return tok.batch_decode(out.sequences.tolist(), skip_special_tokens=True)

N = 150
POINTS = [("baseline", frozen["artifacts"]["generation"]["path"].replace("\\", "/"))]
POINTS += [(f"step {s}", f"results/grpo_composite/checkpoints/step_{s:06d}.pt")
           for s in (200, 400, 800, 1200, 2000)]

print(f"{'checkpoint':<12} {'PV':>7} {'slope':>7} {'Tg@400':>8} {'TP±50':>7}")
print("-" * 46)
for name, path in POINTS:
    model = load(path)
    means, tp_hits, tp_n, pv_hits, pv_n = [], 0, 0, 0, 0
    for target in (300.0, 400.0, 500.0):
        texts = sample(model, target, N)
        rep = evaluate_generation(texts, target_property=target, tolerance=50.0,
                                  property_predictor=auditor)
        pv_hits += rep.counts.n_pv
        pv_n += rep.counts.n_input
        vals = [v for v in auditor(texts) if v == v]
        means.append(st.mean(vals) if vals else float("nan"))
        tp_hits += sum(1 for v in vals if abs(v - target) <= 50.0)
        tp_n += len(vals)
    slope = (means[2] - means[0]) / 200.0
    print(f"{name:<12} {pv_hits/pv_n:>7.3f} {slope:>7.3f} {means[1]:>8.1f} {tp_hits/tp_n:>7.3f}")
    del model
    torch.cuda.empty_cache()
