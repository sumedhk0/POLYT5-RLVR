# Supervised baseline: plan, status, and results

This is the Phase 1 + Phase 2 record: the supervised polyT5 reproduction, independent of any RL work. It is
the document that gets updated as runs complete. Anything here describing a *result* must cite a directory
under `results/`.

## Staged build order

Deliberately incremental — no large-scale training until the whole path is exercised end to end.

| Stage | Content | Status |
|---|---|---|
| **0** | Repo skeleton, environment, docs, data survey | ✅ done |
| **1** | Real polymers → chemistry → tokenizer → span corruption → batch → polyT5-small → forward → loss → backward → optimizer step → checkpoint → reload | ✅ done |
| **2** | Real proxy corpus (PI1M) → full train/validation pipeline | ✅ done |
| **3** | polyT5-small pretraining on the full PI1M corpus | ✅ done |
| **4** | polyT5-medium pretraining on polyOne (92.3M sequences) | ✅ done |
| **5** | Downstream Tg property prediction (beam-4, MAE/RMSE/R²; all 5 splits) | ✅ done |
| **6** | Tg-conditioned generation | ✅ done |
| **7** | Full published evaluation methodology (SV/TSD/DD/PV, SR, SA, novelty, target-property error, sampling sweep) | ✅ done |
| **8** | Freeze baseline; Arm A and Arm B measured and recorded | ✅ done |

Phase 3 (GRPO/RLVR, see [`rlvr_plan.md`](rlvr_plan.md)) begins only after Stage 8.
Stage 8 is now satisfied, and the Phase 3 apparatus is implemented and tested.
**No arm has been trained yet, so this document reports no RLVR result.**

## Hardware reality

| | polyT5 paper | This reproduction |
|---|---|---|
| GPU | 1 × NVIDIA L40S (48 GB) | 1 × NVIDIA RTX 4080 **Laptop** (12 GB) |
| Pretraining batch size | 450 | physical batch × gradient accumulation |
| Corpus | 100.1M polymers | ~1M (PI1M) by default; 100M available on request |
| Precision | unstated | AMP (bf16 where supported) |

The published batch size of 450 is **never hard-coded**. Configs specify `physical_batch_size` and
`gradient_accumulation_steps`; the effective batch size is computed, logged, and asserted. For example
`physical=8 × accum=56 = 448`, or `physical=18 × accum=25 = 450` if memory allows. Every training run
reports: parameter count, physical batch size, accumulation steps, effective batch size, sequence length,
peak VRAM, and tokens/sec.

Note the GPU is the **Laptop** variant of the 4080 (12 GB, lower power envelope than the desktop 4080),
which matters for throughput expectations.

## Reproduction fidelity ledger

What our baseline shares with the paper, and what it cannot:

| Component | Matched? | Note |
|---|---|---|
| Architecture (all three sizes) | ✅ exact | Table S2 values |
| Max sequence length 200 | ✅ exact | |
| Relative position bias | ⚠️ structure yes, hyperparameters guessed | Register C-01, C-02 |
| Span-corruption invariants | ✅ exact | ≤8 spans, ≤3 tokens, ≤15%, non-adjacent, ordered sentinels |
| Span-corruption sampling procedure | ⚠️ ours | Register D-01…D-06 |
| Tokenizer structure (5+100+199+154=458) | ✅ exact | |
| Tokenizer *contents* | ❌ substituted | Register B-01, B-02 — the authors' tokens are unpublished |
| Pretraining corpus | ❌ substituted | ~1M PI1M vs 100.1M withheld |
| Pretraining hyperparameters | ⚠️ partly | Batch 450, AdamW, cross-entropy are the paper's; LR/schedule/decay are ours |
| Fine-tuning hyperparameters | ✅ exact | 30/15 epochs, batch 16, 3e-4, wd 0.01, beam 4 |
| Task I/O formats | ✅ exact | Verbatim from the SI examples |
| Tg dataset | ❌ substituted | 7,367 LamaLab (CC-BY) vs 5,130 withheld |
| Evaluation metric definitions | ✅ exact | SV/TSD/DD/PV, SR, TP, SA, Tanimoto |

**Because the corpus, the tokenizer contents, and the labeled data all differ, our numbers are not
comparable to the paper's on an absolute scale.** What *is* comparable is the *methodology* and the
*relative* findings — most importantly the paper's central claim that span-corruption pretraining
substantially improves downstream performance, which we can test directly by running the same
with/without-pretraining ablation on our own data.

## Reference numbers from the paper (targets to contextualize against, not to match)

polyT5-medium, 30 epochs, 80% train, mean of 5 splits:

| Property | N | RMSE | R² | r |
|---|---|---|---|---|
| Tg (K) | 5,130 | 40.82 | 0.86 | 0.93 |
| Td (K) | 4,204 | 78.59 | 0.53 | 0.76 |
| Tm (K) | 2,151 | 67.07 | 0.61 | 0.80 |
| Eg (eV) | 4,113 | 0.60 | 0.83 | 0.92 |
| ε | 1,569 | 0.65 | 0.71 | 0.86 |

Pretraining ablation (the paper's key evidence): Tg RMSE 40.82 → 89.35 without pretraining; Tm R²
0.61 → −1.96; solubility insoluble-class accuracy 0.92 → 0.04.

Best generation configurations (epochs, temperature, top_p): small (14, 0.9, 0.75),
medium (6, 1.1, 0.75), large (8, 1.1, 0.95).

## Our results

*(Training results start at Stage 3. Each entry links a `results/<experiment>/` directory containing
`config.yaml`, `manifest.json`, `metrics.jsonl`, and checkpoints.)*

### Stage 1 — end-to-end vertical slice ✅

`scripts/run_vertical_slice.py`, results in `results/vertical_slice/`. This is a **wiring and correctness
check on 12 real polymer repeat units**, not a training run — the loss numbers mean nothing beyond
"gradients flow".

| | |
|---|---|
| Chemistry | 12/12 PSMILES → `[At]`-capped → PSELFIES, no drops |
| Tokenizer | 458 tokens, sha256 `76471956…`, **0 unknown tokens** on real polymers |
| Token lengths | min 5, max 32, mean 15.3 (max sequence length 200) |
| Span corruption | mean corrupted fraction 0.119, mean 1.5 spans, mean span length 1.4, 16.7% of sequences too short to host a span |
| Paper invariants | asserted and holding: ≤8 spans, ≤3 tokens/span, ≤15% masked, non-adjacent spans, increasing sentinels |
| Model | polyT5-small, **1,437,312 parameters** (paper Table S2: 1.44 M) |
| Device | RTX 4080 Laptop, 12.0 GB, compute 8.9, bf16 supported |
| Training | 5 steps, loss 6.44 → 3.07, peak VRAM 0.06 GB |
| Checkpoint | 16.5 MB; reload reproduces identical logits |

A representative corrupted example, showing the paper's Figure 2C target format (sentinel-prefixed spans
terminated by a final sentinel):

```
input : <extra_id_0> [C][=Branch1][C] <extra_id_1> [C][=C][C][=C][Branch1][Branch1] …
labels: <extra_id_0> [At] <extra_id_1> [=O] <extra_id_2> [O] <extra_id_3> </s>
```

### Stage 2 — real corpus pipeline ✅

Data preparation on the actual downloaded corpora, with the real tokenizer injected so the paper's
200-token limit is applied to *tokenizer output* rather than to a proxy count:

| Corpus | Input rows | Kept | Rate | Dropped by |
|---|---|---|---|---|
| PI1M v2 (dev subset) | 50,000 | 49,821 | **99.6%** | 166 RDKit parse failures, 13 SELFIES failures, **0 over-length** |
| LamaLab curated Tg | 7,367 | 7,354 | **99.8%** | 13 over 200 tokens |

The PI1M parse failures were checked rather than assumed: they are genuine artifacts of that corpus
(hypervalent Si/P, divalent iodine), not a bug in our conversion.

Both downstream task formats are produced verbatim in the paper's I/O shapes:

```json
prediction: {"source": "[At][C][C][O][C][Branch1][C][At][=O]", "target": "249.2"}
generation: {"source": "209.8", "target": "[At][O][Si]…"}
```

Pretraining runs end to end on this corpus (`scripts/pretrain.py`), producing train and validation loss,
per-epoch checkpoints, and throughput/VRAM telemetry on the RTX 4080.

### Tokenizer coverage — a bug worth recording

Our first vocabulary derived its 199 base tokens from the `selfies` library rather than from the corpus.
It passed every hand-written test and tokenized a dozen textbook polymers with zero unknowns. On the real
corpus it produced **2.1% unknown tokens spread across 22.4% of sequences**.

| Base alphabet source | Unknown tokens | Sequences containing `<unk>` |
|---|---|---|
| `selfies` library, widened | 2.11% | 22.42% |
| **frequency-ranked from the training split** | **0.000%** | **0** |

The missing symbols were stereochemistry (`[/C]` alone: 65,823 occurrences per 200k polymers, plus `[/N]`,
`[\C]`), bonded-terminus forms (`[=At]`, `[/At]`, `[#At]`), and explicit-hydrogen/charged species
(`[SH1]`, `[PH1]`, `[2H]`, `[NH1+1]`). This is exactly what the paper describes doing — deriving the base
vocabulary from the corpus — and it is not optional.

Coverage after the fix:

| Corpus | Unknown rate | Note |
|---|---|---|
| PI1M train (vocabulary source) | 0.0000% | by construction |
| PI1M validation | 0.0000% | |
| PI1M test | 0.0001% (3 tokens / 49,411 sequences) | test-only symbols, correctly left as `<unk>` |
| LamaLab Tg prediction/generation | 0.0003% (1 token / 371k) | **a different dataset entirely** — evidence the alphabet generalizes |

`tests/test_integration.py::test_tokenizer_artifact_covers_the_real_corpus` now asserts a zero unknown
rate over the training corpus, so this cannot regress silently.

Measured parameter counts against Table S2 (tied embeddings, `feed_forward_proj="relu"`, vocab 458):

| Config | Ours | Paper | Relative error |
|---|---|---|---|
| small | 1,437,312 | 1.44 M | 0.19% |
| medium | 7,463,168 | 7.46 M | 0.04% |
| large | 58,976,768 | 58.98 M | 0.006% |

This near-exact agreement is itself a finding: it pins down two details the paper never states — the
original is **T5 v1.0 (ReLU feed-forward) with tied input/output embeddings**, at a vocabulary of ~458.

### Pretraining

| Run | Model | Corpus | Epochs | Eff. batch | Train loss | Val loss | Peak VRAM | tokens/s | Results |
|---|---|---|---|---|---|---|---|---|---|
| `pretrain_small_dev` | polyT5-small (1.44 M) | PI1M dev, 44,839 train / 2,491 val | 1 | 128 (32 × 4) | 1.743 | 1.345 | 0.25 GB | 12,725 | superseded (built on the pre-fix vocabulary) |
| **`pretrain_small_pi1m`** | **polyT5-small (1.44 M)** | **PI1M full, 889,402 train / 49,412 val** | **2** | **450 (90 × 5)** | **0.631** | **0.386** | **1.01 GB** | **38,325** | `results/pretrain_small_pi1m/` |
| **`pretrain_polyt5_medium_polyone`** | **polyT5-medium (7.46 M)** | **polyOne dev, 9,229,474 train / 512,748 val** | **2** | **450 (150 × 3)** | **0.128** | **0.0767** | **3.10 GB** | **21,712** | `results/pretrain_polyt5_medium_polyone/` |
| **`pretrain_medium_polyone_92m`** | **polyT5-medium (7.46 M)** | **polyOne, 83,064,393 train / 4,614,689 val** | **2** | **450 (225 × 2)** | **0.058** | **0.0349** | **4.58 GB** | **27,730** | `results/pretrain_medium_polyone_92m/` |

**The paper-scale pretraining run.** 92,293,770 polymers (7.74 B tokens) against the paper's 100,145,883 —
92% of their corpus size, on a 12 GB laptop GPU. Per-epoch validation loss 0.0433 → **0.0349**, versus
0.0767 for the same model on the 9.2 M corpus: **2.2× lower validation loss from 10× the data**, with model
size, batch size, schedule and every other setting held fixed.

Wall clock: 16.1 h per epoch, ~32 h total, at 27,730 target tokens/s and 4.58 GB peak allocation.

polyT5-medium per-epoch validation loss: 0.1083 → **0.0767**.

> **Sizing note, recorded because it cost throughput.** The medium run peaked at **3.10 GB allocated** on a
> 12 GB card at 150 × 3. An earlier `nvidia-smi` reading of 11.5 GB led me to size the batch conservatively;
> that reading was **two training processes colliding**, not one run's footprint. With ~3 GB per 150
> sequences, 450 × 1 — the paper's batch unfactored, no gradient accumulation at all — fits in roughly
> 9.3 GB and is the right setting for the 92.4 M run. Always attribute VRAM to a PID before sizing from it.

Per-epoch validation loss: 0.9918 → **0.3856**.

Notes on the full run, all measured rather than assumed:

- **The effective batch size is the paper's exact 450**, reached as 90 × 5 on a 12 GB laptop GPU instead of
  as a single 450-sequence batch on a 48 GB L40S. The config asserts the product rather than hard-coding it.
- **Epochs match the paper's actual behaviour (2), not its SI text ("up to 5")** — see register E-02.
- **Peak VRAM is 1.01 GB of 12 GB.** polyT5-small at sequence length 200 is nowhere near memory-bound here;
  polyT5-medium and even large should fit comfortably.
- **Wall clock is dominated by one-time tokenization, not by the GPU.** Epoch 0 took 5,215 s while epoch 1
  took 270 s — a 19× gap, because the first pass tokenizes all 889,402 sequences on the CPU and later
  epochs reuse the cache. Pre-tokenizing the corpus to disk is the obvious optimization before scaling to
  polyT5-medium or to a larger corpus. **Done** — see the memmap corpus below.

### Scaling infrastructure: the pre-tokenized memmap corpus

The plain-text path does not survive contact with 100M polymers. Three measured failures:

| Problem | Measurement |
|---|---|
| Corpus held in RAM as Python strings | ~15–20 GB for 100M sequences once object overhead is counted |
| Tokenization repeated every epoch | epoch 0 = 5,215 s vs epoch 1 = 270 s |
| DataLoader workers each copy the list | Windows uses `spawn`; 4 workers × 20 GB is impossible |

Tokenization itself was never the bottleneck — measured **77,665 seq/s** single-core. The 170 seq/s figure
was DataLoader overhead being misread as tokenizer cost.

The fix (`src/polyt5/data/tokenized_corpus.py`) persists the corpus once as `corpus.bin` (flat `uint16`
token ids, no padding), `corpus.idx` (`uint64` offsets, so sequence *i* is `bin[idx[i]:idx[i+1]]`), and
`corpus.json` (metadata including the tokenizer SHA-256). `uint16` because the vocabulary is 458 — `int64`
would be 4× larger for nothing, and `uint8` cannot hold 458. Ragged storage with offsets rather than
padding to 200 saves ~4.4× (mean length is ~84). The file is memory-mapped read-only and reopened per PID,
so DataLoader workers share one page-cached mapping instead of each holding a copy.

Built from polyOne dev (`C:\Users\sumedh\polyt5-data\processed\polyone_dev\`):

| | |
|---|---|
| Input lines | 10,270,270 |
| Kept | **10,254,971 (99.85%)** |
| Dropped | 15,299 over 200 tokens; **0** parse failures, **0** SELFIES failures, **0** wrong-terminus |
| Tokens | 859,776,551 |
| Length min / mean / max | 7 / 83.8 / 200 |
| Splits | 9,229,474 train / 512,748 val / 512,749 test |
| Tokenizer hash | verified against `76471956…` on load; a mismatch refuses to train |

### Novelty at scale, and two ways it fails silently

The paper's **TSD** filter asks whether a generated polymer is already in the training data. The set-based
`NoveltyIndex` holds canonical PSMILES strings — correct for the 6.6k fine-tuning corpus, hopeless against
100M references (~10–15 GB of Python strings, rebuilt on every process start). The RLVR novelty reward
would query it once per candidate inside the rollout loop, from CPU workers that cannot each hold that.

`ScalableNoveltyIndex` stores 64-bit BLAKE2b hashes in a sorted, memory-mapped `uint64` array: 800 MB per
100M entries, `np.searchsorted` for O(log n) lookup, vectorized over a whole batch, shared across processes
by the page cache. Expected colliding pairs at n = 1e8 are n²/2⁶⁵ ≈ **2.7 × 10⁻⁴**, so a false "already
seen" verdict is effectively impossible — which is why sorted hashes are preferred over a Bloom filter here.

Measured against `apply_filter_cascade` on the real 735-candidate generation set, the two implementations
agree **exactly** (735 → 708 → 665 → 428), so evaluation does not care which one it is handed. That
equivalence is pinned by a test.

Both failure modes below were found by running the thing, not by reading it. Both produce **"everything is
novel"**, which reads as an excellent result rather than a broken index — the most dangerous shape a bug
can take here.

| Failure | Cause | Guard added |
|---|---|---|
| 10,000 of 10,000 known polymers reported novel | Index built from the corpus dedup sidecar, which hashes the **raw input line**; queried with PSELFIES. No type distinguishes the two. | `self_check()` verifies that strings known to be present are found — 99.8% in the right space vs **0.0%** in the wrong one |
| 6,619 rows in, **24 hashed** | JSONL `target` field holds PSELFIES; canonicalizing a PSELFIES as PSMILES returns `None` and the row was dropped silently | build script now fails with a diagnosis when the drop rate exceeds `--max-drop-rate` (default 5%), plus a `--from-pselfies` conversion flag |

A further semantic point: an index in **raw** `psmiles` space is the wrong tool for TSD regardless, because
two different SMILES writings of one polymer hash differently and both count as novel. TSD indices must be
built in `canonical_psmiles` space. The dedup sidecar remains correct for what it is — exact-duplicate
removal during corpus preparation — it is simply not a novelty index.

Built artifacts:

| Index | Entries | Size | Build | Self-check |
|---|---|---|---|---|
| `artifacts/novelty/tg_generation_train` | 6,619 | 0.1 MB | 6 s | 100% |
| `polyone_dev/novelty_canonical` | **10,270,270** | 82.2 MB | 1,398 s (7,348 rows/s) | 100% of 20,000 |

Zero rows dropped on the 10.27M build, and every hash unique — polyOne is already deduplicated.

**Query cost is dominated by canonicalization, not by lookup**, and the gap is large enough to change how
the RL loop should be written:

| Path | Throughput |
|---|---|
| `novelty_mask(psmiles)` — canonicalizes each query | 1,326 candidates/s |
| `novelty_mask(canonical, already_canonical=True)` | **335,120 candidates/s** |

That is a **253× difference**. The filter cascade already canonicalizes every candidate for the DD stage,
so a rollout should canonicalize once and pass `already_canonical=True` — at which point novelty checking
is free relative to generation and property prediction. Canonicalizing per query inside an RL loop would
make novelty the bottleneck for no reason.

### Physical batch size matters more than it looks

The paper's batch of 450 says nothing about how to *factor* it, and the factorisation dominates throughput
on a 12 GB card. Measured on polyT5-medium against the memmap corpus:

| Factorisation | GPU utilisation | VRAM | Relative step rate |
|---|---|---|---|
| 45 × 10 | 35% | 1.6 GB | 1.0× |
| 225 × 2 | 93% | 10.5 GB | **2.4×** |
| 150 × 3 *(chosen)* | ~100% | 11.5 GB | ~1.5–2× |

At 45 × 10 the GPU is starved by per-batch Python and DataLoader overhead — the model is small enough that
the fixed cost per step dominates. Larger physical batches amortise it. 150 × 3 was chosen over 225 × 2 to
keep headroom for worst-case all-200-token batches over a multi-hour run. All three give the paper's
effective batch of 450, which `TrainerConfig` asserts rather than assumes.

### Tg property prediction

polyT5-small, 30 epochs, batch 16, AdamW, lr 3e-4, weight decay 0.01, beam search width 4 — all the
paper's fine-tuning hyperparameters. Split 0 of 5, LamaLab Tg (5,295 train / 588 val / 1,471 test).

| Run | Pretrained on | MAE (K) | RMSE (K) | R² | Pearson r | Non-numeric | Results |
|---|---|---|---|---|---|---|---|
| **`finetune_tg_prediction`** | **PI1M full, 2 epochs** | **35.12** | **49.53** | **0.813** | **0.905** | 0/1471 | `results/finetune_tg_prediction/` |
| `finetune_tg_prediction_scratch` | none (random init) | 47.42 | 63.19 | 0.695 | 0.856 | 0/1471 | `results/finetune_tg_prediction_scratch/` |
| **our pretraining effect** | | **−12.30** | **−13.66** | **+0.118** | **+0.049** | | |
| *paper reports (medium, their data)* | *100.1 M polymers* | *—* | *40.82* | *0.86* | *0.93* | *—* | *Table S5* |
| *paper reports, no pretraining* | *—* | *—* | *89.35* | *0.31* | *0.65* | *—* | *Table S5* |

**The paper's central claim reproduces qualitatively.** Span-corruption pretraining improves downstream Tg
prediction substantially: RMSE 63.19 → 49.53, a **21.6% relative reduction**, with R² 0.695 → 0.813. The
paper reports a larger effect (89.35 → 40.82, 54% relative), which is what one would expect from a
pretraining corpus roughly **100× larger** than ours (100.1 M vs 0.89 M polymers) on a larger model. Same
direction, same mechanism, smaller magnitude at smaller scale.

**These numbers are still not a replication of the paper's**, and must never be presented as one:
different corpus (PI1M substitute), different labels (7,367 LamaLab vs 5,130 withheld), different model
size (small vs medium), substitute tokenizer.

#### Five random splits — the paper's protocol, with error bars

polyT5-small pretrained on the full PI1M corpus, then fine-tuned independently on five random 80/20 splits
(`results/tg_prediction_5splits/`). These are *independent draws*, not a partitioning k-fold, so the test
sets legitimately overlap — that is what the paper describes.

| Split | MAE (K) | RMSE (K) | R² | Pearson r | non-numeric |
|---|---|---|---|---|---|
| 0 | 35.12 | 49.53 | 0.813 | 0.905 | 0 |
| 1 | 33.85 | 48.92 | 0.812 | 0.908 | 0 |
| 2 | 33.29 | 47.54 | 0.821 | 0.910 | 0 |
| 3 | 34.71 | 49.74 | 0.808 | 0.903 | 0 |
| 4 | 36.26 | 51.68 | 0.789 | 0.908 | 0 |
| **mean ± sd** | **34.65 ± 1.15** | **49.48 ± 1.50** | **0.808 ± 0.012** | **0.907 ± 0.003** | **0** |

#### Scaling toward the paper, all on the paper's five-split protocol

Every row below is five independent 80/20 splits, identical fine-tuning hyperparameters (30 epochs,
batch 16, lr 3e-4, wd 0.01, beam width 4), identical evaluation. Only the initialisation differs.

| Arm | Model | Pretraining corpus | MAE (K) | RMSE (K) | R² | Pearson r |
|---|---|---|---|---|---|---|
| random init | small | — | 46.24 ± 2.03 | 62.39 ± 2.28 | 0.695 ± 0.020 | 0.853 ± 0.007 |
| pretrained | small | PI1M, 0.89 M | 34.65 ± 1.15 | 49.48 ± 1.50 | 0.808 ± 0.012 | 0.907 ± 0.003 |
| pretrained | medium | polyOne, 9.2 M | 31.05 ± 1.27 | 45.85 ± 1.78 | 0.836 ± 0.012 | 0.917 ± 0.006 |
| **pretrained** | **medium** | **polyOne, 92.3 M** | **28.67 ± 0.76** | **44.45 ± 2.52** | **0.845 ± 0.018** | **0.923 ± 0.008** |
| *paper reports* | *medium* | *withheld, 100.1 M* | *—* | *40.82 ± 1.33* | *0.86* | *0.93 ± 0.01* |

#### Against the published external baselines

The npj version adds model comparisons the preprint did not carry. They reframe our result: the question
is not only "how close to polyT5" but "where does this sit among published chemical language models on the
same task".

| Model | Tg RMSE (K) |
|---|---|
| GPT-3.5, fine-tuned on PSMILES | 47.2 |
| **our reproduction, medium @ 92.3 M** | **44.45 ± 2.52** |
| polyT5-medium (the paper) | 40.82 ± 1.33 |
| Llama-3, fine-tuned | 39.5 |
| polyBART embeddings + Gaussian process regression | 39.9 |

**Our reproduction beats fine-tuned GPT-3.5 and sits within ~4 K of polyT5, polyBART+GPR and Llama-3** —
on substitute data, a substitute tokenizer, and a laptop GPU. That is a more useful statement of where we
landed than "9% short of the paper".

> **Label-noise floor, from the published Methods.** The thermal datasets "should be interpreted as
> literature-reported values under varying conditions" — molecular weight, dispersity and measurement
> protocol are not held constant. That is an irreducible error floor beneath every RMSE in this table, and
> it bounds how much any method can improve. It also matters for Phase 3: it is noise the RLVR reward
> inherits directly.

**At matched scale we land close to the paper.** Our final configuration — polyT5-medium pretrained on
92.3 M polymers, 92% of their corpus size — reaches RMSE **44.45** against their **40.82**, R² **0.845**
against **0.860**, and Pearson r **0.923** against **0.930**. The residual gap is ~9% in RMSE, on
*different labeled data* (7,367 LamaLab experimental values vs their withheld 5,130), a *different*
pretraining corpus (fragment-recombination polyOne vs reaction-enumerated), and a substitute tokenizer.
Two runs on different datasets cannot be compared by a significance test, but the agreement is well inside
what those differences would explain.

The trajectory matters as much as the endpoint: RMSE 62.4 → 49.5 → 45.9 → **44.5** and R² 0.695 → 0.808 →
0.836 → **0.845**, each step moving the expected direction by roughly the expected amount. That is stronger
evidence of a faithful reproduction than any single number matching would be.

**Where the gains came from, now cleanly separated:**

| Change | Δ RMSE | Note |
|---|---|---|
| random init → pretrained (small, 0.89 M) | **−12.9** | ~7 σ; the paper's central claim |
| small → medium *and* 0.89 M → 9.2 M | −3.6 | confounded: two variables at once |
| 9.2 M → 92.3 M corpus, model fixed | **−1.4** | controlled; also val loss 0.0767 → 0.0349 |

Most of the benefit comes from pretraining existing at all, not from scale — the last 10× of corpus buys
1.4 K of RMSE. That is consistent with the paper's own finding that polyT5-large offers only "marginal
improvements over the medium variant", and it is a useful result in its own right: the expensive part of
this pipeline is not the one that pays.

#### The pretraining ablation, both arms over five splits

Identical protocol for both arms — same splits, same hyperparameters, same evaluation — differing only in
whether the model starts from the PI1M-pretrained checkpoint or from random initialisation.

| Metric | Pretrained (n=5) | From scratch (n=5) | Δ | Effect size |
|---|---|---|---|---|
| MAE (K) | **34.65 ± 1.15** | 46.24 ± 2.03 | **−11.59** | **7.0 σ** |
| RMSE (K) | **49.48 ± 1.50** | 62.39 ± 2.28 | **−12.91** | **6.7 σ** |
| R² | **0.808 ± 0.012** | 0.695 ± 0.020 | +0.113 | 6.8 σ |
| Pearson r | **0.907 ± 0.003** | 0.853 ± 0.007 | +0.054 | 9.8 σ |

**The paper's central claim reproduces, unambiguously.** Span-corruption pretraining cuts Tg prediction
error by ~21% relative, at an effect size of roughly seven pooled standard deviations. This is not a
one-split artifact: both arms were run over the paper's own five-split protocol, so the comparison is
symmetric and the variance is measured rather than assumed.

The paper's own effect is larger (RMSE 40.82 pretrained vs 89.35 scratch). That is the expected direction
of the difference: their pretraining corpus is ~100× ours (100.1 M vs 0.89 M polymers) and their model is
medium rather than small. We reproduce the *mechanism* at a fraction of the scale, and the scale gap shows
up exactly where it should — in the magnitude of the benefit, not its existence.

Every split decoded all 1,471 test polymers to valid numbers under beam-4 search: **zero non-numeric
outputs anywhere**, so the paper's "filtered to remove any invalid or non-numeric outputs" step removes
nothing at this model size.

The five runs also leave **five independently trained checkpoints**, which is what the ensemble and the
held-out auditor are built from — see `docs/rlvr_plan.md` §7. That capability is a by-product of running
the paper's own protocol properly, not extra work.

Both arms decoded all 1,471 test polymers to valid numbers under beam-4 search: **zero non-numeric
outputs**, so the paper's "filtered to remove any invalid or non-numeric outputs" step removed nothing here.

An earlier version of this table (pretraining on a 45k dev subset for one epoch) showed **no** pretraining
effect at all — RMSE 60.96 vs 61.61. That null result was a scale artifact, not a finding, and is recorded
here only because it is the reason the full-corpus run was done.

### Tg-conditioned generation

polyT5-small initialized from `pretrain_small_pi1m`, fine-tuned 15 epochs, batch 16, lr 3e-4, wd 0.01,
90/10 split, no masking — the paper's settings. Decoded by sampling (**not** beam search, per the paper) at
`top_p = 0.75`, `temperature = 1.1`. Input is a bare target Tg string, output is PSELFIES. Final
train loss 0.758, val loss 0.661.

The paper's four nested filters, applied in order (SV ⊇ TSD ⊇ DD ⊇ PV), over 735 conditioned samples:

| Filter | Meaning | Count | Rate |
|---|---|---|---|
| SV | RDKit-parseable | 735 | **100.0%** |
| TSD | not in the training set | 708 | 96.3% |
| DD | unique within the batch | 665 | 90.5% |
| PV | exactly two `[At]`, each valency 1 | 428 | **58.2%** |

| Other metrics | |
|---|---|
| SELFIES reproducibility (SR) | 25.4% |
| Duplicate rate | 7.3% |
| SA score (mean / median) | 4.24 / 4.10 |
| SA above 6 (the paper's awkwardness threshold) | 11.4% |
| Mean pairwise Tanimoto (ECFP6, 200-pair sample) | 0.106 |

An earlier version of this table reported TSD at 100%; that run supplied **no novelty index**, so nothing
could fail the filter. With the real training corpus indexed, 27 of 735 candidates (3.7%) turn out to be
training-set members. Measuring novelty without a reference set measures nothing.

> **PV pass rate is our clearest shortfall against the paper.** The published version reports **~80.6%**
> of candidates passing PV at its medium optimum (6 epochs, T = 1.1, top_p = 0.75). Our best is **58.6%** —
> a 22-point gap. This is the one place we fall short of a *published number* rather than merely differing
> on substitute data, so it is worth naming plainly.
>
> The most likely cause is an axis we never swept. The paper tunes `(epochs, temperature, top_p)` jointly
> and its optimum sits at **6 fine-tuning epochs**; our generation models were fine-tuned for 15, and the
> sampling sweep varied only temperature and top_p against a single checkpoint. The paper also reports that
> validity *improves* with fine-tuning epochs up to a point and then degrades into duplication — so 15
> epochs may sit past that peak. Re-running the generation fine-tune with every-epoch checkpoint retention,
> then sweeping the epoch axis, is the experiment that would close or explain this gap.
>
> For context the published version also reports polyBART at **86.7%** on analogous filters, so ~80% is
> achievable rather than exceptional.

#### Conditioning fidelity — the number RLVR has to beat

The candidates above were conditioned on the validation set's own Tg values, which span **167.2 – 729.1 K**.
Scoring each candidate against *its own* requested target (not against a fixed 500 K) gives:

| | |
|---|---|
| Scored | 667 of 667 PV-passing candidates |
| **MAE vs requested Tg** | **50.4 K** |
| Median absolute error | 40.0 K |
| Mean signed bias | **−4.5 K** (well calibrated — no systematic drift) |
| Within ±25 K | 34.5% |
| Within ±50 K | 59.2% |
| Within ±100 K | 88.5% |

For scale, the Tg predictor's own held-out MAE is 35.1 K, so conditioning adds roughly 15 K on top of
predictor error.

> ⚠️ **Circularity caveat.** The predictor and the generator were fine-tuned on the same LamaLab Tg data, so
> using one to score the other is partly self-referential. This is the exact failure mode the RLVR phase
> must defend against, since the policy would be free to farm the predictor's blind spots. The mitigation —
> an ensemble of five independently-split Tg models plus one held-out auditor never used in any reward — is
> being built now (see `docs/rlvr_plan.md` §7).

Reported at a fixed 500 K target instead, for comparability with the paper's Figure S10 protocol:
**TP = 31.5%** within 500 ± 50 K. That figure answers a different question than the table above, and the
distinction is recorded in `results/finetune_tg_generation/evaluation.json`.

### Two measurement findings

Both were flagged as suspected bugs by separate tracks. Neither is a bug; both change how the metrics
should be read, and both matter for the RLVR phase.

#### 1. SELFIES reproducibility is a ring-encoding artifact

SR looked alarmingly low (0.00–0.03 across the sampling sweep). Breaking it down by structure:

| Generated structure | SR | n |
|---|---|---|
| **contains no `[Ring*]` token** | **100.0%** | 79 |
| contains `[Ring*]` | 16.5% | 656 |

| Generated length (SELFIES tokens) | SR | n |
|---|---|---|
| 0–19 | 96.4% | 28 |
| 20–39 | 59.6% | 89 |
| 40–59 | 8.0% | 150 |
| 60–119 | 2–4% | 356 |

So SR is not measuring model quality in any general sense — it measures whether a molecule survives a
**ring-numbering re-derivation**. Decode a ring-bearing SELFIES to SMILES and re-encode, and the ring
index tokens come back in a different but chemically equivalent phrasing, so string equality fails. Every
ring-free output round-trips perfectly.

For context, the paper reports SR = 18.5% (3,978 of 21,457) on its screened set — the same regime as our
16.5% for ring-bearing structures.

> **Consequence for RLVR:** rewarding SR directly would push the policy toward **ring-free aliphatic
> polymers**, which is a degenerate incentive — most high-Tg polymers are aromatic. If SR enters the
> reward at all, it must be conditioned on structure class or weighted near zero. Recorded in
> `docs/rlvr_plan.md` §7.

#### 2. The Tg predictor's output is coarse, because the labels are

The predictor emits only **146 distinct values across 1,471 test polymers**, and 92.9% of its predictions
end in `.1`. That looked like a decoding collapse. It is not — it mirrors the label distribution:

| | training labels | predictions |
|---|---|---|
| Distinct values | 1,095 of 5,295 | 146 of 1,471 |
| Fraction ending in `.1` | 80.3% | 92.9% |

The LamaLab Tg values are Celsius measurements converted to Kelvin (+273.15), so they pile up on `.1`
decimals, and common round-number Celsius values repeat heavily — 32 training polymers share the label
323.1 K (= 50 °C). The model reproduces that lumpiness, slightly amplified.

> **Consequence for RLVR:** the property reward is **coarse-grained and multi-modal**, with a large
> attractor at the most common training labels (22.9% of scored generations land on exactly 503.1 K).
> A policy optimizing it can satisfy "≈500 K" by steering into a modal bucket rather than by genuinely
> controlling Tg. This is a concrete, measured reward-hacking surface, not a hypothetical one, and it is
> the strongest argument for the ensemble-disagreement gate.

For context, not comparison: the paper reports SR of 18.5% (3,978 of 21,457) on its screened candidate set,
so our 25.4% is in the same regime. Per-configuration SV/TSD/DD/PV rates were never published numerically
(heatmaps only), so there is nothing to compare the 60.4% against.

**What is not yet measured:** target-property error. `evaluate_generation` takes the property model as an
injected callable, and wiring the Stage-5 Tg predictor into it is the next step — that yields the paper's
TP metric (fraction within target ± 50 K) and closes Stage 7.
