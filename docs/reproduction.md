# Reproduction notes: what polyT5 actually specifies

**Target paper.** Harikrishna Sahu, Wei Xiong, Anagha Savit, Shivank S. Shukla, Rampi Ramprasad.
*POLYT5: an encoder-decoder foundation chemical language model for generative polymer design.*
npj Artificial Intelligence **2**, 30 (2026). DOI [10.1038/s44387-026-00087-1](https://doi.org/10.1038/s44387-026-00087-1).
Preprint: [arXiv:2510.18860](https://arxiv.org/abs/2510.18860) (titled *"An Encoder-Decoder Foundation
Chemical Language Model for Generative Polymer Design"* — the `POLYT5:` prefix was added in review).

This file is the **evidence base** for the reproduction. It separates three things that must never be
conflated:

| Tag | Meaning |
|---|---|
| `[PAPER]` | Stated explicitly in the paper or its Supplementary Information. Quoted or numerically exact. |
| `[IMPLIED]` | Derivable from the paper with a short argument, but never stated. |
| `[OURS]` | A choice **we** made because the paper does not specify it. Every one of these is a deviation risk. |

A companion machine-readable list of every unspecified detail lives in
[`docs/ambiguity_register.md`](ambiguity_register.md). Nothing in this repository may claim to reproduce a
number without a corresponding entry here.

---

## 0. Headline finding: this paper is not reproducible from released artifacts

`[PAPER]` Data availability statement, quoted verbatim from the published version:

> "The datasets generated and/or analysed during the current study are **not publicly available due to IP
> protection being considered at authors' institution**."

`[PAPER]` There is **no code availability section at all**. Independent search confirms:

- No `polyT5` repository in the [Ramprasad-Group GitHub org](https://github.com/Ramprasad-Group) (full repo
  list enumerated via the GitHub API, 2026-08-17).
- GitHub code search for `polyt5` returns **0 repositories**.
- HuggingFace model search for `polyT5` returns **an empty list**.
- No Zenodo / Figshare / Dryad deposit is linked to the DOI.

Consequently **the weights, the 100M-polymer pretraining corpus, the labeled property datasets, and the
458-token tokenizer artifact are all unavailable.** What we can reproduce is the *method*, on *substitute
data*, at *reduced scale*. See [`docs/data.md`](data.md) for the substitute corpus and its licensing, and
§9 below for what this costs us scientifically.

---

## 1. Representation: PSMILES → PSELFIES

`[PAPER]` Polymers start as **PSMILES**, ordinary SMILES in which **two `[*]` tokens denote the terminal
ends of the polymer chain**. Note this is `[*]`, *not* `[At]`.

`[PAPER]` The paper then explains why `[*]` cannot be used for the model, quoted verbatim:

> "However, since [*] is conventionally used to denote dummy atoms and is not supported in SELFIES—an
> inherently robust representation well suited for generative modeling of molecules—we developed a custom
> conversion strategy to overcome this limitation."

We verified this claim locally: `selfies.encoder('[*]CC[*]')` raises `EncoderError`, while
`selfies.encoder('[At]CCO[At]')` returns `[At][C][C][O][At]` and round-trips exactly.

`[PAPER]` The conversion procedure, verbatim — **and it is more subtle than a token substitution**:

> "During this process, the two polymer ends were first joined by removing the [*] tokens to form a cyclic
> structure, followed by canonicalization to mitigate any initial sequence bias. Subsequently, a single
> bond within the backbone was strategically cleaved, and Astatine (At) atoms were attached at the
> cleavage points. At was selected because it is rarely encountered in polymer structures and was entirely
> absent from our training dataset, minimizing the risk of unintended bias. The resulting pseudo-polymer
> SMILES—molecular representations with At atoms marking the polymer termini—were subsequently converted
> to SELFIES, referred to as PSELFIES."

So the pipeline is:

```
PSMILES with [*]…[*]
    → delete the two [*] and bond their neighbours together  (cyclize)
    → canonicalize the ring                                   (removes starting-point bias)
    → cleave ONE backbone bond                                (the "strategic" step)
    → cap both cleavage points with [At]                      (pseudo-polymer SMILES)
    → selfies.encoder(...)                                    = PSELFIES
```

> **`[OURS]` — the single largest representation ambiguity.** The paper never says *which* backbone bond is
> "strategically cleaved", nor which canonicalizer is used. Different cleavage choices produce **different
> PSELFIES strings for the same polymer**, which changes the entire token distribution the model learns.
> Our implementation lives in `src/polyt5/chemistry/conversion.py` and offers an explicit, named strategy
> (see `docs/ambiguity_register.md` entry A-01). The simple `[*]→[At]` substitution is *also* provided,
> and is what we use by default for the development-scale pipeline, because it is invertible and testable;
> the loop-canonicalize-break variant is provided for faithfulness experiments.

Implementing it surfaced three things worth recording, because the paper states none of them:

1. **The canonicalization claim checks out.** The paper says canonicalizing the ring "mitigate[s] any
   initial sequence bias". We tested this directly: 14 different SMILES writings of 4 polymers — reversed
   chain direction, branch-first, kekulized versus aromatic, `[*]` versus `[At]` — each collapse to
   exactly one loop-break PSELFIES, with the four groups staying mutually distinct.
2. **The pipeline genuinely relocates the termini**, which is the whole reason the ambiguity matters.
   Poly(ethylene oxide) `[At]CCO[At]` cyclizes to oxirane; the canonical-rank rule then cleaves the C–C
   bond rather than the C–O bond, yielding `[At]COC[At]` — a different *phase* of the same infinite chain.
   PET-like and nylon-6-like backbones relocate too; bisphenol-A does not.
3. **The paper's described pipeline cannot run on vinyl polymers at all.** Joining the two termini requires
   a one- or two-membered ring whenever both termini sit on the same atom or on two already-bonded atoms.
   That describes polyethylene, polypropylene, PVC and every other vinyl polymer, plus ortho-phenylene.
   RDKit cannot represent such rings. The paper never mentions this class, and its corpus composition
   table (Table S1) reports hydrocarbons at <0.1%, so it may simply never have arisen at scale — but any
   re-implementation must decide what to do, and we fall back to the direct substitution while counting
   how often that happens (register A-04).

`[PAPER]` Worked example (Figure 2A, a polycarbonate):

```
PSMILES : CC(C)(C1=CC=C(O[*])C=C1)C1=CC=C(OC([*])=O)C=C1
PSELFIES: [C][C][Branch1][C][C][Branch1][C][At][C][=C][C][=C][Branch2][Ring1][C][O][C]
          [=Branch1][C][=O][O][C][=C][C][=C][Branch1][C][At][C][=C][Ring1][#Branch1]
          [C][=C][Ring1][P]
```

`[PAPER]` **Maximum sequence length = 200**, chosen because it "is sufficient to cover 99.91% of the
SELFIES tokens present in the training dataset" (Figure S1). Applies to pretraining, both fine-tuning
tasks, and generation.

`[OURS]` The `selfies` package **version is never stated** — and the SELFIES grammar changed materially
between v1.x and v2.x. The bracket forms in the paper's own examples (`[Branch2][Ring1][C]`, `[Ring1][P]`)
are consistent with **v2.x**, so we pin `selfies>=2.1.1` and record the exact version in every run
manifest. RDKit's version is likewise unstated; we record ours.

`[PAPER]` `[At]` is a *representation device for chain ends*, never a chemically meaningful product. The
paper's own evaluation replaces `[At]`/`[*]` with hydrogens before computing synthetic accessibility.

---

## 2. Tokenizer

`[PAPER]` Total vocabulary = **458 tokens**, and SentencePiece is used only as a *compatibility wrapper*
around a fully **predefined** symbol table. Verbatim from the SI:

> "For tokenizing the PSELFIES strings, each substring enclosed within square brackets (e.g., [C], [O])
> was treated as a distinct token, resulting in a base vocabulary of **199 unique tokens**. Several special
> tokens were also introduced, including start- and end-of-sequence markers, unknown and padding tokens, a
> whitespace marker, and **100 sentinel tokens** for masking during pre-training. To further expand the
> vocabulary and enable property-conditional generation and prediction, an additional **154 tokens** were
> incorporated. These included property names, numerical digits (0-9), decimal point (.), units, arithmetic
> and relational operators (+, -, >, <, =, etc.), boolean values, and a set of common polymer-related
> keywords. This resulted in a final vocabulary size of **458 tokens**. To ensure compatibility with the
> SentencePiece tokenizer framework, all SELFIES tokens and additional custom tokens were included as
> predefined tokens."

| Group | Count | Tag |
|---|---|---|
| Base SELFIES tokens (bracketed substrings) | 199 | `[PAPER]` |
| Special tokens (SOS, EOS, UNK, PAD, whitespace marker) | 5 | `[IMPLIED]` — named but never counted |
| Sentinel tokens `<extra_id_n>` | 100 | `[PAPER]` |
| Additional property/conditioning tokens | 154 | `[PAPER]` |
| **Total** | **458** | `[PAPER]` |

199 + 5 + 100 + 154 = 458 exactly, which is why we treat "several special tokens" as exactly the five
named ones. The published Methods paraphrase can also be read as 199 + 100 + 159, folding the specials into
the "additional" bucket. Both sum to 458; the 154-vs-159 boundary is soft.

**Two hard blockers, both recorded in the ambiguity register:**

- `[OURS]` **The 199 base tokens are corpus-derived and the corpus is withheld.** We cannot regenerate them.
- `[OURS]` **The 154 conditioning tokens are never enumerated** — no table, no appendix, in either version.
  Only their *categories* are given. Observed instances in the paper's worked examples are just
  `property`, `polymer`, `solvent`, `;`, `soluble`, `insoluble`.

`[OURS]` **Our substitute**, in `src/polyt5/tokenization/`, matches the paper's *structure* and headline
size: 5 specials + 100 sentinels + a 199-token SELFIES alphabet + an explicitly enumerated 154-token
conditioning set of our own design. The artifact is content-hashed (SHA-256) and its metadata states
plainly which groups are the paper's and which are ours. It is built once by `scripts/build_tokenizer.py`
and reused byte-identically for pretraining, fine-tuning, evaluation, generation, and future RL rollouts.

> **The base alphabet must be derived from the corpus, not from the SELFIES library.** We learned this the
> expensive way. Our first build derived the 199 base tokens from the `selfies` package's semantic-robust
> alphabet, widened for polymer elements. It tokenized a dozen textbook polymers with zero unknowns — and
> then hit **2.1% unknown tokens across 22.4% of sequences** on the real corpus. The missing tokens were
> dominated by stereochemistry (`[/C]` alone accounted for 65,823 occurrences in 200k polymers, plus
> `[/N]`, `[\C]`), by **bonded-terminus variants** (`[=At]`, `[/At]`, `[#At]` — astatine carrying a double,
> directional or triple bond), and by explicit-hydrogen and charged forms (`[SH1]`, `[PH1]`, `[2H]`,
> `[NH1+1]`).
>
> Rebuilding the base block frequency-ranked from the training split — which is what the paper describes
> doing — gives **0 unknown tokens on train and validation**, and 3 unknown tokens across 49,411 test
> sequences (0.0001%), those being symbols that genuinely occur only in held-out data. Deriving the
> vocabulary from the training split only, and letting rare test-only tokens map to `<unk>`, is the honest
> treatment.
>
> A side observation on the paper's own number: the ~1M-polymer PI1M corpus contains only **111 distinct
> SELFIES bracket tokens**. The paper's 199 came from a 100M-polymer corpus enumerated across far more
> reaction families, so 199 is plausible for their data and comfortably oversized for ours — we keep 199
> slots so the structure matches, with the surplus filled by explicit reserved placeholders rather than
> invented chemistry.

`[OURS]` We do **not** train a SentencePiece model. Since every token is predefined, a learned subword
model would be an identity wrapper at best; a deterministic longest-match tokenizer over the symbol table
is exactly equivalent and vastly more inspectable. A SentencePiece-style `.vocab` side-artifact can be
emitted for interoperability.

`[OURS]` Standard T5 has **no BOS token** and uses `<pad>` as `decoder_start_token_id`, yet the paper claims
a start-of-sequence marker among its five specials. We include `<s>` so the arithmetic closes, and default
`decoder_start_token_id` to the pad id per T5 convention.

`[PAPER]` A useful sanity note from the paper's own discussion of the loss plateau: during *pretraining*
only the 199 SELFIES tokens plus sentinels are actually exercised — the 154 conditioning tokens are inert
until fine-tuning.

---

## 3. Architecture

`[PAPER]` Table S2, exactly as published:

| Parameter | small | medium | large |
|---|---|---|---|
| `d_model` | 128 | 256 | 512 |
| `num_layers` (encoder **and** decoder) | 3 | 4 | 8 |
| `d_ff` | 512 | 1024 | 2048 |
| `num_heads` | 4 | 4 | 8 |
| `d_kv` | 32 | 64 | 64 |
| `n_positions` | 200 | 200 | 200 |
| **total parameters (M)** | **1.44** | **7.46** | **58.98** |

`[PAPER]` "All models employ **relative positional encodings** with a maximum input length of 200 tokens."
`[IMPLIED]` `d_kv × num_heads == d_model` holds for all three variants — standard T5 convention.

`[IMPLIED]` **The original is almost certainly HuggingFace `T5ForConditionalGeneration`.** Evidence: Table S2
uses the exact HF `T5Config` field names including the HF-specific alias `n_positions`; fine-tuning used
the "HuggingFace `Seq2SeqTrainer` API" (stated verbatim); and label padding is set to `-100`, the
HF/PyTorch `ignore_index` idiom. The paper never states the implementation used for *pretraining*.

`[OURS]` **We implement T5 from scratch in plain PyTorch** rather than depending on `transformers`. The
reasons are deliberate: the user wants an inspectable research implementation, and the planned GRPO/RLVR
extension needs direct access to per-token log-probabilities, a reference-policy copy, and a custom loss —
all of which are cleaner without a trainer framework in the way. We reproduce HF T5 *semantics*
(T5LayerNorm without mean subtraction, no bias terms, unscaled attention logits, first-layer-only relative
position bias, `_shift_right`, tied embeddings with the `d_model**-0.5` rescale).

`[OURS]` **Never stated by the paper, so configurable with HF defaults**: `dropout_rate` (0.1),
`relative_attention_num_buckets` (32), `relative_attention_max_distance` (128), `feed_forward_proj`
(T5 v1.0 `relu` vs v1.1 `gated-gelu`), `tie_word_embeddings`, `layer_norm_epsilon` (1e-6), initializer
scheme, `decoder_start_token_id`.

> Note a real consequence: with `n_positions = 200` and the HF default `max_distance = 128`, relative
> positions beyond 128 saturate into the final bucket. The paper does not address this.

---

## 4. Span-corruption pretraining

`[PAPER]` Verbatim:

> "For each polymer sequence, **up to 8 non-overlapping masked spans (each up to 3 tokens long)** were
> randomly selected to **mask up to 15% of the input tokens**. These spans were replaced with sentinel
> tokens (`<extra_id_n>`) in the input sequence, and the target sequence was constructed by **concatenating
> the masked spans, each prefixed with its corresponding sentinel token**. The sentinel tokens were
> **assigned in increasing numerical order** of n and placed such that **no two masked spans were adjacent,
> ensuring at least one unmasked token between them**."

`[PAPER]` Figure 2C worked example (span0 = positions 1–3, gap 3, span1 = positions 7–9, gap 2,
span2 = positions 12–13), and the target terminates with a **trailing final sentinel** — the standard T5
convention.

> **This is not canonical T5.** Canonical T5 uses a 15% rate with *mean* span length 3 and derives the span
> *count* from sequence length. Here the span count is hard-capped at 8 and span length hard-capped at 3.
> Since 8 × 3 = 24 tokens, the 15% cap only binds for sequences longer than ~160 tokens; below that, the
> 8×3 cap binds. Implementing ordinary masked LM here would be a reproduction error.

`[OURS]` The paper leaves the *sampling procedure* undefined: which of the three "up to" limits is applied
first, whether span length is uniform on {1,2,3}, whether 15% is of padded or unpadded length, what happens
to sequences too short to host a span, and whether EOS may be masked. Our disambiguation is implemented and
documented in `src/polyt5/data/span_corruption.py` and tested against every invariant the paper *does*
state (≤8 spans, ≤3 tokens, ≤15%, non-adjacency, increasing sentinels, exact reconstruction).

### Pretraining corpus and schedule

| Item | Value | Tag |
|---|---|---|
| Experimentally synthesized polymers from literature | 12,473 | `[PAPER]` |
| Hypothetical polymers | 100,145,883 (Table S1) | `[PAPER]` |
| Training split | 90% ≈ 90M masked sequences | `[PAPER]` |
| Held out | 10% for "validation and testing" (sub-split unstated) | `[PAPER]` / `[OURS]` |
| Batch size | 450 | `[PAPER]` |
| Epochs | **contradictory — see below** | `[PAPER]` |
| Hardware | a single NVIDIA L40S | `[PAPER]` |
| Optimizer | AdamW | `[PAPER]` |
| Loss | token-level cross-entropy | `[PAPER]` |
| Checkpoints | after each epoch | `[PAPER]` |
| Learning rate, LR schedule, warmup, weight decay | **never stated** | `[OURS]` |
| Mixed precision, grad accumulation, grad clipping, seeds, wall-clock | **never stated** | `[OURS]` |

> **`[PAPER]` Internal contradiction.** The SI says pretraining ran "for **up to 5 epochs**", but the main
> text says Figure S2 shows loss "across **two epochs** for all three polyT5 model variants", and the
> Figure S2 x-axis runs 0.00 → 2.00. We treat **2 epochs** as what was actually done and record the
> discrepancy.

`[PAPER]` How the 100M hypothetical polymers were made: known polymerization reactions (polyamides,
polyimides, polyesters, polyethers, polyureas, polyurethanes) applied to commercially available small
molecules from **eMolecules** (retrieved 2024-05-31), **ChEMBL**, and **ZINC-15**; plus **ROMP**; plus click
chemistry — **CuAAC, SPAAC, thiol-ene/yne/bromo coupling, Diels-Alder (furan-maleimide), SuFEx, and
oxime** reactions. `[OURS]` **No reaction SMARTS, monomer filters, screening rules, or dedup procedure are
given.** This generation process is *not* reproducible from this paper, and per the project brief we do not
attempt to recreate it — we substitute a documented public corpus instead (see [`docs/data.md`](data.md)).

---

## 5. Downstream task A — property prediction

`[PAPER]` Formulated as seq2seq: PSELFIES in, a *textual* numeric value out. Verbatim SI examples:

```
Thermal / electronic (Tg, Tm, Td, Eg):
  INPUT : [C][C][C][C][Branch1][C][At][C][At]
  OUTPUT: 236.0

Dielectric constant (ε):
  INPUT : property 4.1; polymer [C][C][Branch1][C][At][C][At]
  OUTPUT: 3.7

Solubility:
  INPUT : polymer [C][C][Branch1][C][At][C][At]; solvent [C][C][C][O][C][Ring1][Branch1]
  OUTPUT: soluble        (or: insoluble)
```

Three format facts that are easy to get wrong:

1. `[PAPER]` **There is no task prefix** for the thermal/electronic tasks. The input is the bare PSELFIES.
   The property identity is carried by *which fine-tuned checkpoint you load* — "fine-tuning was performed
   separately for each property".
2. `[PAPER]` The literal token `property` in the ε example is a **field label for the log-frequency value**,
   not a property name. The SI glosses the format as "Property tag with log(frequency) followed by
   PSELFIES", and `4.1 ≈ log₁₀(10⁴ Hz)` matches one of the nine measured frequencies. `[IMPLIED]` base 10.
3. `[PAPER]` Solvents carry **no** `[At]` — they are ordinary molecular SELFIES. The separator is `"; "` and
   field labels are bare lowercase words.

`[IMPLIED]` Numeric outputs use **one decimal place** (`236.0`, `3.7`) — but this rests on two examples, and
whether ε and Eg use the same precision is `[OURS]`.

`[PAPER]` Hyperparameters: up to **30 epochs**, batch size **16**, AdamW, LR **3e-4**, weight decay **0.01**,
token-level cross-entropy with pad labels set to **-100**, evaluation each epoch, **beam search with beam
width 4**, decoded outputs "filtered to remove any invalid or non-numeric outputs", metric **MAE** (also
RMSE, R², Pearson r reported), **five random splits**, headline split **80/20**, learning curves swept from
20% to 80% training fraction.

`[OURS]` Unstated: LR schedule/warmup, dropout, seeds, checkpoint-selection criterion (best vs last epoch —
indeed no validation set is described for this task at all), beam-search length penalty, how the rare
non-numeric outputs are counted in the metric, and whether targets were normalized before stringification.

### `[PAPER]` Reported results (polyT5-medium, 30 epochs, 80% train, mean over 5 splits, σ in parens)

| Property | N | RMSE | R² | r |
|---|---|---|---|---|
| Tg (K) | 5,130 | 40.82 (1.33) | 0.86 (0.00) | 0.93 (0.01) |
| Td (K) | 4,204 | 78.59 (3.56) | 0.53 (0.03) | 0.76 (0.02) |
| Tm (K) | 2,151 | 67.07 (5.18) | 0.61 (0.07) | 0.80 (0.03) |
| Eg (eV) | 4,113 | 0.60 (0.03) | 0.83 (0.02) | 0.92 (0.01) |
| ε | 1,569 | 0.65 (0.10) | 0.71 (0.11) | 0.86 (0.04) |

Solubility (29,215 cases = 6,246 polymers × 58 solvents): soluble 0.96, insoluble 0.92, overall 0.94.

The pretraining ablation is striking and is the paper's core evidence: without pretraining, Tg RMSE
degrades 40.82 → 89.35, Tm R² goes **negative** (0.61 → −1.96), and the solubility classifier collapses to
predicting "soluble" almost always (insoluble accuracy 0.92 → 0.04).

---

## 6. Downstream task B — Tg-conditioned generation

`[PAPER]` Verbatim SI format — note it is the *exact inverse* of property prediction:

```
  INPUT : 236.0                                   (bare numeric target Tg, no prefix, no unit)
  OUTPUT: [C][C][C][C][Branch1][C][At][C][At]     (PSELFIES)
```

`[PAPER]` Only **Tg** was used for generation, "due to its significance in materials design, broad coverage
across diverse chemistries, and the relative completeness of the available data".

`[PAPER]` Hyperparameters: HuggingFace `Seq2SeqTrainer`, up to **15 epochs**, batch size **16**, AdamW,
LR **3e-4**, weight decay **0.01**, cross-entropy with `-100` label padding, **90% train / 10% validation**,
max sequence length 200, **no masking in this phase**, checkpoint each epoch, `predict_with_generate`
enabled.

`[PAPER]` Sampling — **not beam search** for this task:

- `top_p ∈ {0.75, 0.95}`
- temperature **0.1 → 2.0 in steps of 0.1** (20 values)
- fine-tuning epochs 1 → 15 (15 values)
- **10,000 polymers generated per configuration**
- Full grid = 3 models × 15 epochs × 20 temperatures × 2 top_p = **1,800 configurations**

`[PAPER]` Best configurations (Figure 4A caption), as (epochs, temperature, top_p):

| Model | epochs | temperature | top_p |
|---|---|---|---|
| small | 14 | 0.9 | 0.75 |
| **medium** | **6** | **1.1** | **0.75** |
| large | 8 | 1.1 | 0.95 |

> ✅ **80.6% is real — and it is a reproduction target.** An earlier version of this document recorded the
> brief's "approximately 80.6% of candidates passing the PV filter" as unverifiable. That was correct for
> the **arXiv preprint**, which publishes per-configuration results only as unlabeled heatmaps (Figures 4A,
> S7, S9, S10). The **published npj version states it outright**:
>
> > "the best balance we observed for POLYT5-medium was obtained at 6 fine-tuning epochs with top_p=0.75
> > and a sampling temperature of T = 1.1, yielding **~80.6% of generated candidates passing the PV filter**."
>
> So PV pass rate is a quotable target after all. **Ours is 58.6%** at our best configuration — a 22-point
> gap, and the first place our reproduction falls clearly short of a published number rather than merely
> differing on substitute data. The most likely cause is the axis we never swept: the paper tunes
> `(epochs, temperature, top_p)` jointly and its optimum sits at **6 fine-tuning epochs**, while our
> generation models were fine-tuned for 15 and only a few epoch checkpoints were retained. See
> `docs/baseline.md`.

`[PAPER]` Qualitative trade-offs the paper does state: more fine-tuning epochs first reduces invalid
candidates but eventually increases duplicates (pronounced for medium/large); higher temperature increases
invalid PSMILES; lower temperature increases duplication; `top_p = 0.95` gives more diversity and more
invalid structures, `top_p = 0.75` gives fewer invalid and more duplicates.

---

## 7. Evaluation metrics — exact definitions

`[PAPER]` Four **nested** generation filters, verbatim:

> "First, **SMILES Validity (SV)** was determined using RDKit to ensure the chemical validity of the
> generated structures. Next, **Training Set Deduplication (TSD)** filtered out any candidates that were
> already present in the training dataset. The **Dataset Deduplication (DD)** step removed duplicates within
> the generated set itself, retaining only unique candidates. Finally, **PSMILES Validity (PV)** ensured that
> each retained candidate contained **exactly two Astatine (At) atoms, each with a valency of one**, as
> required by the polymer design rules. These filters follow a nested relationship: SV ⊇ TSD ⊇ DD ⊇ PV."

> ⚠️ **Correction to the project brief.** **PV = "PSMILES Validity"**, not a property-validity filter. Its
> criterion is purely structural: *exactly two `[At]` atoms, each with valency 1*. The filters are
> **sequential and nested**, so every reported count is conditional on passing all prior filters — a
> "PV pass rate" is therefore a rate over the *original* sample, already net of validity, novelty, and
> uniqueness.

Mapping to the usual vocabulary: **validity** = SV, **novelty** = TSD (absent from training set),
**uniqueness** = DD (deduplicated within the generated batch), **PV** = the polymer-terminus check.

`[PAPER]` **SR — SELFIES Reproducibility**, verbatim:

> "the SELFIES reproducibility (SR) metric—defined as the fraction of generated SELFIES strings that, when
> converted to SMILES and back to SELFIES, yield the identical string"

i.e. `SR = mean[ selfies.encoder(selfies.decoder(s)) == s ]`. The paper reports a **5-fold improvement**
from pretraining (`[OURS]`: the two absolute values are not given), and that only **3,978 of 21,457**
screened candidates (18.5%) passed SR.

`[PAPER]` **TP** (Figure S10) = "proportion of candidates with predicted Tg within 500 ± 50 K".

`[PAPER]` **SA score**: replace `[At]`/`[*]` with hydrogens — "effectively treating each polymer as a
monomer" — then score with RDKit. Known training polymers sit "below 6, with most falling within the 2–3
range". `[OURS]` The specific `sascorer` implementation/version is unstated (RDKit ships Ertl's scorer as a
*contrib* module, not core).

`[PAPER]` **Tanimoto similarity**: ECFP6 **2048-bit** fingerprints in RDKit, computed after **connecting the
two ends of the repeat unit into a loop** "to eliminate terminal effects and better approximate infinite
polymer chains".

`[PAPER]` The dielectric screening cascade over the pooled 6,171,066 PV-passing candidates:

| Stage | Criterion | Count | % of pool |
|---|---|---|---|
| 0 | PV-passing candidates | 6,171,066 | 100.0 |
| 1 | Tg > 400 K | 5,017,991 | 81.3 |
| 2 | Eg > 4 eV | 520,803 | 8.4 |
| 3 | ε > 3 | 348,272 | 5.6 |
| 4 | Tm − Tg > 100 K | 177,985 | 2.9 |
| 5 | Td − Tg > 100 K | 168,815 | 2.7 |
| 6 | soluble in H₂O or EtOH | 21,457 | 0.3 |

`[PAPER]` One candidate (a polyamide from glutaryl dichloride + 4,4′-diaminodiphenylmethane) was
synthesized and measured: predicted vs measured Tg 483 / 472 K, Tm 603 / 543 K, Td 643 / 607 K, Eg 4.45 /
4.53 eV (Eg vs HSE06 DFT).

---

## 8. What we can and cannot reproduce

### Reproducible from the paper alone

- The **representation pipeline** (PSMILES → PSELFIES with `[At]` termini), modulo the cleavage ambiguity.
- The **tokenizer structure**: 5 + 100 + 199 + 154 = 458, bracket-atomic SELFIES tokens, predefined symbols.
- The **architecture** exactly: all three size configs, relative position bias, max length 200.
- The **span-corruption objective** exactly as specified, including every stated invariant.
- Both **downstream task formats**, verbatim from the SI examples.
- All **fine-tuning hyperparameters** (epochs, batch 16, AdamW, 3e-4, wd 0.01, beam 4, splits).
- All **evaluation metric definitions** (SV / TSD / DD / PV, SR, TP, SA, Tanimoto).
- The **sampling grid** and the reported best configurations.

### Not reproducible — data

- The 100,145,883-polymer pretraining corpus (withheld for IP).
- The 12,473 experimental polymers.
- Every labeled property dataset, including the 5,130-polymer Tg set used for generation fine-tuning.
- The virtual-polymerization generation procedure (no reaction SMARTS or screening rules published).

### Not reproducible — artifacts

- Weights, tokenizer file, code: none released.

### Not reproducible — training configuration

- Pretraining **learning rate, schedule, warmup, weight decay** — never stated.
- **Dropout** — never stated at any stage.
- Number of pretraining **epochs** — the paper contradicts itself (2 vs 5).
- **Relative position bias** bucket count and max distance.
- **T5 variant** (v1.0 relu vs v1.1 gated-gelu) and `tie_word_embeddings`.
- Mixed precision, gradient accumulation, clipping, seeds, wall-clock.
- The span-corruption **sampling procedure** (only its invariants are given).
- Checkpoint-selection criterion for the downstream tasks.

### Not reproducible — reported numbers

- Per-configuration SV/TSD/DD/PV counts across the 1,800-config sweep exist **only as heatmaps**.
- The absolute SR values behind the "5-fold improvement".
- The provenance of the pooled 6,171,066 candidates (all configs vs optimal only; 300 K vs 500 K mix).

---

## 9. Rules of scientific integrity for this repository

1. Every claim is tagged. `paper reports` / `our reproduction obtains` / `our RLVR extension obtains` are
   three different sentences and are never merged.
2. No number from a run of ours is ever described as "matching the paper" unless the data, tokenizer,
   architecture, training regime, and evaluation procedure are all matched — which, per §8, they cannot
   currently be. Our runs are **method reproductions on substitute data**, not replications.
3. Every unspecified detail gets an entry in [`docs/ambiguity_register.md`](ambiguity_register.md) with the
   choice we made and its justification. Silent invention is a bug.
4. No dataset is fabricated. Where the original data is unavailable we use a **named, licensed, publicly
   downloadable substitute** and say so in the run manifest — see [`docs/data.md`](data.md).
5. The supervised baseline is frozen and evaluated **before** any RLVR work begins, and the RLVR extension
   is documented separately in [`docs/rlvr_plan.md`](rlvr_plan.md). GRPO and RLVR are **not** part of the
   published polyT5 method and must never be described as such.
