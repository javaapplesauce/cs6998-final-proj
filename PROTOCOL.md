# Pre-registered protocol

Committed before any production rollout (plan §2.4 step 5). Changing anything below after production starts
must be recorded here with a reason and the commit hash. The head recipe is frozen at G6: the hash printed by
`repstudy.policy.head_tag(CONFIG)` is recorded in `results/gates.json` (G6) and in every eval record.

## Hypotheses and where each is tested

| | hypothesis | primary evidence (`results/summary.md`) | status |
|---|---|---|---|
| H1 | No family wins every suite; the ranking depends on whether the suite stresses layout (Spatial), object identity (Object) or goal-conditional behaviour in a fixed scene (Goal). | per-suite success table with CIs; a family is "winning" a suite if its best model's CI lies above every other family's best | descriptive |
| H2 | Along Qwen3-VL-2B → Cosmos-Reason2 → GR00T N1.7 at LLM layer 16, state decodability and success rise at each rung. | contrasts 1 and 2 (paired, Holm); probe rows at L16 | confirmatory (contrasts), descriptive (probes) |
| H3 | Along Wan2.2 → LingBot-VA, action co-training raises state and action decodability; predicted-future tokens are at least as useful as observed-frame tokens. | contrast 3; probes at the matched block; "Observed vs predicted-future tokens" table (held-out chunk L1, offset R²) | confirmatory (contrast 3), descriptive (rest) |
| H4 | Linear decodability of privileged state, especially the gripper→object offset, predicts success across families. | Spearman link, Tier-1 models only (~11 points; ablations excluded) | exploratory |
| H5 | After null calibration, global CKA shows little cross-family agreement while mutual k-NN stays significant; models cluster by shared ancestry more than by family label. | calibrated CKA / mutual-kNN tables with permutation p; dendrogram + MDS | descriptive |

Ancestry coding for H5 (from source, not marketing): GR00T N1.7 ⊂ Cosmos-Reason2 ⊂ Qwen3-VL-2B; MolmoAct2's LLM
is Qwen3-4B (shares Qwen3 text ancestry with the L1 ladder), vision SigLIP(2) so400m; LingBot-VA ⊂ Wan2.2-TI2V-5B
(shared VAE + UMT5 bytes); C-RADIOv4 distilled from SigLIP2-g + DINOv3-7B + SAM3; V-JEPA 2.1-L distilled from 2.1-G
(no shared init with V-JEPA 2-L); LingBot-Vision-L distilled from a self-supervised ViT-g.

## Planned contrasts (the only confirmatory tests)

Paired over (suite, task, initial state) cells — identical across models by construction — cluster bootstrap
over (suite, task) with 10,000 draws, two-sided p floored at 1/(n+1), Holm across these four:

1. Cosmos-Reason2 − Qwen3-VL-2B at layer 16
2. GR00T N1.7 backbone − Cosmos-Reason2 at layer 16
3. LingBot-VA − Wan2.2 at the matched block (LingBot-VA's calibrated site; both at noise t = 999)
4. mean(non-perception Tier 1) − mean(perception encoders); a model missing at analysis time is named in the row

Everything else is descriptive: per-suite success with 95% CIs, seed spread as SD (three seeds are too few to
bootstrap), probe R², calibrated similarity.

## Rules fixed in advance

- **Text**: every text-capable model gets `"What should the robot do next?"`; instruction-on is a labelled
  secondary condition (offline for all text models; closed loop for the L3 pair only while not cut). For models
  whose image tokens cannot see the prompt (`CONFIG['prompt_invariant']`, asserted by G5) it is reported as
  "equals generic".
- **Readout site**: four depth-matched candidates (eight for the video models), argmin of held-out chunk L1 from
  a brief head on ~10% of episodes per suite, fixed before any rollout and never re-picked once calibrated for
  the production suites. Ladder rungs use matched sites (L1: layer 16; L3: LingBot-VA's index; ablations:
  their base model's).
- **Blind rule**: if the blind head (proprio + task one-hot) exceeds 30% on a suite, proprio is dropped for every
  model on that suite. The rule reads blind's with-proprio result only.
- **Head**: one recipe for every model, tuned only at the G6 pilot on V-JEPA 2.1 / libero_spatial / seed 0, then
  frozen. No per-model tuning, no early stopping, final checkpoint only.
- **Evaluation**: 3 seeds × 10 tasks × 10 initial states per suite; seed s runs initial states 10s…10s+9 for
  every model; 8-step chunks open loop; step caps 280/280/300.
- **Cuts** (§1.9, in this order; set by G7, persisted in `results/cut_level`): Tier 2 → closed loops of
  secondary conditions and ablations → the suite with the highest blind score → Wan2.2 closed loop → the third
  seed. Never cut: blind baseline, random floor, gates.
- **Gates**: G0–G7 must pass before production hours are spent; their reports live in `results/gates.json`
  and per-model `gates` sections.

## Known confounds (stated, not corrected)

GR00T saw egocentric video and actions and was truncated to 16 LLM layers; its own eval crops 0.95 (not applied
here so all L1 rungs see identical input). LingBot-VA was pretrained on three cameras at 256×320 and stored bf16
(bf16 rounding floor reported by the weight-diff audit). Xiaomi-R1 trained with view text before each image;
the image-first prompt here drops it. Cosmos 3 defines a LIBERO action embodiment, suggesting LIBERO data in its
training mix. The frozen head is a measuring instrument, not a deployment recipe: fine-tuned encoders are
reported to score far higher (DEM, arXiv 2609.18374).
