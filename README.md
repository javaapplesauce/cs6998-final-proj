# How 2026 foundation models represent the manipulation world — and what that does to control

CS6998 final project, v2. One harness, one benchmark (LIBERO Spatial/Object/Goal), one identical readout head
applied to frozen models from four families, reporting closed-loop success, state probes and null-calibrated
representational similarity. Design: [ARCHITECTURE.md](ARCHITECTURE.md). v1 notebooks: [legacy/](legacy/).

## Setup (Linux, 1× A100/H100 80GB, 16+ vCPUs, ~200 GB disk)

```bash
python -m venv .venv && . .venv/bin/activate          # Python >= 3.12
pip install "lerobot[libero,groot,molmoact2,lingbot_va]==0.6.1" "diffusers==0.39.0" sentencepiece scikit-learn \
    open_clip_torch==3.3.0 git+https://github.com/robbyant/lingbot-vision.git pytest
# Tier-2 anchors only
pip install git+https://github.com/facebookresearch/r3m.git@b2334e726887fa0206962d7984c69c5fb09cceab
pip install --no-deps git+https://github.com/facebookresearch/vggt-omega.git@48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
export MUJOCO_GL=egl OMP_NUM_THREADS=1
huggingface-cli login
```

Why the extras: C-RADIOv4's remote code imports `open_clip`; Cosmos 3 needs diffusers 0.39.0 exactly (inside
lerobot's range); vggt-omega pins numpy<2, which lerobot cannot accept, so it goes in with `--no-deps`.
torchcodec needs system FFmpeg 4–8 built with an AV1 decoder (G1 checks). Disk: MolmoAct2 is a 21.8 GB fp32
download, Wan2.2 ~34 GB (transformer + UMT5 + VAE), V-JEPA 2.1 5.2 GB. A full Tier-1 token cache would be
~445 GB, so `train` deletes each suite's tokens once its heads exist and the driver extracts one suite at a time
(peak: one suite of one model, ≤ 38 GB for the 16×16 ablations). scikit-learn is not pulled in by lerobot.

Day 0: request gated access to `nvidia/Cosmos-Reason2-2B` (GR00T's tokenizer comes from it too),
`facebook/VGGT-Omega`, `facebook/dinov3-vitl16-pretrain-lvd1689m`. `repstudy.policy.libero_setup()` writes
`~/.libero/config.yaml` before the first `import libero.libero` (so nothing prompts on stdin) and pre-downloads
`lerobot/libero-assets` to `~/.cache/libero/assets`, the only place hf-libero 0.1.4 looks (it ignores the
config's `assets` key; without this, 30 env workers race to download).

The Mac/CPU path runs only the smoke test: `pytest tests/test_smoke.py` (G4, ~10 s).

## Run order

```bash
python -m repstudy.run --stage g0      # env: imports, no prompt, assets in place, one EGL frame
python -m repstudy.run --stage g1      # dataset contract + decoder parity (downloads lerobot/libero @ a1aaacb7)
python -m repstudy.run --stage g2      # first-frame pixel + state parity vs the 50 init states
python -m repstudy.run --stage g3      # replayed demos succeed
pytest tests/test_smoke.py             # G4

for i in 1 2; do for s in train eval; do python -m repstudy.run --model blind --stage $s; done; done   # §1.5 rule
for s in g5 calibrate extract train eval; do                                                  # G6 pilot
  python -m repstudy.run --model vjepa2_1 --stage $s --suites libero_spatial --seeds 0; done
python -m repstudy.run --stage g6      # pilot clearly above blind; else edit CONFIG['head'] and redo the pilot
python -m repstudy.run --stage g7      # suggested cut level -> results/gates.json
CUT=0                                  # from G7; persisted in results/cut_level by the first call that passes it

run() {  # one model, tokens one suite at a time
  python -m repstudy.run --model $1 --stage g5 --cut-level $CUT && python -m repstudy.run --model $1 --stage calibrate
  for su in libero_spatial libero_object libero_goal; do
    for s in extract train; do python -m repstudy.run --model $1 --stage $s --suites $su; done; done
  for s in eval probe; do python -m repstudy.run --model $1 --stage $s; done
}
for m in random_vit vjepa2_1 cradio4 qwen3vl_2b cosmos_reason2 groot_n17 lingbot_va wan22 molmoact2 tipsv2 \
         lingbot_vision vjepa2_1_t256 cradio4_t256; do run $m; done            # Tier 1, then §1.5 ablations
python -m repstudy.run --model wan22_512 --stage probe                         # §1.5 resolution check (offline)
for m in qwen3vl_2b cosmos_reason2 groot_n17 molmoact2 lingbot_va wan22; do    # §1.4 secondary, offline
  python -m repstudy.run --model $m --stage calibrate --text instruction; done
for m in lingbot_va wan22; do for su in libero_spatial libero_object libero_goal; do   # secondary, L3 closed loop
  for s in extract train; do python -m repstudy.run --model $m --stage $s --suites $su --text instruction; done; done
  python -m repstudy.run --model $m --stage eval --text instruction; done
for m in vggt_omega cosmos3_reason cosmos3_gen xiaomi_r1 qwen3vl_4b dinov3_l vjepa2_l r3m; do run $m; done  # Tier 2
python -m repstudy.run --stage weightdiff   # CPU-only ladder audit
python -m repstudy.run --stage analyze      # results/summary.md + figures (uses the persisted cut level)
```

Stages skip finished work: calibration is reused once done for the configured suites, heads and evals are keyed
by site, seeds, proprio condition and a hash of the head config (a G6 retune never reuses stale heads), probes
are reused when every site is cached, and cut closed loops (§1.9) skip their extraction and training too.
`run_colab.ipynb` drives the same sequence.

## Outputs

| path | content |
|---|---|
| `results/<model>.json` | gates, calibration curve, selected site, training, per-trial eval, probes, timings |
| `results/heads/<model>/<site>/` | self-contained head checkpoints (normalisation in buffers) |
| `results/gates.json` | G1/G2/G3/G7 reports |
| `results/weightdiff.json` | per-module relative weight change along each ladder |
| `results/summary.md`, `results/analysis.json`, `results/fig_*.png` | final tables, contrasts, similarity, figures |

## Models

Tier 1 runs in this order: blind, random_vit, vjepa2_1, cradio4, qwen3vl_2b, cosmos_reason2, groot_n17,
lingbot_va, wan22, molmoact2, tipsv2, lingbot_vision. Tier 2 (as hours remain): vggt_omega, cosmos3 reasoner
and generator, qwen3vl_4b → xiaomi_r1, dinov3_l, vjepa2_l, r3m. Ablations: vjepa2_1_t256 and cradio4_t256
(16×16 token budget, closed loop), wan22_512 (offline resolution check).

| model | weights | input → native grid | D | sites (1-indexed depth) | prompt reaches tokens? |
|---|---|---|---:|---|---|
| vjepa2_1 / random_vit | vjepa2 hub @204698b + `vjepa2_1_vitl_dist_vitG_384.pt` (`ema_encoder`) | 384² as (B,3,1,·,·), image tokenizer → 24×24 | 1024 | L6 L12 L18 L24 (per-level norms_block) | – |
| cradio4 | nvidia/C-RADIOv4-H @0057b33 (remote code) | [0,1] 512² → 32×32 (−3 summary −7 reg) | 1280 | L8 L16 L24 L32 | – |
| tipsv2 | google/tipsv2-l14 @52847a7 (remote code) | [0,1] 448² → 32×32 (−CLS −1 reg) | 1024 | L6 L12 L18 L24 (norm=False) | – (dual encoder) |
| lingbot_vision | `load_pretrained_backbone(variant="large")` | 512² → 32×32 (−CLS −4 storage) | 1024 | L6 L12 L18 L24 | – |
| qwen3vl_2b / cosmos_reason2 / groot_n17 | Qwen3Backbone(select_layer=16); GR00TN17.backbone | 256² → 16×16 → 2×2 merge → 8×8 | 2048 | L4 L8 L12 L16 (hooks, pre-norm) | no: image before text, causal |
| molmoact2 | allenai/MolmoAct2 @e432d85 via MolmoAct2Policy | 378² → 27×27 ViT → 14×14 | 2560 | L9 L18 L27 L36 (pre ln_f) | no: image before text |
| wan22 / lingbot_va | Wan2.2-TI2V-5B-Diffusers; lerobot/lingbot_va_base | 256² → VAE 16×16 → patch 2 → 8×8 per latent frame | 3072 | b8 b15 b22 b30 × {obs, pred} | yes (cross-attention) |
| vggt_omega (T2) | facebook/VGGT-Omega 1B-512 (gated) | [0,1] 512² → 32×32 | 2048 | A5 A12 A18 A24 (aggregator) | – |
| qwen3vl_4b / xiaomi_r1 / cosmos3_reason (T2) | Qwen3-VL-4B; XR1 `model_states.pt` VLM half; Cosmos3-Nano und tower | 256² → 8×8 | 2560 / 2560 / 4096 | L9 L18 L27 L36 | no |
| cosmos3_gen (T2) | Cosmos3-Nano gen tower (diffusers 0.39) | 5×256² → 2 latent frames × 8×8 | 4096 | {O,P} × 9 18 27 36 | yes |
| dinov3_l / vjepa2_l / r3m (T2 anchors) | gated DINOv3 ViT-L/16; HF vjepa2-vitl-fpc64-256; R3M RN50 | 512² → 32×32; 256² → 16×16; 256² → stage maps | 1024 / 1024 / 256–2048 | L6–L24; L6–L24; layer1–4 | – |

Ladder notes (from source, not the plan text): GR00T keeps LLM layers 0–15 and its action head reads the
pre-norm layer-16 output; LingBot-VA's checkpoint is 839 tensors = 823 of Wan's 825 + Wan's Conv3d patch
embedding re-expressed as a Linear + 14 new action tensors, stored bf16; both L3 rungs share VAE and UMT5 bytes
and read out at the same noise timestep (t=999). Instruction-on is a no-op for prompt-invariant models
(`CONFIG['prompt_invariant']`, asserted by G5), so its closed loop is only run for the L3 pair.
