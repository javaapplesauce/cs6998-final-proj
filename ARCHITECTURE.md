# Architecture & Design Document

**Project:** Three-way benchmark of visual representations for robot manipulation, evaluated on R3M (Nair et al., CoRL 2022)'s published behavior-cloning protocol.

**Three baselines:**
- **Baseline 1 — R3M (paper).** ResNet50 with TCN+VLA+L1/L2 contrastive loss on Ego4D 1M steps. Numbers hardcoded from Tables 1-3 of arXiv:2203.12601.
- **Baseline 2 — DINOv3 + LoRA + CLIP.** `baseline_2_dinov3_lora.ipynb`. DINOv3 ViT-B/16 frozen + LoRA r=16 trainable, R3M's exact loss formulation, Ego4D 150k steps (6.7× cut from R3M).
- **Baseline 3 — DINOv3 + SAM 3 (frozen).** `baseline_3_dinov3_sam3.ipynb`. DINOv3 ViT-B/16 fully frozen + SAM 3 concept masking, **no Ego4D pretraining**, BC MLP only.

**The hypothesis:** if Baseline 3 ≥ Baseline 2, then Ego4D pretraining is redundant given DINOv3's strong init plus SAM 3's text-grounded object localization. If Baseline 2 > Baseline 3, the contrastive Ego4D pretraining adds something the foundation models alone don't capture.

**Constraint:** ≤ 4 hours wall-clock per pipeline on a single A100 80GB.

---

## 1. Project overview & comparison contract

R3M is the SOTA general-purpose visual representation for robotic manipulation (as of CoRL 2022). It pretrains a ResNet50 on Ego4D human videos with three loss terms: time-contrastive InfoNCE, video-language alignment via DistilBERT, and L1+L2 regularization on the embedding. It is then frozen and used as a perception module under a small BC policy.

We compare two challengers against R3M's published numbers on R3M's exact BC protocol:

**Baseline 2 hypothesis:** *a foundation visual encoder (DINOv3) with cheap parameter-efficient adaptation (LoRA) can match or exceed R3M's downstream performance using a fraction of the pretrain compute.* R3M trained from scratch on Ego4D for 1M steps; DINOv3 is already trained on ~1.7B images self-supervised. Most of R3M's pretrain budget goes into learning generic visual features that DINOv3 has already learned. We expect 5-10× pretrain speedup with comparable downstream success rate.

**Baseline 3 hypothesis:** *frozen foundation models with object-centric grounding need no Ego4D pretraining at all.* DINOv3's dense features carry spatial structure; SAM 3's text-prompted concept masks zero out distractor regions before global pooling. The two foundation models stay frozen; only a 3-layer MLP policy head trains on the demos. This isolates whether contrastive Ego4D pretraining (Baseline 2) is necessary or just a path-dependent legacy of R3M's 2022 design.

The comparison is **strict apples-to-apples on the downstream side**: identical demo data, identical optimizer (Adam, lr 1e-3, batch 32, 20k steps, eval every 1k), identical environments and demo sizes, 3 seeds. Differences live in the visual encoder and (for B3) the policy MLP shape.

---

## 2. Why this architecture (and not the alternatives)

### 2.1 Why DINOv3 ViT-B/16 over R3M's ResNet50 or DINOv2?

| Property | R3M ResNet50 | DINOv3 ViT-B/16 (ours) |
|---|---|---|
| Params | 23 M | 86 M |
| Pretrain data | Ego4D only (~5M clips) | LVD-1689M (~1.7B curated images) |
| Pretrain objective | Time-contrastive + VLA + L1/L2 | DINO + iBOT + Gram Anchoring (DINOv3) |
| Effective receptive field | local conv stack | global attention |
| Released features | 2048-dim global pool | 768-dim CLS + 196 patch tokens |

**Pros of DINOv3 (over DINOv2):**
- Trained on **12× more data** than DINOv2 (1.7B vs 142M images). Released August 2025.
- Gram Anchoring objective specifically prevents dense-feature collapse — patch tokens stay sharp and semantically distinct, which directly benefits Baseline 3's masking and is generally good for fine-grained manipulation perception even in Baseline 2's CLS-only path.
- Modern attention block, same `qkv/proj/fc1/fc2` structure as DINOv2 → LoRA injection logic is identical.

**Cons of DINOv3:**
- **Gated weights** (request access at https://ai.meta.com/dino-v3 → URL-emailed weights, or `hf auth login` for HuggingFace mirror).
- /16 patch (vs /14 in DINOv2) — 196 patches at 224 instead of 256, ~24% fewer tokens. Less spatial detail but faster forward.
- Pretrain distribution (curated web images) has a larger domain gap to egocentric Ego4D than R3M's in-domain training. Whether this matters for manipulation transfer is exactly what we're testing.

**Alternatives considered and rejected:**
- **DINOv3 ViT-L/16** (300 M) — better features but 3-4× compute. Reserved as a CONFIG flag for users with more budget. Default to ViT-B/16 for like-for-like comparison with R3M's 23M-param ResNet50 (already 4× larger).
- **DINOv3 ViT-S/16** — 21M, would fit easily in budget but loses the headline.
- **DINOv2 ViT-B/14** — original baseline; superseded by DINOv3 (12× more pretrain data + Gram Anchoring) and so used only as a fallback.
- **MAE ViT-B/16** — strong but pretrains via reconstruction. R3M's head-to-head against MVP (MAE on Ego-Soup) showed MAE losing by ~10% on Franka.
- **CLIP ViT-B visual** — already strong on object semantics but loses spatial grounding. R3M showed CLIP underperforming R3M (~42% vs 53% on Franka). We use CLIP for text only.

### 2.2 Why LoRA r=16 instead of full fine-tuning or no adaptation?

**The core question:** if DINOv2 is so good, why adapt it at all?

Answer: R3M's loss formulation requires the encoder to produce **temporally-aware** embeddings (TCN loss) and **language-aligned** embeddings (VLA loss). DINOv2's pretraining objectives are spatial/instance discrimination — there is no notion of time or language in DINO/iBOT. So if we freeze DINOv2 entirely, the TCN and VLA losses can only flow through the language head, not back into the visual representation. The visual representation never learns the structure R3M's downstream protocol expects to find in the embedding.

LoRA r=16 strikes the best balance:

| Adaptation | Trainable params | Pros | Cons |
|---|---|---|---|
| **Frozen** (no adaptation) | 0 | fastest; minimal risk of degrading DINOv2 | TCN/VLA gradients can't reach visual features; representation never becomes time-aware |
| **Linear probe** (one Linear after backbone) | ~590K | trivial to train | doesn't add temporal/semantic structure to features themselves, only re-projects them |
| **Adapters (Houlsby/Pfeiffer)** | ~1-3M | mature, well-studied | extra forward latency; more invasive code change |
| **LoRA r=8** | ~1.2M | minimal extra fwd cost | underparameterized for adapting 86M-param model? |
| **LoRA r=16** ← chosen | ~2.4M | 2-3% of backbone, enough capacity for adaptation | more memory than r=8 |
| **LoRA r=32** | ~4.7M | more capacity | diminishing returns; closer to "fine-tune" failure modes |
| **Full fine-tune** | 86M | maximum flexibility | optimizer state, gradients, save sizes all explode; risk of catastrophic forgetting; 4h budget likely impossible |

**Pros of LoRA r=16:**
- Optimizer state (Adam first/second moments) for 2.4M params is ~30 MB instead of ~1 GB for full FT.
- LoRA-B initialized to zero → at step 0 the model is **bit-identical** to frozen DINOv2; adaptation is purely additive.
- Adapter-only checkpoints are tiny (~10 MB) — easy to share, easy to ablate.
- Doesn't risk catastrophic forgetting of DINOv2's spatial features.

**Cons of LoRA r=16:**
- Forward pass still pays full backbone cost (LoRA adds a small overhead, doesn't save compute).
- Adaptation is restricted to per-block low-rank structure — if the optimal adaptation is high-rank in some block, LoRA can't represent it.
- LoRA-A receives **zero gradient on step 0** (because dL/dA ∝ B and B starts at zero). First few steps update only B; A unlocks after.

**Alternatives in the LoRA family considered:**
- **LoRA on attention only** (`qkv` + `proj`) — saves ~half the trainable params; rejected because R3M's losses care about token-level features and MLP layers do most of the channel-mixing.
- **DoRA** — magnitude/direction decomposition; modest gains over LoRA in some benchmarks but more code complexity.
- **(IA)³ / prompt tuning** — even smaller param count but generally weaker.

LoRA r=16 on `qkv/proj/fc1/fc2` (4 modules × 12 blocks = 48 sites) is the well-trodden middle ground.

### 2.3 Why CLIP ViT-B/32-quickgelu for VLA text instead of DistilBERT?

R3M used DistilBERT (768-dim) for the language side of the VLA loss. We use CLIP ViT-B/32 text (512-dim).

**Pros of CLIP text:**
- Text features are pretrained to be **vision-aligned** (contrastive on 400M image-text pairs). This is a substantially better prior for a video-language alignment loss than DistilBERT, which was pretrained on text-only masked language modeling and has no notion of visual concepts.
- CLIP's text tower is small and fast — encoding ~5M unique Ego4D narrations takes 3-5 minutes once.
- The 512-dim is small → smaller language head, less memory.

**Cons of CLIP text:**
- The text representation is biased toward CLIP's training distribution (mostly web alt-text). Ego4D narrations are short, declarative, and observational ("the person picks up the apple"); CLIP handles these well but the bias direction is worth flagging.
- Smaller dimensionality could in principle limit expressiveness — but we observe no degradation in our smoke runs.
- Different from R3M, so head-to-head comparison technically conflates "text encoder choice" with "everything else." We accept this because the text encoder is part of the architectural change being evaluated.

**Critical implementation note:** `pretrained='openai'` weights require **QuickGELU** activation, but `open_clip`'s default `'ViT-B-32'` uses standard GELU. Loading openai weights into a non-QuickGELU model produces a silent **0.35 max-diff** in text embeddings that materially degrades downstream VLA quality. **Must use `'ViT-B-32-quickgelu'`.** This was caught during static review and verified empirically.

**Alternatives considered:**
- **DistilBERT** (R3M's choice) — slower per token, no visual prior. Strict apples-to-apples on the language side but harder to pretrain quickly.
- **Sentence-BERT / MPNet** — purpose-built for sentence embedding but no visual alignment.
- **CLIP ViT-L/14 text** — better but heavier. ViT-B/32 is the right cost point given this is text-only and cached.

---

### 2.4 Baseline 3-specific decisions

**Why fully frozen, no LoRA?**
The hypothesis Baseline 3 tests is "frozen foundation models suffice." LoRA would conflate "did adaptation help?" with "did the frozen foundation help?" — exactly Baseline 2's question. Baseline 3 must keep both DINOv3 and SAM 3 frozen for the comparison to be meaningful.

**Why dense feature map (h × w × D) instead of CLS token (D)?**
SAM 3's mask is spatial. Multiplying a (14×14) mask against a CLS-pooled vector is meaningless. The whole point of Baseline 3's design is to mask **before** pooling — so we use `get_intermediate_layers(reshape=True)` to get `(B, D, h, w)`, apply the mask, then GAP. DINOv3's Gram Anchoring is what makes this practical: in DINOv2, dense features can collapse (all patches look similar after fine-tuning); in DINOv3 they stay distinct.

**Why SAM 3 once per episode at eval time?**
Real-time SAM at every rollout step would be 50ms × 200 horizons × 10 episodes × 20 evals × 100+ BC runs ≈ hours per BC run. The proposal's "concept-aware mask at every step" is computationally infeasible in our 4 h budget. Compromise: SAM runs once at episode start; mask is reused across the episode. Configurable refresh via `CONFIG['sam_refresh_steps']`. This is a mild approximation — for static target objects (most of the R3M task suite), the mask stays accurate for the episode duration.

**Why 3-layer MLP + LayerNorm instead of B2's 2-layer + BatchNorm?**
This is the proposal's exact spec (proposal §3.1). LayerNorm is appropriate for non-batch-correlated features; BatchNorm at input was R3M's choice because their CNN features were less normalized. Frozen DINOv3 features are already L2-bounded by the final LayerNorm in the backbone, so BatchNorm at input has less utility — LayerNorm in the MLP body is more standard for this regime.

**Why empty-mask fallback?**
If SAM 3 returns no instances for a given prompt, `mask.sum() == 0` and `feat ⊙ mask` is all zeros, killing all signal. Baseline 3's `masked_feature_pool` falls back to unmasked GAP when the mask is empty — keeping the policy operational even when the segmenter fails. Logged as a degraded-mode warning (no explicit log yet — TODO if patterns emerge).

**Why same demo data as Baseline 2?**
The proposal originally specifies "~50 teleoperated demonstrations" in Isaac Gym/Robosuite. We deviate: Baseline 3 uses R3M's pre-existing demo `.pkl` files in mj_envs / MetaWorld / Adroit. Reasoning: (a) avoids 30+ hours of teleoperation collection, (b) gives us a shared demo set across all three baselines for direct comparison, (c) tests the proposal's architecture on a more demanding benchmark than 50 demos × 2 tasks. The cost is moving away from the proposal's claimed eval task list (which is its prerogative, not a strict requirement).


## 3. High-level architecture

```
                    ┌─ Ego4D pre-extracted frames + narrations ─┐
                    │                                            │
                    │       (offline, not in 4h budget)          │
                    └──────────────┬─────────────────────────────┘
                                   │
                                   v
       ┌───────────────────────────────────────────────────────┐
       │  PRETRAIN  (~2 h on A100)                             │
       │                                                       │
       │   ┌─────────┐   forward 80 imgs/step                  │
       │   │ DINOv2  │   ─────────────────►                    │
       │   │ ViT-B/14│   (B,5,3,224,224) → (B,5,768)           │
       │   │ frozen  │                                         │
       │   │ +LoRA   │      ┌──┐ TCN loss   (Eq 1)             │
       │   │  r=16   │ ───► ├──┤ VLA loss   (Eq 2 + A.3)       │
       │   └─────────┘      └──┘ L1/L2 reg                     │
       │       ▲                  │                            │
       │       │                  v                            │
       │   ┌───┴────┐         optimizer (Adam, lr=1e-4)        │
       │   │ CLIP   │         updates LoRA + lang_head only    │
       │   │  text  │                                          │
       │   │ cached │                                          │
       │   └────────┘                                          │
       │                                                       │
       │   output: cache/pretrain.pt (LoRA + lang_head only,   │
       │                              ~10 MB)                  │
       └───────────────────────────────────────────────────────┘
                                   │
                                   v
       ┌───────────────────────────────────────────────────────┐
       │  FEATURE EXTRACTION  (~10-15 min)                     │
       │                                                       │
       │   For every (env, task, view):                        │
       │     load demo episodes (images, actions, proprio)     │
       │     encode all frames once with DINOv2+LoRA           │
       │     save (T, 768) tensor + actions + proprio to disk  │
       │                                                       │
       │   output: cache/features/<env>/<task>__<view>.pt      │
       └───────────────────────────────────────────────────────┘
                                   │
                                   v
       ┌───────────────────────────────────────────────────────┐
       │  BC SWEEP  (~60-80 min)                               │
       │                                                       │
       │   For env × task × view × demo_size × seed:           │
       │     load cached features                              │
       │     train 2-layer MLP+BatchNorm policy (R3M's exact)  │
       │       lr 1e-3, bs 32, 20k steps                       │
       │     every 1k steps, eval N rollouts in real sim       │
       │     plateau early-stop after 3 non-improving evals    │
       │     record best success rate                          │
       │                                                       │
       │   output: results.json with full sweep history        │
       └───────────────────────────────────────────────────────┘
                                   │
                                   v
       ┌───────────────────────────────────────────────────────┐
       │  AGGREGATE  (~2 min)                                  │
       │   summarize seeds → per-(env,task,view,demo) means    │
       │   delta vs R3M baseline (paper Tables 1-3)            │
       │   side-by-side bars, per-task bars, training curves   │
       │   auto-generated summary.md                           │
       └───────────────────────────────────────────────────────┘
```

The three-stage structure (pretrain → feature extraction → BC) is deliberate: every stage is independently re-runnable, and the expensive stages (pretrain, feature extraction) cache their outputs to disk. Iterating on BC hyperparameters costs ~80 min; iterating on pretrain hyperparameters requires deleting the cache and starting over.

---

## 4. Component-level decisions

### 4.1 The CONFIG dictionary as single source of truth

Every hyperparameter — paths, model dims, LoRA rank, λ values, BC protocol, env list, seeds, cut levels — lives in one `CONFIG` dict at the top of the notebook. Functions take `cfg` as their first argument; **no magic numbers are buried inside function bodies.**

**Pros:**
- Iteration is trivial: change one cell, re-run.
- Self-documenting: a reader sees every knob at once.
- `cut_level` is part of CONFIG, so degrading scope when running long is a single integer change.
- Saved into `results.json`, so every run record is fully reproducible.

**Cons:**
- Functions need explicit `cfg` plumbing.
- Easier to typo a key than a positional arg, but this is mitigated by Python's dict access raising `KeyError` cleanly.

**Alternatives considered:**
- **dataclass** — slightly more rigorous; rejected because flat-dict CONFIG is more conducive to running experiments where you toggle one thing without re-defining a class.
- **Hydra/OmegaConf** — overkill for a single notebook.
- **YAML file** — splits the notebook from its config; we want one self-contained file.

### 4.2 The `Results` logger (dual stdout + results.json)

**Pros:**
- Atomic writes (write tmp, rename) — no partial-file corruption if the kernel dies mid-write.
- Single ground-truth file for the morning presentation; the markdown summary, the plots, and any post-hoc analysis all read from this.
- Both append (for time-series like loss curves) and log (for one-shots like baselines, summaries).

**Cons:**
- Re-running the notebook reads the existing JSON and appends — running multiple experiments to the same `results_dir` can produce hybrid records. Mitigated by changing `CONFIG['results_dir']` per experiment.

**Alternatives considered:**
- **Weights & Biases** — fantastic for cluster runs but adds a network dep, and the user wanted a self-contained file output.
- **TensorBoard** — fine for scalars, awkward for nested structured records.
- **CSV** — flat structure poorly suited to per-run history.

### 4.3 R3M baseline as hardcoded table vs. on-demand fetch

**Pros of hardcoded:**
- No network dependency, no future PDF-parsing fragility.
- Numbers are versioned with the notebook — git diff shows any accidental changes.
- Available even if running offline.

**Cons:**
- Per-task and per-view numbers from Figs 4/8/9 of the paper are read off bar charts, **not numerically printed.** We mark these `'_approx'` and recommend overriding from R3M's actual eval JSONs in `facebookresearch/r3m/evaluation/results/` if available. Aggregates (Table 1-3) are paper-exact.

### 4.4 Ego4D dataset: pre-extracted JPEG cache vs. live video decoding

The training pipeline reads from `<clip_uid>/{0..4}.jpg` + `manifest.parquet` — a layout produced by the in-notebook `preprocess_ego4d` function (cells P0-P3). Live video decoding from `.mp4` would be **~100× slower** at the I/O level: each `__getitem__` would seek into a video file, decode N frames, and copy them out. Pre-extracted frames let the kernel cache them and let the OS prefetch sequentially.

**Why we ship our own preprocessing instead of running R3M's:** R3M's repo doesn't include a preprocessing script — their README explicitly states "the dataset has been parsed into frames" and assumes the user has done it. Our `preprocess_ego4d` function reads `narrations.json` (Ego4D's annotation file), filters to `#C` (camera-wearer) narrations, carves a `±clip_window_sec` window around each narration's `timestamp_sec`, and applies R3M's exact 5-frame sampling rule. The output schema (column names, frame layout, JPEG quality) matches what `Ego4DR3MDataset` expects.

**Decoder choice:** the function tries `decord` (single-pass, random access) first; falls back to OpenCV (per-frame seek) if decord isn't available. decord is the recommended path on Linux/Colab (`pip install decord`); OpenCV is the portability fallback for arm64 Mac and other platforms without decord wheels. OpenCV is ~3× slower for random-access decoding because each `cap.set(CAP_PROP_POS_FRAMES, idx)` re-seeks rather than streaming, but it works everywhere.

**Idempotency:** the function checks `manifest.parquet` and `<clip_uid>/4.jpg` before reprocessing each clip. Killing and resuming is safe; a Colab disconnect mid-preprocessing only loses the current 500-clip checkpoint window (set by `log_every`).

**Pros of pre-extracted cache:**
- Fast random access; persistent across runs.
- Frame-level operations (resize, crop) are deterministic.
- Disk usage is small relative to source video (5 frames vs. ~30s video).

**Cons:**
- Frame sampling strategy is **frozen at extraction time.** R3M sampled 5 frames per clip (I0 from first 20%, Ig from last 20%, Ii<Ij<Ik in between). If we wanted different sampling, we'd need to re-extract.
- ~50-100 GB of JPEGs.

**Mitigation:** if `n_frames > 5`, the dataset re-samples per epoch using R3M's sampling rule. If exactly 5 frames are cached (R3M's original layout), we use them as-is. Both paths supported.

### 4.5 Video-level random crop vs. per-frame augmentation

**R3M's exact rule:** within a video, the **same** crop is applied to all 5 frames. This is critical because:
- The TCN loss says "frames closer in time should be closer in embedding space." If each frame has a different random crop, the loss gets a free ride: even two frames at the same timestep would be far apart in embedding due to crop alone, polluting the temporal signal.
- The VLA loss says "the score should increase over the video." Per-frame crops would make the score depend on crop, not on temporal progress.

**Pros:**
- Faithful to R3M.
- Forces the encoder to learn temporal structure from actual content change, not crop variance.

**Cons:**
- Less data diversity than per-frame crops would give. But the goal is structure preservation, not maximal augmentation.

We implement this in `_load_with_video_crop` by computing the random crop parameters once per call to `__getitem__` and applying the **same** `(x, y, w, h)` to all 5 PIL images before resizing.

### 4.6 CLIP text embedding pre-cache vs. on-line encoding

We compute CLIP text embeds for all unique narrations **once**, save to a single `(N, 512)` tensor, and look up by index in the inner training loop.

**Pros:**
- Removes the text encoder entirely from the inner loop. Even a small text encoder costs ~20 ms/step at batch 16; over 150k steps that's **50 minutes of pure waste.**
- Deduplication: many Ego4D clips share narrations; encoding each once is a 5-10× saving.
- The CLIP model can be **dropped from memory** after caching, freeing ~600 MB.

**Cons:**
- Locks in the text encoder at cache time. If we want to ablate the text encoder, we re-cache.
- Cache file size: `5M × 512 × 4 bytes ≈ 10 GB` if all narrations are unique. Practically less due to dedup.

**Mitigation:** the cache loader short-circuits if the file exists — re-runs are instant.

### 4.7 Custom 30-line LoRA vs. peft library

We ship a 30-line `LoRALinear` rather than depending on `peft`.

**Pros of custom:**
- One module class, one injection function, no library version pinning.
- Direct control over which `nn.Linear` modules get adapted.
- No dispatch overhead.
- Trivial to debug — the LoRA forward is literally `base(x) + (x @ A.T @ B.T) * scaling`.

**Cons of custom:**
- We re-implement well-known machinery.
- No support for QLoRA, DoRA, etc. (not needed here).

**Pros of peft:**
- Battle-tested, supports many adapter types.
- Dense ecosystem (gradient checkpointing, save_adapter, etc.).

**Cons of peft:**
- Module-name targeting can be finicky for hub-loaded models that aren't in HuggingFace's transformers registry. DINOv2 from torch.hub falls outside peft's expected nomenclature, requiring manual `target_modules` strings anyway. Once you're naming modules manually, the value of peft drops.

For this notebook, the custom LoRA wins on simplicity + transparency.

### 4.8 LoRA injection target choice (`qkv`, `proj`, `fc1`, `fc2`)

We replace 4 linear modules per ViT block × 12 blocks = **48 sites total**, ~2.4M trainable params on a 86M-param backbone (~2.7%).

**Why these 4:**
- `qkv` and `proj` are the attention projections — the place where DINOv2's "what attends to what" structure lives. Adapting these lets us re-shape attention to attend to motion-relevant cues.
- `fc1` and `fc2` are the MLP projections — channel mixing. The MLP block carries 2/3 of parameter mass; ignoring it would severely under-adapt.

**Alternatives considered:**
- **Attention-only** — saves half the params; worth trying as an ablation but a priori we expect the MLP adaptation to matter.
- **Only the last K blocks** — limits to top-of-network adaptation; saves params but may starve early-block features of temporal structure. Typical for classification heads, not for representation modulation.
- **Patch embed + first norm** — too localized; would just re-tokenize.

### 4.9 Loss formulation (preserved exactly from R3M)

**Time-contrastive (Eq 1):** for triplet (zi, zj, zk) with i<j<k, InfoNCE with positive (zi, zj), hard negative (zi, zk), and 3 cross-video negatives. Similarity = **negative L2 distance**, not cosine — R3M's choice, preserved.

**Why neg-L2 and not cosine?** Negative L2 is sensitive to embedding **magnitude**, which together with the L2 regularization gives R3M an implicit "stay near the origin" pressure. Cosine would normalize away magnitude and lose this. We preserve to keep behavior comparable.

**Video-language alignment (Eq 2 + Appendix A.3):** three positive/negative pair sets per element:
- (z0, zg) vs (z0, z0)
- (z0, zj) vs (z0, zi)
- (z0, zk) vs (z0, zj)

Plus a wrong-language negative (z0, z_pos, l_wrong). The loss is the mean of the three sets.

**Why three pair sets?** Per Appendix A.3: "Using a larger number of positive examples from a single video and multiple negative examples from different videos stabilizes training." It's effectively curriculum: the (z0, zi) negative is "early" frame; (z0, zj), (z0, zk) progressively further. If the model can distinguish all three, it has learned a temporally-smooth score.

**Cross-video wrong-language negative:** we use `l_wrong = l_embed[(arange + 1) % B]` — a guaranteed cyclic-shift derangement. The original code used a re-roll loop until no fixed point, which works in practice (mean 2.7 retries at B=16) but has unbounded worst case. The cyclic shift gives the same statistical effect (every wrong-l_b has matching l_(b+1) which is independent of the right answer) without retries.

**Regularization:** `λ3·||z||₁ + λ4·||z||²` with `λ3=λ4=1e-5` (paper exact). Applied to (zi, zj, zk) middle frames.

**Pros of preserving the loss:**
- Apples-to-apples — any improvement is attributable to architecture, not loss tuning.
- R3M validated these λ values empirically; we don't have budget to re-sweep.

**Cons:**
- We may be leaving improvement on the table — DINOv2 features are already L2-bounded by LayerNorm, so the L2 penalty's role differs from R3M's case where the embedding could grow unboundedly. A future ablation should remove the L2 reg.

### 4.10 Pretrain budget: 150k steps (vs. R3M's 1M)

This is the most consequential design decision.

**The hypothesis:** 150k is enough because DINOv2 starts from a strong representation. We're adapting, not training from scratch.

**Why 150k specifically?**
- Time budget: 150k × 80 ms/step ≈ 200 min ≈ 3.3 hours of pretrain alone, leaving 30-50 min for downstream eval. Above 150k, the budget breaks.
- The MAE-init / DINO-init literature shows ~10× pretrain speedups at fine-tuning time; 6.7× speedup (1M → 150k) is a defensible point inside that range.
- LoRA adapters + lang head together are ~7.6M params. With Adam, that converges very fast.

**Pros:**
- Fits the budget. 200k or 250k would be marginal.
- The optimizer has fewer parameters to fit, so per-step it learns faster.

**Cons:**
- We can't verify in advance that 150k is enough. The training curves might show loss still declining at 150k. If so, the comparison underrates DINOv2+LoRA — but that's a *worst-case* interpretation; if our 150k checkpoint already matches R3M's 1M ResNet50, the speedup story holds.
- R3M's released model (1.5M steps) outperforms its 1M ablation. We are comparing against the 1M variant.

**Mitigation:** the cut-level ladder protects us; if anything else goes long, we cut other knobs first and keep pretrain at 150k. Pretrain reduction is `cut_level=5` — last resort.

### 4.11 Behavior cloning: 2-layer MLP + BatchNorm (R3M-exact)

Hidden sizes [256, 256], BatchNorm at the input, ReLU between layers, MSE loss to expert action. lr 1e-3, batch 32, 20k steps, eval every 1k.

**Pros of preserving exactly:**
- The single greatest risk to a representation-comparison study is **policy hyperparameter drift.** If we used a slightly bigger MLP or a different optimizer, "wins" could be due to policy capacity rather than features. Strict identity on the policy isolates the comparison.
- BatchNorm at the input acts as feature whitening — important when feature distributions differ across encoders.

**Cons:**
- The R3M policy was tuned for R3M's 2048-dim ResNet features; DINOv2's 768-dim features may benefit from a different MLP shape. We don't get to tune.
- 20k steps may be too many for some envs and overfit; too few for others. R3M's choice was a one-size-fits-all.

### 4.12 Eval: 10 episodes per checkpoint vs. R3M's 25

R3M reports 25 episodes per eval × 20 eval rounds × N runs = **lots of MuJoCo time.** Our budget can't sustain 25; we use 10. Std error scales as 1/√N, so we move from ~3 pp std error to ~5 pp.

**Pros:**
- 2.5× speedup on the BC sweep.
- ±5 pp is acceptable for an A/B between methods that differ by >5 pp (which is the regime we expect — anything smaller than 5 pp should be reported as "indistinguishable").

**Cons:**
- Wider error bars. If the actual delta is 2-3 pp, our test is underpowered to call it.
- The reported numbers are noisier than R3M's; we should explicitly note this when comparing.

### 4.13 Plateau early-stop (3 non-improving evals)

Cuts off BC training when success rate hasn't improved for 3 consecutive evals (3000 BC steps).

**Pros:**
- Saves ~30-40% of BC steps in practice — most envs plateau around step 8-12k of 20k.
- Reduces overfitting risk on tiny demo sets (5 demos × short horizon).

**Cons:**
- A noisy eval can trigger early-stop on a model that would have improved later. 3-eval patience mitigates but doesn't eliminate.
- "Best success" reported is the best ever seen, so early-stop only saves time, doesn't change the reported number.

### 4.14 Pre-extract features once per (task, view) instead of running encoder in BC inner loop

This is the single biggest BC-time speedup.

**Without pre-extraction:** every BC step samples (image, action) pairs; the encoder runs `bs=32` images per gradient step × 20k steps × ~300 BC runs = **~190 million encoder forwards** for the whole sweep. Each forward is ~3 ms on A100 → 9.5 hours.

**With pre-extraction:** per (task, view), encoder runs once over **all** demo frames (~10k frames per task) → ~30s. 12 task-view combinations × 30s ≈ 6 min. Then BC inner loop is just MLP forward over cached tensors — 0.3 ms/step.

**Speedup:** ~95× on the BC encoder cost.

**Pros:**
- BC sweep is now MLP-bound, not encoder-bound. The whole sweep fits in ~80 min.
- Identical to R3M's protocol — they also freeze the encoder at downstream time.

**Cons:**
- Encoder is fixed at the post-pretrain LoRA weights for the entire BC sweep. Can't adapt encoder during BC. This is desired for a clean comparison but worth flagging.
- Cache files are large (~50 MB per task-view).

### 4.15 Cut-level ladder

```python
cut_level=0: full sweep (Franka + MetaWorld + Adroit, 3 views, 3 demo sizes, 3 seeds)
cut_level=1: drop Adroit                                 ~10 min saved
cut_level=2: demo sizes 3 → 2                            ~33% saved
cut_level=3: seeds 3 → 2                                 ~33% saved
cut_level=4: drop view3                                  ~33% saved
cut_level=5: pretrain 150k → 100k                        ~45 min saved
```

Order is intentional:
1. **Adroit first** — only 2 tasks; loses least information.
2. **Demo sizes** before seeds because R3M's headline numbers average over demo sizes anyway.
3. **Seeds** before views because seed variance is well-understood; view variance is part of R3M's robustness story.
4. **Views** last among BC cuts.
5. **Pretrain reduction last** — directly hurts comparison fairness.

---

## 5. Per-cell walkthrough

### Cell 1-3 (markdown intro)
Sets the comparison contract upfront. The bottleneck table forces the reader to confront the wall-clock budget before reading any code. The setup cell (cell 3) has the pip install line for first-time users.

### Cell 4 (CONFIG + setup)
- Imports stdlib, torch, DataLoader.
- Builds CONFIG with every hyperparameter named explicitly.
- Calls `apply_cuts(CONFIG)` so the config the rest of the notebook sees is post-cut.
- Sets `tf32` flags and `set_float32_matmul_precision('high')` — costs nothing, gains 10-30% on matmul-heavy ops.
- Picks `DEVICE` and `DTYPE` from precision setting.

**Decision: bf16 over fp16.** bf16 has the same exponent range as fp32 — no `GradScaler` needed, no Inf/NaN dance. Dynamic-range issues that plague fp16 contrastive losses (TCN's neg-L2 over many samples can produce large gradients) just disappear with bf16. A100 has full bf16 throughput.

### Cells 5-6 (Colab / Drive setup, optional)
A single optional cell that auto-detects Colab, mounts Drive, clones the project repo into `/content/` (small-file I/O is much faster on `/content/` than on Drive), checks dependencies, and **patches** `CONFIG` paths to point at Drive — so caches and the 2-hour pretrain checkpoint survive Colab session disconnects. The cell is a no-op on non-Colab boxes where `DRIVE_ROOT` doesn't exist; the user keeps Cell 4's defaults. `CONFIG.update(...)` is used (not reassignment), so the 60+ hyperparameters from Cell 4 are preserved.

### Cells 7-11 (Ego4D preprocessing, optional, gated)
R3M's repo doesn't ship a preprocessing script — their README explicitly states the dataset is assumed to be already parsed into frames. We replicate equivalent behavior:
- **P0** (cell 8): clones R3M's repo for reference. Their `r3m/utils/data_loaders.py` is the source-of-truth on the 5-frame sampling rule. The clone is informational only.
- **P1** (cell 9): defines `preprocess_ego4d(...)`. Reads `narrations.json`, filters to `#C` (camera-wearer) narrations, opens videos with `decord` (fallback: OpenCV per-frame seek), carves `±clip_window_sec` windows around each `timestamp_sec`, applies R3M's 5-frame sampling rule (I0 from first 20%, Ig from last 20%, Ii<Ij<Ik in between), resizes to 224×224, writes JPEGs and `manifest.parquet`.
- **P2** (cell 10): gated by `RUN_PREPROCESSING = False` (default). When set to True, runs preprocessing on the first `subset_n` narrations (default 50,000 — full Ego4D is 1-2 weeks of single-process work).
- **P3** (cell 11): validates `manifest.parquet` columns + spot-checks frames on disk. Soft-warn (not hard fail) if data is missing — surfaces path issues before they block pretrain.

The `RUN_PREPROCESSING=False` default means re-running the notebook after preprocessing has already happened is a no-op. P0's R3M clone happens unconditionally (it's tiny and useful as reference). Idempotency: P2 checks `manifest.parquet` and `<clip_uid>/4.jpg` before reprocessing each clip; killing and resuming preprocessing is safe.

### Cell 5 (logger)
- Atomic JSON writes via `tmp` + `replace`.
- Module-level `log(msg)` is the only print path used elsewhere — keeps logging consistent.

### Cell 6 (R3M baseline)
- Hardcoded; logged into `results.json` so the delta-vs-baseline cell has a single source.
- Per-task / per-view / per-demo-size flagged `_approx` because they're read off paper bars.

### Cell 7 (Ego4D dataset class)
- `_sample_indices` handles short clips robustly (n>=5 always works, even degenerate cases).
- `_load_with_video_crop` reads all 5 PIL images first, then applies one shared crop — order matters here because PIL's `crop` is cheap but recreating PIL images would be slow.
- `_ego4d_collate` is a **module-level function** (not a lambda) — required because Linux fork is fine but Mac/Windows spawn-based multiprocessing cannot pickle lambdas.

### Cell 8 (CLIP text cache)
- Uses `'ViT-B-32-quickgelu'` to match openai weights' QuickGELU activation. Without `-quickgelu`, max embedding diff is 0.35 — silent corruption.
- Cache file is keyed by narration string → integer index. Inner loop does `embeds[batch_idx]` — a single tensor index op.

### Cell 9 (LoRA)
- `LoRALinear`: `lora_B` initialized to zero so step-0 output is identical to frozen DINOv2.
- `inject_lora` walks `named_modules` once, replaces matching `nn.Linear` children. The `isinstance` check on `nn.Linear` ensures we skip Conv2d (e.g. `patch_embed.proj` is a Conv2d, not a Linear, so it's not adapted — correct behavior).

### Cell 10 (Model)
- `forward_pretrain(frames)` reshapes (B, 5, 3, H, W) → (B*5, 3, H, W), runs the backbone once, splits the features back into the 5 named tensors.
- The 5-layer language head exactly matches Appendix A.2's spec (just with input dim 2*768+512=2048 instead of 2*2048+768=4864 due to our different feature dims).

### Cell 11 (losses)
- TCN, VLA, regularization functions take `out` dict + scalars; no global state.
- VLA cyclic-shift derangement is stateless and faster than rejection sampling.

### Cell 12 (pretrain function)
- `make_model_and_optim`: try `torch.compile`, fall back to eager if it fails. Compile of LoRA-wrapped attention works in PyTorch ≥ 2.3.
- Optimizer takes only `[p for p in parameters() if p.requires_grad]` — i.e. LoRA + lang head, ~2.4M + ~5.2M = ~7.6M params.
- Saves only `lora_*` and `lang_head*` keys — checkpoint is ~10 MB instead of ~330 MB.

### Cell 13 (feature extraction)
- `load_pretrained_model` loads with `strict=False` — base DINOv2 weights come from `torch.hub`, not the checkpoint, so the checkpoint only contains the trainable subset.
- `encode_images` does on-the-fly resize when input shape doesn't match `image_size` — important for demos that may have been collected at different resolutions.

### Cell 14 (BC dataset + policy)
- `FeatureBCDataset` concatenates feat + proprio at `__getitem__` time. Could pre-concatenate at `__init__` but per-step is cheap enough.
- `BCPolicy` uses `BatchNorm1d` at the input — exact R3M Appendix B.5.

### Cell 15 (sim env wrappers + rollout)
- `make_env` has `# TODO: confirm` flags for camera names per view and proprio composition. These are env-version-specific and will surface as named exceptions on first call rather than silent miscompares.
- `render_rgb` cascades through 4 kwarg combos — handles gym vs gymnasium vs newer MuJoCo backends.
- `rollout` handles 4-tuple (gym) and 5-tuple (gymnasium) `step()` returns, and `(obs, info)` 2-tuple `reset()` returns.

### Cell 16 (BC train + early stop)
- `num_workers=0` on the BC inner DataLoader — the dataset is in-memory features, no I/O, workers add multiprocessing fragility for zero gain.
- `policy.eval() / policy.train()` toggle is necessary for BatchNorm to compute correctly during eval.

### Cell 17 (BC sweep)
- Per-run try/except — one failure (e.g. a missing demo path) doesn't abort the sweep. Each failure is recorded as `{'best_success': None, 'error': ...}` so the markdown summary can flag it.

### Cell 18 (aggregation + plots)
- `summarize` reduces over seeds. The `n` field is the number of runs; small n (e.g. 1) is suspicious.
- `plot_per_task` overlays our per-task with R3M's `_approx` per-task — visually convenient, but the `_approx` caveat applies.
- `plot_training_curves` reads from `results.data['pretrain']['losses']` — pure side-effect-free function.

### Cell 19 (markdown summary)
- Builds the morning-presentation markdown directly from `results.json`. The presentation should never need to re-run the notebook — it just reads `summary.md` and the three PNGs.

### Cell 20 (smoke section header)
Explains what's stubbed (sim envs) and what's real (DINOv2, CLIP, all losses, optimizer step, feature extraction, BC inner loop). The `RUN_SMOKE_TESTS` flag is on by default; users can flip to skip on subsequent runs.

### Cell 21 (smoke setup)
- 8 synthetic clips × 5 frames + 6 demo episodes built into `_smoke/`.
- `SMOKE_CONFIG` is a deep copy of CONFIG with all expensive knobs reduced (`image_size=84` so the ViT processes 36 patches instead of 256, 4 pretrain steps, 50 BC steps, 2 eval episodes, no compile, no workers).

### Cell 22 (smoke pipeline)
- 9 stages, each with an `assert` that fails the cell if the contract is violated.
- Stage 8 (BC train) monkey-patches `make_env` and `rollout` in `globals()` because `train_bc_one` (defined in an earlier cell) looks them up by name at call time. After the BC test, the originals are restored.
- Stage 9 (aggregation) runs against a synthetic SWEEP — verifies plotting and markdown without needing a real BC sweep.

### Cell 23 (smoke cleanup)
- `rm -rf _smoke/` — production caches in `cache/` and `results/` are untouched.

### Cell 24 (production header)
- Skip-condition table tells the user how to re-run any stage.

### Cells 25-30 (production execution)
- Each is small and gated. The pretrain cell skips if `cache/pretrain.pt` exists — re-running the notebook after a successful pretrain is fast.

---

## 6. Performance & memory budget

### A100 80GB during pretrain

| Resource | Estimate | Headroom |
|---|---|---|
| Model + LoRA params (bf16) | ~340 MB | trivial |
| Activations (80 imgs × ViT-B/14, bf16) | ~20 GB | most of the headroom |
| Optimizer state (2.4M LoRA + 5.2M lang head, fp32 Adam) | ~90 MB | trivial |
| Gradients | ~30 MB | trivial |
| Batch buffer + DataLoader prefetch (4 batches) | ~2 GB | fine |
| **Total estimated** | **~22-25 GB** | **~55 GB headroom** |

If the A100 is shared, halve `batch_clips` to 8 and the working set drops to ~12 GB.

**Why not bigger batches?** R3M used batch=16 specifically because TCN/VLA losses need cross-video negatives sampled from within the batch. Larger batches give more negatives per anchor, which is generally better — but R3M's λ values were tuned at batch 16, so deviating would risk loss-balance issues.

### CLIP cache size

~5M unique narrations × 512 dim × 4 bytes = ~10 GB. If the actual unique count is lower (likely with R3M's preprocessing), proportionally smaller.

### Per-cell wall-clock (A100 measured estimates, not guaranteed)

| Cell | Operation | Estimate |
|---|---|---|
| 25 (PROD 1) | CLIP cache (5M narrations, bs=4096) | 3-5 min |
| 26 (PROD 2) | Pretrain 150k steps, bs=16×5 frames, ViT-B/14+LoRA | 2.0-2.3 h |
| 27 (PROD 3) | Feature extraction (12 task-views × ~10k frames) | 10-15 min |
| 28 (PROD 4) | BC sweep (default cut, ~120 runs) | 60-80 min |
| 29-30 (PROD 5-6) | Aggregate + plots + summary | 2-3 min |
| **Total** | | **~3.4-3.8 h** |

---

## 7. Tradeoffs and known limitations

### 7.1 We test architecture, not loss

R3M's loss formulation is preserved exactly. So our test answers "Does DINOv2+LoRA outperform R3M's ResNet50 under R3M's loss?" — *not* "What's the best representation for manipulation?" A future study should ablate λ values and loss components against the new architecture.

### 7.2 CLIP text encoder is a confound

Strict apples-to-apples would use DistilBERT. We made a deliberate choice to use CLIP because (a) it's vision-aligned text, more appropriate for VLA, and (b) preserving DistilBERT means another 768-dim embedding to manage. The reader should interpret our results as "DINOv2+LoRA backbone + CLIP text" vs "ResNet50 + DistilBERT text". If we wanted to isolate the visual backbone effect, we'd need a third condition with R3M's ResNet50 + CLIP text — outside the budget.

### 7.3 Pretrain budget reduction may underrate us

If our 150k checkpoint is still improving at end of pretrain, our reported numbers underrate what the architecture could achieve at full scale. This is a *positive* failure mode for the headline ("DINOv2+LoRA matches R3M with 6.7× less pretrain"), but a misleading one if reported as final ceiling.

**Mitigation:** the training curves plot tells this story directly. If the loss is plateauing at 150k, the comparison is fair. If it's still declining steeply, we report that explicitly.

### 7.4 Eval episode count reduction widens error bars

±5 pp instead of R3M's ±3 pp. For deltas of 5+ pp this is fine; for smaller deltas the comparison is underpowered.

### 7.5 Adroit may be cut

`cut_level >= 1` drops Adroit. Adroit is the env where R3M's language alignment showed the biggest contribution (Table 1: removing Lang dropped Adroit by 19.4 pp vs ~6 pp on Franka and 2 pp on MW). Cutting Adroit removes our strongest test of language alignment quality.

**Mitigation:** if pretrain finishes ahead of schedule, drop `cut_level` to 0 dynamically.

### 7.6 The R3M paper's per-task / per-view bars are read approximately

Per-task numbers in the baseline come from bar heights in Figs 4/8/9 and are accurate to ~3 pp. Aggregate numbers in Table 1-3 are paper-exact. The delta-vs-R3M plots use the aggregate numbers (exact); per-task plots use the approximate numbers. Flagged in the baseline cell.

### 7.7 Sim-env stubbing in smoke tests

The smoke tests stub `make_env` and `rollout`. We do not catch sim-env-specific failures (wrong camera name, wrong proprio composition, gym vs gymnasium API mismatches) until the production BC sweep. The TODOs in Cell 15 are explicit signals for what to verify on the A100 box first.

**Mitigation:** the first BC sweep call will surface the actual env name on `gym.make` failure; copy/paste from the error message into `task_to_id`.

### 7.8 DINOv2 was trained at 518×518; we use 224×224

Position embeddings are interpolated. DINOv2 paper reports 1-3 pp drop on dense prediction tasks at 224. For our purposes (CLS-token features for BC), this is likely smaller, but it is a real cost we pay for staying within budget.

**Alternative:** train at 518×518. Cost: ~10× more activation memory; batch must drop to 2-4 clips; pretrain time roughly doubles. Outside the budget.

---

## 8. Future work (post-presentation)

Things the design supports but we don't run:

1. **Patch-token features** instead of CLS. DINOv2's 256 patch tokens could feed a transformer-based BC policy that attends to relevant objects per timestep.
2. **Higher LoRA rank** (32, 64) once we have a baseline at r=16.
3. **Loss ablations**: drop L1, drop L2, drop VLA — rerun the BC sweep on the same downstream protocol to see which matter for DINOv2-init.
4. **518×518 training** with batch=4 — 5-8× longer pretrain but full DINOv2 fidelity.
5. **Cross-architecture comparison**: DINOv2 ViT-S/14 (22M) vs ViT-B/14 (86M) vs ViT-L/14 (300M).
6. **Real-robot eval** (Table 3 in R3M) — out of scope but the trained representation is a drop-in replacement.

---

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Ego4D path/cache layout mismatch | medium | high (blocks pretrain) | smoke test surfaces FileNotFoundError before production; P3 validates manifest before pretrain |
| Preprocessing crash on a corrupt video | medium | low | per-clip try/except; `n_failed` counter logged; sweep continues |
| `decord` not installable on Colab | low | low | P1 falls back to OpenCV automatically (~3× slower but portable) |
| Colab session disconnects mid-preprocessing | medium | medium (re-run preprocessing) | manifest checkpointed every `log_every` clips; resume is automatic |
| `mj_envs` env name drift | medium | medium (blocks one BC config) | per-run try/except logs the failing key; sweep continues |
| CUDA OOM at full batch on shared A100 | low-medium | medium | halve `batch_clips`, OOM is loud |
| `torch.compile` recompile loop on LoRA backbone | low | low | try/except falls back to eager |
| 150k pretrain is too short | medium | medium (lower bound on numbers) | training curves tell the story; can extend with a re-run |
| Sim rollouts hang or crash | low | high | per-run try/except; `env.close()` on cleanup |
| DINOv2 hub download fails (network) | low | high (blocks model load) | pre-warm by running smoke tests on the A100 box first |
| Pillow major version drift removes `Image.BICUBIC` | low | low | already migrated to `Image.Resampling.BICUBIC` |
| open_clip version drift breaks `'ViT-B-32-quickgelu'` | low | low | open_clip is stable; flagged in setup cell |

---

## 10. Files in this project

| File | Purpose |
|---|---|
| `baseline_2_dinov3_lora.ipynb` | Baseline 2 deliverable — 37-cell notebook (DINOv3 + LoRA + R3M-loss pretrain on Ego4D) |
| `baseline_3_dinov3_sam3.ipynb` | Baseline 3 deliverable — 29-cell notebook (DINOv3 + SAM 3, fully frozen, no Ego4D pretrain) |
| `build_notebook.py` | Regenerates baseline 2 from a single Python source (easier to diff than raw .ipynb JSON) |
| `build_notebook_b3.py` | Regenerates baseline 3 |
| `smoke_test.py` | Standalone CPU smoke test for baseline 2's core pipeline (DINOv3 stubbed) |
| `smoke_test_preprocess.py` | Standalone smoke test for `preprocess_ego4d` against synthetic mp4 + narrations.json |
| `run_notebook_offline.py` | Integration test that exec()s baseline 2's cells in order with DINOv3 + pyarrow + sim deps stubbed |
| `run_notebook_offline_b3.py` | Integration test for baseline 3 (DINOv3 + SAM 3 + sim deps stubbed) |
| `2203.12601v3.pdf` | The R3M paper (reference) |
| `paper.txt` | Plain-text extract of the paper (used during development to look up exact numbers) |
| `ARCHITECTURE.md` | This file |
| `README.md` | Operational instructions |

---

## 11. The single most important thing to verify on the A100 box before the morning

Run the smoke tests. If they pass, every code path in the production cells has been exercised end-to-end except for sim rollouts. The remaining work is to confirm the env-name strings and camera-name mappings match the user's `mj_envs` and `metaworld` versions — and the first failed `gym.make` will tell you exactly what to fix.

In other words: **don't kick off the 4-hour run blind.** Run smoke tests first (~2 min), inspect the `_smoke/` outputs visually, and only then run the production cells.
