"""Frozen-encoder adapters.

Contract (every adapter, no exceptions):
    enc = load(name, device)
    enc.sites                      # depth-ordered candidate readout sites, e.g. ['L4', 'L8', 'L12', 'L16']
    enc.encode(frames, text, sites) -> {site: float32 cpu tensor (B, 64, D)}

frames: uint8 numpy (B, 256, 256, 3), RGB, dataset orientation (already 180-degree rotated like
        lerobot/libero; env frames go through LiberoProcessorStep first).
text:   list[str] of length B for text-capable models (GENERIC_PROMPT in the primary condition, the
        task instruction in the instruction-on condition), or None. Non-text adapters ignore it.
Every adapter returns exactly 64 tokens per frame (8x8 grid, row-major); dense grids are average-pooled
with pool8. Token standardisation happens in extract.py, never here.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

GENERIC_PROMPT = "What should the robot do next?"
SIDE = 8  # 8x8 = 64 tokens per frame for every model


@dataclass
class Encoder:
    name: str
    sites: list
    fn: Callable  # fn(frames uint8 (B,256,256,3), text list[str] | None, sites) -> {site: (B,64,D)}
    text: bool = False
    n_tokens: int = SIDE * SIDE      # 256 only for the 16x16 token-budget ablation variants
    offline_sites: tuple = ()        # probed and compared, never rolled out (e.g. action-head tokens)

    def encode(self, frames, text=None, sites=None):
        sites = list(sites or self.sites)
        with torch.inference_mode():
            out = self.fn(frames, text if self.text else None, sites)
        assert set(out) == set(sites), (self.name, sorted(out), sites)
        return {s: v.float().cpu() for s, v in out.items()}


# ---------------------------------------------------------------- helpers

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def to_tensor(frames, size, device, mean_std=IMAGENET, dtype=torch.float32):
    """uint8 (B,H,W,3) -> float (B,3,size,size), bicubic+antialias, optionally normalised."""
    x = torch.from_numpy(np.array(frames)).to(device).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
    if mean_std is not None:
        m = torch.tensor(mean_std[0], device=device).view(1, 3, 1, 1)
        s = torch.tensor(mean_std[1], device=device).view(1, 3, 1, 1)
        x = (x - m) / s
    return x.to(dtype)


def pool8(x, h, w, side=SIDE):
    """(B, h*w, D) row-major grid -> (B, side*side, D) by average pooling."""
    if (h, w) == (side, side):
        return x
    B, N, D = x.shape
    assert N == h * w, (x.shape, h, w)
    g = x.transpose(1, 2).reshape(B, D, h, w).float()
    return F.adaptive_avg_pool2d(g, side).flatten(2).transpose(1, 2)


@contextlib.contextmanager
def capture(modules: dict):
    """Forward hooks: {site: module} -> dict filled with {site: first output tensor} during the block."""
    got = {}

    def hook(site):
        def f(_m, _i, out):
            got[site] = out[0] if isinstance(out, (tuple, list)) else out
        return f

    hs = [m.register_forward_hook(hook(s)) for s, m in modules.items()]
    try:
        yield got
    finally:
        for h in hs:
            h.remove()


# ---------------------------------------------------------------- stub (CPU smoke test, G4)

def _stub(device, d=32):
    """Deterministic, rotation-sensitive toy encoder: 32x32 patch means -> fixed random projection."""
    g = torch.Generator().manual_seed(0)
    w0 = torch.randn(3 * 4, d, generator=g)
    w1 = torch.randn(d, d, generator=g) / d ** 0.5

    def fn(frames, text, sites):
        x = to_tensor(frames, 256, "cpu", mean_std=None)                  # (B,3,256,256)
        q = F.avg_pool2d(x, 16)                                             # (B,3,16,16)
        q = q.unfold(2, 2, 2).unfold(3, 2, 2)                               # (B,3,8,8,2,2)
        q = q.permute(0, 2, 3, 1, 4, 5).reshape(len(frames), 64, 12)        # 2x2 sub-patch layout keeps orientation
        s0 = q @ w0
        out = {"s0": s0, "s1": torch.tanh(s0 @ w1)}
        return {s: out[s] for s in sites}

    return Encoder("stub", ["s0", "s1"], fn)


ADAPTERS: dict[str, Callable] = {"stub": _stub}

# Shapes and API facts below were read from configs/source (lerobot 0.6.1, transformers 5.5, diffusers 0.39, hub
# remote code at the pinned revisions); G5 asserts them on real weights. Sites are named by 1-indexed block depth.


def _bf16(device):
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _cast_params(module, dtype):
    """Cast parameters only: Module.to(dtype) would also cast fp32 RoPE inv_freq buffers and perturb positions."""
    for p in module.parameters():
        p.data = p.data.to(dtype)
    return module


# ---------------------------------------------------------------- V-JEPA 2.1 ViT-L (Tier 1), random floor, V-JEPA 2 (Tier 2)

VJEPA2_HUB = "facebookresearch/vjepa2:204698b45b3712590f06245fbfba32d3be539812"
VJEPA2_1_L_CKPT = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"  # 5.15 GB, key ema_encoder
VJEPA_SITES = {"L6": 6, "L12": 12, "L18": 18, "L24": 24}


def _vjepa2_1(device, pretrained=True, side=SIDE):
    """V-JEPA 2.1 ViT-L/16-384. A (B,3,1,384,384) input takes the image tokenizer (tubelet 1): nothing duplicated.
    No CLS/registers: 24x24 grid. Each site is that level's block output through its own norms_block LayerNorm
    (== hierarchical_layers [5,11,17,23]; L24 is the default output). fp32 weights under bf16 autocast: casting the
    model to bf16 hands SDPA mixed dtypes (RoPE stays fp32)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)  # random_vit: this init IS the model; vjepa2_1: overwritten below
        enc, _ = torch.hub.load(VJEPA2_HUB, "vjepa2_1_vit_large_384", pretrained=False, skip_validation=True)
    if pretrained:  # the hub's base URL is localhost, so the public checkpoint is loaded by hand
        ckpt = torch.hub.load_state_dict_from_url(VJEPA2_1_L_CKPT, map_location="cpu", weights_only=True)
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt["ema_encoder"].items()}
        enc.load_state_dict(sd, strict=True)
        del ckpt, sd
    enc.out_layers = [b - 1 for b in VJEPA_SITES.values()]
    enc = enc.to(device).eval()

    def fn(frames, text, sites):
        x = to_tensor(frames, 384, device).unsqueeze(2)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outs = dict(zip(VJEPA_SITES, enc(x)))
        return {s: pool8(outs[s], 24, 24, side) for s in sites}

    name = ("vjepa2_1" if pretrained else "random_vit") + ("" if side == SIDE else f"_t{side * side}")
    return Encoder(name, list(VJEPA_SITES), fn, n_tokens=side * side)


def _vjepa2_l(device):
    """V-JEPA 2 ViT-L/16-256 (HF port), exactly 256^2 (RoPE grid fixed at 16). One frame as pixel_values_videos
    (B,1,3,256,256), repeated internally to tubelet 2. Sites: hidden_states[k] through the final encoder LayerNorm,
    as upstream out_layers does. Not a weight ladder with 2.1 (2.1-L is distilled from 2.1-G, new modules, 384^2)."""
    from transformers import AutoModel

    m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", revision="b3c1679b7c34d3255ef3547f27c7b226aefab26f",
                                  dtype=torch.float32).to(device).eval()

    def fn(frames, text, sites):
        x = to_tensor(frames, 256, device).unsqueeze(1)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hs = m(pixel_values_videos=x, skip_predictor=True, output_hidden_states=True).hidden_states
            return {s: pool8(m.encoder.layernorm(hs[VJEPA_SITES[s]]), 16, 16) for s in sites}

    return Encoder("vjepa2_l", list(VJEPA_SITES), fn)


# ---------------------------------------------------------------- C-RADIOv4-H, TIPSv2-L/14 (hub remote code, pinned)

CRADIO4 = ("nvidia/C-RADIOv4-H", "0057b339059c0b9e1b4ba996f975410ebbfdfcc8")  # needs open_clip_torch in the venv
TIPSV2 = ("google/tipsv2-l14", "52847a71eb02a3082c370fbcd98cd799687bd4d0")


def _block_outputs(sites, run):
    """sites 'L<k>' -> run(sorted 0-based block indices) -> {site: tokens}; the APIs return in block order."""
    idx = sorted({int(s[1:]) - 1 for s in sites})
    out = dict(zip(idx, run(idx)))
    return {s: out[int(s[1:]) - 1] for s in sites}


def _cradio4(device, side=SIDE):
    """C-RADIOv4-H: ViT-H/16, 32 blocks, D=1280. [0,1] input at 512^2 (the model applies its own mean/std) -> 32x32.
    forward_intermediates (on the inner radio_model; the HF wrapper lacks it) drops 3 summary + 7 register tokens.
    fp32 weights + bf16 autocast: .to(bf16) breaks the input conditioner."""
    from transformers import AutoModel

    rm = AutoModel.from_pretrained(CRADIO4[0], revision=CRADIO4[1], trust_remote_code=True).radio_model
    rm = rm.to(device).eval()

    def fn(frames, text, sites):
        x = to_tensor(frames, 512, device, mean_std=None)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = _block_outputs(sites, lambda idx: rm.forward_intermediates(
                x, indices=idx, output_fmt="NLC", intermediates_only=True, stop_early=True))
        return {s: pool8(v, 32, 32, side) for s, v in out.items()}

    return Encoder("cradio4" + ("" if side == SIDE else f"_t{side * side}"), ["L8", "L16", "L24", "L32"], fn,
                   n_tokens=side * side)


def _tipsv2(device):
    """TIPSv2 L/14 image tower: 24 blocks, D=1024. [0,1] input at 448^2 (no normalisation) -> 32x32.
    get_intermediate_layers drops CLS + 1 register; norm=False = raw residual stream like the hooked ViTs.
    Dual encoder: image tokens never see text."""
    from transformers import AutoModel

    vit = AutoModel.from_pretrained(TIPSV2[0], revision=TIPSV2[1], trust_remote_code=True).vision_encoder
    vit = vit.to(device).eval()

    def fn(frames, text, sites):
        x = to_tensor(frames, 448, device, mean_std=None)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = _block_outputs(sites, lambda idx: vit.get_intermediate_layers(x, n=idx, norm=False))
        return {s: pool8(v, 32, 32) for s, v in out.items()}

    return Encoder("tipsv2", ["L6", "L12", "L18", "L24"], fn)


# ---------------------------------------------------------------- LingBot-Vision ViT-L/16 (Tier 1), DINOv3, R3M (Tier 2)

VIT_L_SITES = ["L6", "L12", "L18", "L24"]


def _lingbot_vision(device):
    """LingBot-Vision ViT-L/16 (distilled from the self-supervised ViT-g teacher). ImageNet-normalised 512^2 -> 32x32.
    get_intermediate_layers(norm=False) == hooks on backbone.blocks with CLS + 4 storage tokens sliced off."""
    from lingbot_vision import load_pretrained_backbone

    dtype = _bf16(device)
    # keyword on purpose: the first positional parameter is repo_id_or_path, so ("large") would hit a hub repo "large"
    model, _ = load_pretrained_backbone(variant="large", device=device, dtype=dtype)

    def fn(frames, text, sites):
        x = to_tensor(frames, 512, device, dtype=dtype)
        out = _block_outputs(sites, lambda idx: model.get_intermediate_layers(x, n=idx, norm=False))
        return {s: pool8(v, 32, 32) for s, v in out.items()}

    return Encoder("lingbot_vision", VIT_L_SITES, fn)


def _dinov3_l(device):
    """DINOv3 ViT-L/16 (gated, manual approval). 512^2 -> 32x32 like lingbot_vision; CLS + 4 registers dropped.
    Hooks on model.layer (pre final norm == hidden_states[k]); output_hidden_states would keep all 25 layers alive."""
    from transformers import AutoModel

    dtype = _bf16(device)
    model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", dtype=dtype).to(device).eval()
    skip = 1 + model.config.num_register_tokens

    def fn(frames, text, sites):
        x = to_tensor(frames, 512, device, dtype=dtype)
        with capture({s: model.model.layer[int(s[1:]) - 1] for s in sites}) as got:
            model(pixel_values=x)
        return {s: pool8(got[s][:, skip:], 32, 32) for s in sites}

    return Encoder("dinov3_l", VIT_L_SITES, fn)


def _r3m(device):
    """R3M ResNet-50 trunk (load_r3m wraps it in DataParallel; fc=Identity). No tokens: stage feature maps.
    256^2 (trained at 224) so layer4 is exactly 8x8. eval() is required: load_r3m leaves BatchNorm in train mode."""
    from r3m import load_r3m

    dtype = _bf16(device)
    trunk = load_r3m("resnet50").module.convnet.to(device, dtype).eval()

    def fn(frames, text, sites):
        x = to_tensor(frames, 256, device, dtype=dtype)  # == R3M.forward: x/255 then ImageNet Normalize
        with capture({s: getattr(trunk, s) for s in sites}) as got:
            trunk(x)
        return {s: pool8(got[s].flatten(2).transpose(1, 2), *got[s].shape[-2:]) for s in sites}

    return Encoder("r3m", ["layer1", "layer2", "layer3", "layer4"], fn)


# ---------------------------------------------------------------- action-head offline sites (§1.6), shared by GR00T + MolmoAct2


@contextlib.contextmanager
def capture_inputs(modules: dict):
    """Forward pre-hooks: {site: module} -> dict filled with {site: (args, kwargs)} of that module's call."""
    got = {}
    hs = [m.register_forward_pre_hook(lambda _m, a, kw, s=s: got.__setitem__(s, (a, kw)), with_kwargs=True)
          for s, m in modules.items()]
    try:
        yield got
    finally:
        for h in hs:
            h.remove()


def xattn_read(q, k, v, bias, w_out):
    """Per-source-token read of one cross-attention layer: token j -> W_o concat_h(sum_i A[h,i,j] v[h,j]).
    Summed over j it equals the layer's output summed over its queries (out bias excluded), so it splits exactly what
    the action tokens read, by source token, and keeps the image grid. q (B,h,Q,d); k, v (B,h,L,d); additive bias
    broadcastable to (B,h,Q,L); w_out (D, h*d) -> (B,L,D) fp32. Call outside autocast."""
    a = torch.softmax(q.float() @ k.float().transpose(-1, -2) * q.shape[-1] ** -0.5 + bias.float(), -1)
    return (a.sum(2)[..., None] * v.float()).transpose(1, 2).flatten(2) @ w_out.float().T


# ---------------------------------------------------------------- Qwen3-VL family: L1 ladder (+ Tier-2 36-layer models)

# Base N1.7 was pretrained on xdof / oxe_droid / g1 / r1_pro only (experiment_cfg/conf.yaml); libero_sim (id 2) has
# untrained category weights. DROID is a Franka arm like LIBERO's.
GROOT_EMBODIMENT = "oxe_droid_relative_eef_relative_joint"


def _qwen_vl(device, name, repo, sites=("L4", "L8", "L12", "L16"), select_layer=16, groot=False, state_dict=None):
    """One code path for every Qwen3-VL rung: LeRobot's Qwen3Backbone (LLM truncated to select_layer) + LeRobot's
    Qwen3-VL processor. Frames forced to 256x256 -> 16x16 patches -> 2x2 merge -> 8x8 = 64 image tokens, row-major.
    Image tokens precede the text (4 fixed tokens before them) and the LLM is causal, so image-position states
    cannot see the prompt. Sites are forward hooks on language_model.layers[k-1] (pre-final-norm residual stream);
    HF hidden_states[select_layer] would be post-norm and its early entries omit the deepstack add.
    No crop for GR00T (its own eval does a 0.95 centre crop): all rungs get identical inputs.
    GR00T only, offline sites (probed, never rolled out):
      AH_vl   action_head.vl_self_attention output (vlln + 4 bidirectional blocks over L16) at the 64 image positions:
              the K/V source of every DiT cross-attention. Unlike the L sites it sees the prompt.
      AH_x31  DiT block 31 (last image cross-attention; image blocks are 3, 7, ..., 31) read per image token
              (xattn_read). One flow step at t=0 (the sampler's first step, pure noise), fixed noise, zero normalised
              state, GROOT_EMBODIMENT. The DiT's own 41 tokens (state + 40-step chunk) have no 8x8 layout; this does."""
    from lerobot.policies.groot import processor_groot
    from lerobot.policies.groot.groot_n1_7 import GR00TN17, Qwen3Backbone

    dt = _bf16(device)
    ah = None
    if groot:
        # Backbone config is hard-coded in LeRobot; ViT + LLM layers 0-15 come from the GR00T checkpoint.
        # GrootPolicy is skipped because it casts every parameter to fp32 (model_params_fp32=True).
        m = GR00TN17.from_pretrained(repo, use_flash_attention=False)
        bb, ah = m.backbone, _cast_params(m.action_head, dt).to(device).eval()
        proc = processor_groot._build_n1_7_processor()   # default nvidia/Cosmos-Reason2-2B, as LeRobot does
        hc = ah.config
        eid = processor_groot.N1_7_EMBODIMENT_MAPPING[GROOT_EMBODIMENT]
        noise = torch.randn(1, hc.action_horizon, hc.max_action_dim, generator=torch.Generator().manual_seed(0))
        noise = noise.to(device, dt)
    else:
        bb = Qwen3Backbone(model_name=repo, select_layer=select_layer, load_bf16=True,
                           load_pretrained_weights=state_dict is None)
        if state_dict is not None:
            sd = state_dict()
            sd.pop("lm_head.weight", None)   # Qwen3Backbone ties lm_head to embed_tokens
            info = bb.model.load_state_dict(sd, strict=False)
            assert set(info.missing_keys) <= {"lm_head.weight"} and not info.unexpected_keys, info
        proc = processor_groot._build_n1_7_processor(repo)
    bb = _cast_params(bb, dt).to(device).eval()
    layers = bb.language_model.layers
    assert len(layers) == select_layer, len(layers)
    image_token_id = bb.model.config.image_token_id

    def head(bo, sites):
        """GR00T action-head sites from Qwen3Backbone.forward's output; mirrors get_action_with_features' 1st step."""
        img, B = bo["image_mask"], len(bo["image_mask"])
        vl = ah.process_backbone_output(bo)["backbone_features"]   # vlln + vl_self_attention (in place on bo)
        out = {s: vl[img].view(B, SIDE * SIDE, -1) for s in sites if s == "AH_vl"}
        xs = {s: ah.model.transformer_blocks[int(s[4:]) - 1].attn1 for s in sites if s.startswith("AH_x")}
        if xs:
            e = torch.full((B,), eid, device=device)
            t = torch.zeros(B, dtype=torch.long, device=device)
            sf = ah.state_encoder(torch.zeros(B, 1, hc.max_state_dim, device=device, dtype=dt), e)
            af = ah.action_encoder(noise.expand(B, -1, -1), t, e) + ah.position_embedding(
                torch.arange(hc.action_horizon, device=device))
            with capture_inputs(xs) as got:
                ah.model(hidden_states=torch.cat([sf, af], 1), encoder_hidden_states=vl, timestep=t,
                         image_mask=img, backbone_attention_mask=bo["backbone_attention_mask"])
            for s, a in xs.items():
                (h,), kw = got[s]
                assert torch.equal(kw["attention_mask"], img & bo["backbone_attention_mask"]), f"{s}: not an image block"
                heads = lambda z: z.unflatten(-1, (a.heads, -1)).transpose(1, 2)  # noqa: E731
                ctx = kw["encoder_hidden_states"]
                bias = torch.zeros(img.shape, device=device).masked_fill(~kw["attention_mask"], float("-inf"))
                r = xattn_read(heads(a.to_q(h)), heads(a.to_k(ctx)), heads(a.to_v(ctx)), bias[:, None, None],
                               a.to_out[0].weight)
                out[s] = r[img].view(B, SIDE * SIDE, -1)
        return out

    def fn(frames, text, sites):
        B = len(frames)
        text = text or [GENERIC_PROMPT] * B
        prompts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
                                            tokenize=False, add_generation_prompt=False) for t in text]
        images = list(torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2))
        enc = proc(text=prompts, images=images, return_tensors="pt", padding=True,
                   min_pixels=256 * 256, max_pixels=256 * 256).to(device)
        mask = enc["input_ids"] == image_token_id
        assert (mask.sum(1) == SIDE * SIDE).all(), mask.sum(1)
        llm = [s for s in sites if s[0] == "L"]
        # ponytail: Qwen3Backbone.forward also computes full-vocab logits (~B*L*152k bf16); fine at B<=128
        with capture({s: layers[int(s[1:]) - 1] for s in llm}) as got:
            bo = bb(enc)
        out = {s: got[s][mask].view(B, SIDE * SIDE, -1) for s in llm}
        return out if ah is None else out | head(bo, sites)

    return Encoder(name, list(sites), fn, text=True, offline_sites=("AH_vl", "AH_x31") if groot else ())


def _qwen3vl_2b(device):
    return _qwen_vl(device, "qwen3vl_2b", "Qwen/Qwen3-VL-2B-Instruct")


def _cosmos_reason2(device):
    return _qwen_vl(device, "cosmos_reason2", "nvidia/Cosmos-Reason2-2B")


def _groot_n17(device):
    return _qwen_vl(device, "groot_n17", "nvidia/GR00T-N1.7-3B", groot=True)


# ---------------------------------------------------------------- MolmoAct2

MOLMOACT2_REV = "e432d85f6e039edca44afb93c262f3084ab72a9c"  # allenai/MolmoAct2 (not the LIBERO fine-tune); fp32, 21.8 GB


def _molmoact2(device):
    """allenai/MolmoAct2 via LeRobot's MolmoAct2Policy backbone (ViT + connector + LLM; LLM from Qwen3-4B).
    One 378^2 crop -> 14x14 = 196 <im_patch> tokens (27x27 ViT patches, 2x2 attention-pooled), D=2560, pooled to 8x8.
    Sites: residual stream after blocks 9/18/27/36 (pre ln_f). Sequence: <bos><im_start>196x<im_patch><im_end> then
    the chat turn; image tokens attend only to each other and earlier tokens, so they never see the prompt.
    LeRobot's pack step is bypassed so no state clause is added.
    Offline sites (probed, never rolled out): AH_x18 / AH_x36 = action-expert block 18 / 36 cross-attention read per
    image token (xattn_read; D=768, 196 -> 8x8). Block k reads LLM layer k's KV cache (projections of that layer's
    input, i.e. of L(k-1)), so AH_xk is the expert's view of that layer. One flow step at t=0 (the first of 10 Euler
    steps, pure noise), fixed noise zeroed past LIBERO's 7 action dims (mask_action_dim_padding), no state (MolmoAct2
    takes state only as prompt tokens). The expert's 30 action tokens have no 8x8 layout; this does."""
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy, _resolve_checkpoint_location
    from lerobot.policies.molmoact2.processor_molmoact2 import _load_local_molmoact2_processor

    cuda = device.type == "cuda"
    cfg = MolmoAct2Config(
        checkpoint_revision=MOLMOACT2_REV, device=device.type, model_dtype="bfloat16" if cuda else "float32",
        input_features={"observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))})
    bb = MolmoAct2Policy(cfg)._backbone().to(device).eval()
    # ponytail: private lerobot==0.6.1 helpers, the exact path the policy preprocessor uses (minus its state clause)
    proc = _load_local_molmoact2_processor(_resolve_checkpoint_location(cfg.checkpoint_path, revision=MOLMOACT2_REV))
    blocks = bb.transformer.blocks
    assert len(blocks) == 36 and bb.config.hidden_size == 2560, (len(blocks), bb.config.hidden_size)
    ae = bb._require_action_expert()
    # depth gate off in this checkpoint, so generate_actions_from_inputs' gating step is a no-op and is skipped
    assert len(ae.blocks) == 36 and not bb.config.action_expert_depth_gate, (len(ae.blocks), bb.config.action_expert_depth_gate)
    H = int(bb.config.max_action_horizon)   # 30 (= cfg.chunk_size, written back by the policy)
    noise = torch.randn(1, H, bb.config.max_action_dim, generator=torch.Generator().manual_seed(0))
    noise[..., 7:] = 0   # LIBERO actions are 7-d; the policy zeroes padded dims of the initial noise
    noise = noise.to(device, ae.action_embed.weight.dtype)

    def fn(frames, text, sites):
        frames = np.ascontiguousarray(frames)  # the processor's torch.from_numpy rejects negative-stride views
        B = len(frames)
        text = text or [GENERIC_PROMPT] * B
        prompts = [f"<|image|><|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n" for t in text]
        inputs = proc(text=prompts, images=list(frames), return_tensors="pt", padding=True).to(device)
        is_patch = inputs["input_ids"] == bb.config.image_patch_id  # exactly 196 per row
        llm = [s for s in sites if s[0] == "L"]
        xs = {s: ae.blocks[int(s[4:]) - 1].cross_attn for s in sites if s.startswith("AH_x")}
        with capture({s: blocks[int(s[1:]) - 1] for s in llm}) as got, capture_inputs(xs) as got_x, \
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
            o = bb(**inputs, use_cache=bool(xs))
            if xs:  # == generate_actions_from_inputs' first Euler step
                ctx = ae.prepare_context(
                    encoder_kv_states=bb._extract_kv_states(o.past_key_values),
                    encoder_attention_mask=bb._get_encoder_attention_mask(inputs["input_ids"], inputs.get("attention_mask")),
                    batch_size=B, seq_len=H, device=device, dtype=noise.dtype)
                ae.forward_with_context(noise.expand(B, -1, -1), torch.zeros(B, device=device), context=ctx)
        out = {s: got[s][is_patch] for s in llm}
        for s, ca in xs.items():
            (x,), kw = got_x[s]
            q = ca.q_norm(ca.q_proj(x).unflatten(-1, (ca.num_heads, -1))).transpose(1, 2)
            out[s] = xattn_read(q, kw["kv_k"].transpose(1, 2), kw["kv_v"].transpose(1, 2), kw["attn_mask"],
                                ca.out_proj.weight)[is_patch]
        return {s: pool8(v.view(B, 196, -1), 14, 14) for s, v in out.items()}

    return Encoder("molmoact2", ["L9", "L18", "L27", "L36"], fn, text=True, offline_sites=("AH_x18", "AH_x36"))


# ---------------------------------------------------------------- video world models: Wan2.2-TI2V-5B -> LingBot-VA (L3)

WAN = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
LINGBOT = "lerobot/lingbot_va_base"


def _wan_family(device, name, lingbot=False, size=256):
    """One forward over two latent frames, as in the first step of TI2V image-to-video: frame 0 = VAE(obs), clean,
    timestep 0; frame 1 = fixed-seed noise at t=999 for BOTH rungs (Wan's first sampler step; inside LingBot's
    training grid, whose own first step is 1000), so the L3 contrast differs only in weights.
    Site 'b{k}/obs' | 'b{k}/pred' = output of block k, tokens of latent frame 0 | 1 (frame-major). Tokens attend to
    the text through cross-attention, so they are NOT prompt-invariant. VAE + UMT5 are byte-identical across rungs.
    The noise draw changes obs tokens too (bidirectional attention): it is fixed and must never change."""
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel

    dt = _bf16(device)
    vae = AutoencoderKLWan.from_pretrained(WAN, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    tok = AutoTokenizer.from_pretrained(WAN, subfolder="tokenizer")
    # ponytail: UMT5-XXL stays resident (+11 GB VRAM) so instruction prompts encode fast; .cpu() it if VRAM is tight
    te = UMT5EncoderModel.from_pretrained(WAN, subfolder="text_encoder", dtype=dt).to(device).eval()
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    side = size // 32  # VAE /16, then patch (1,2,2)
    n = side * side
    noise = torch.randn(1, 48, 1, 2 * side, 2 * side, generator=torch.Generator().manual_seed(0)).to(device, dt)
    t_noise = 999.0

    if lingbot:
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.lingbot_va import LingBotVAPolicy
        from lerobot.policies.lingbot_va.utils import get_mesh_id

        cfg = PreTrainedConfig.from_pretrained(LINGBOT)
        cfg.device, cfg.guidance_scale = str(device), 1.0
        cfg.dtype = str(dt).removeprefix("torch.")  # the policy casts the whole transformer to this
        cfg.obs_cam_keys, cfg.height, cfg.width = ["observation.images.image"], size, size  # base: 3 cams, 256x320
        tr = LingBotVAPolicy.from_pretrained(LINGBOT, config=cfg, strict=True).transformer
        grid = get_mesh_id(2, side, side, 0).to(device)[None]  # RoPE ids (frame, row, col, stream=video)
    else:
        from diffusers import WanTransformer3DModel

        tr = WanTransformer3DModel.from_pretrained(WAN, subfolder="transformer", torch_dtype=dt).to(device).eval()

    cache = {}

    def embed(prompts):
        for p in set(prompts) - cache.keys():
            t = tok([p], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
            m = t.attention_mask.to(device)
            cache[p] = (te(t.input_ids.to(device), m).last_hidden_state * m[..., None]).to(dt)
        return torch.cat([cache[p] for p in prompts])

    def fn(frames, text, sites):
        B = len(frames)
        x = to_tensor(frames, size, device, ((0.5,) * 3, (0.5,) * 3))  # [-1, 1]
        z = vae.encode(x[:, :, None]).latent_dist.mode()                # one image -> one latent frame
        lat = torch.cat([((z - mean) / std).to(dt), noise.expand(B, -1, -1, -1, -1)], 2)
        ts = torch.tensor([0.0, t_noise], device=device).expand(B, 2)
        emb = embed(text or [GENERIC_PROMPT] * B)
        with capture({b: tr.blocks[int(b[1:]) - 1] for b in {s.split("/")[0] for s in sites}}) as got:
            if lingbot:  # video stream only, no KV cache -> any batch size
                tr({"noisy_latents": lat, "timesteps": ts, "grid_id": grid, "text_emb": emb})
            else:
                tr(hidden_states=lat, timestep=ts.repeat_interleave(n, 1), encoder_hidden_states=emb, return_dict=False)
        out = {}
        for s in sites:
            b, which = s.split("/")
            out[s] = pool8(got[b][:, :n] if which == "obs" else got[b][:, n:], side, side)
        return out

    return Encoder(name, [f"b{k}/{w}" for k in (8, 15, 22, 30) for w in ("obs", "pred")], fn, text=True)


def _wan22(device):
    return _wan_family(device, "wan22")


def _wan22_512(device):  # §1.5 resolution check (offline): 16x16 tokens per latent frame, pooled to 8x8
    return _wan_family(device, "wan22_512", size=512)


def _lingbot_va(device):
    return _wan_family(device, "lingbot_va", lingbot=True)


# ---------------------------------------------------------------- Tier 2: VGGT-Omega


def _vggt_omega(device):
    """VGGT-Omega 1B-512 (gated, manual approval; FAIR noncommercial). Each frame is its own one-view scene, so global
    attention is intra-frame (blocks 2/6/9/14/20 are register-only). Sites A5/A12/A18/A24 = the aggregator's cached
    blocks (0-indexed 4/11/17/23, what its heads read): concat(frame-attn, global-attn) tokens, D=2048; camera +
    16 register tokens dropped; 512^2 -> 32x32."""
    from huggingface_hub import hf_hub_download
    from vggt_omega.models import VGGTOmega

    sd = torch.load(hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt"), map_location="cpu", weights_only=True)
    model = VGGTOmega()
    model.load_state_dict(sd.get("model", sd), strict=True)
    agg = model.aggregator.to(device).eval()   # heads unused
    blocks = sorted(agg.cached_layer_indices)
    assert blocks == [4, 11, 17, 23], blocks

    def fn(frames, text, sites):
        x = to_tensor(frames, 512, device, mean_std=None)[:, None]   # (B,1,3,512,512) in [0,1]; aggregator normalises
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outs, start = agg(x)
        return {s: pool8(outs[int(s[1:]) - 1][:, 0, start:], 32, 32) for s in sites}

    return Encoder("vggt_omega", [f"A{i + 1}" for i in blocks], fn)


# ---------------------------------------------------------------- Tier 2: 36-layer Qwen3-VL models (same path as L1)
# Qwen3-VL-4B -> Xiaomi-Robotics-1 ladder (XR1's action DiT reads all 36 layers' KV, so no truncation) and the
# Cosmos 3 Nano reasoner (its understanding tower is Qwen3-VL-8B: identical configs and tokenizer files).

Q36 = ("L9", "L18", "L27", "L36")


def _xr1_vlm_state_dict():
    """Xiaomi-Robotics-1's VLM half: model_states.pt {"module": {...}} with the VLM under "model.vlm." in Qwen3-VL-4B
    names, plus XR1-only action/score token embeddings (dropped). XR1's own code needs transformers 4.57; the VLM
    weights load into stock Qwen3-VL under 5.5, so no second venv for the readout. Known input shift: XR1 trained
    with view text before each <image>; the image-first prompt here drops it so both ladder rungs see identical input."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("XiaomiRobotics/Xiaomi-Robotics-1-5B", "model_states.pt")
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)["module"]
    sd = {k[len("model.vlm."):]: v for k, v in sd.items() if k.startswith("model.vlm.")}
    del sd["model.action_embed.weight"], sd["model.score_embed.weight"]
    return sd


# NVIDIA transformers_cosmos3 KEY_MAPPING entries that fire on Cosmos3-Nano's flat diffusers keys (single-group
# rewrites: transformers 5.5's key_mapping loader breaks the plugin's \1\2 rules, so the mapping is applied here)
COSMOS3_RENAMES = (
    (r"\.self_attn\.to_q\.", ".self_attn.q_proj."), (r"\.self_attn\.to_k\.", ".self_attn.k_proj."),
    (r"\.self_attn\.to_v\.", ".self_attn.v_proj."), (r"\.self_attn\.to_out\.", ".self_attn.o_proj."),
    (r"\.self_attn\.norm_q\.", ".self_attn.q_norm."), (r"\.self_attn\.norm_k\.", ".self_attn.k_norm."),
    (r"^(embed_tokens\.|layers\.|norm\.)", r"model.language_model.\1"),
    (r"^(blocks\.|merger\.|patch_embed\.|pos_embed\.|deepstack_merger_list\.)", r"model.visual.\1"),
)


def _cosmos3_und_state_dict():
    """Cosmos 3 Nano understanding tower in HF Qwen3-VL names (749 tensors, ~16.3 GB bf16, in transformer shards 1-6
    + vision_encoder); generation-tower keys skipped. Fetches all shards: cosmos3_gen needs shard 7 anyway."""
    import glob
    import re

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    gen_only = re.compile(r"_moe_gen|\.self_attn\.(add_|to_add_out|norm_added_)")
    root = snapshot_download("nvidia/Cosmos3-Nano", allow_patterns=["transformer/*.safetensors", "vision_encoder/*.safetensors"])
    sd = {}
    for path in sorted(glob.glob(f"{root}/transformer/*.safetensors") + glob.glob(f"{root}/vision_encoder/*.safetensors")):
        with safe_open(path, "pt") as f:
            for k in f.keys():
                h = k
                for pat, rep in COSMOS3_RENAMES:
                    h = re.sub(pat, rep, h)
                if h.startswith(("model.language_model.", "model.visual.")) and not gen_only.search(k):
                    assert h not in sd, h
                    sd[h] = f.get_tensor(k)
    return sd


# ---------------------------------------------------------------- Tier 2: Cosmos 3 Nano generator tower

COSMOS3_GEN_SITES = ["O9", "O18", "O27", "O36", "P9", "P18", "P27", "P36"]   # {observed, predicted} x MoT layer


def _cosmos3_gen(device):
    """Generation half of the Cosmos 3 Nano MoT through diffusers 0.39's Cosmos3OmniPipeline helpers (private; pin
    diffusers==0.39.0), replaying the conditional pass of the first image-to-video step: the observation repeated to 5
    frames -> Wan2.2 VAE -> 2 latent frames x 8x8 tokens. Frame 0 = clean condition (O), frame 1 = seeded noise at the
    first scheduler timestep (P). Hooks read the generation stream (layer output[1]), pre-final-norm. The transformer
    packs one sample per sequence, so frames run one at a time. Gen tokens attend to the text (not prompt-invariant)."""
    from diffusers import Cosmos3OmniPipeline
    from PIL import Image

    dtype = _bf16(device)
    pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=dtype, sound_tokenizer=None,
                                               enable_safety_checker=False).to(device)
    pipe.scheduler.set_timesteps(35, device=device)   # __call__'s default num_inference_steps
    t0 = pipe.scheduler.timesteps[0].item()
    layers = pipe.transformer.layers
    assert len(layers) == 36, len(layers)
    nf, hw, n = 5, 256, SIDE * SIDE

    def fn(frames, text, sites):
        text = text or [GENERIC_PROMPT] * len(frames)
        depths = sorted({int(s[1:]) for s in sites})
        got = {d: [] for d in depths}
        hooks = [layers[d - 1].register_forward_hook(lambda _m, _i, out, d=d: got[d].append(out[1])) for d in depths]
        try:
            for frame, prompt in zip(frames, text):
                ids, _ = pipe.tokenize_prompt(prompt, num_frames=nf, height=hw, width=hw, fps=24.0)
                seg = pipe._prepare_text_segment(ids, device=device)
                lat = pipe.prepare_latents(image=Image.fromarray(np.ascontiguousarray(frame)), num_frames=nf, height=hw,
                                           width=hw, fps=24.0, generator=torch.Generator(device).manual_seed(0),
                                           device=device, dtype=dtype)[0]
                vis = pipe._prepare_vision_segment(
                    input_vision_tokens=lat, has_image_condition=True, mrope_offset=seg["vision_start_temporal_offset"],
                    vision_fps=24.0, curr=seg["und_len"], device=device, condition_frame_indexes=[0])
                assert vis["vision_token_shapes"] == [(2, SIDE, SIDE)], vis["vision_token_shapes"]
                pipe.transformer(
                    input_ids=seg["input_ids"], text_indexes=seg["text_indexes"],
                    position_ids=torch.cat([seg["text_mrope_ids"], vis["vision_mrope_ids"]], dim=1),
                    und_len=seg["und_len"], sequence_length=seg["und_len"] + vis["num_vision_tokens"],
                    vision_tokens=[lat.to(dtype)], vision_token_shapes=vis["vision_token_shapes"],
                    vision_sequence_indexes=vis["vision_sequence_indexes"],
                    vision_mse_loss_indexes=vis["vision_mse_loss_indexes"],
                    vision_timesteps=torch.full((vis["num_noisy_vision_tokens"],), t0, device=device),
                    vision_noisy_frame_indexes=vis["vision_noisy_frame_indexes"])
        finally:
            for h in hooks:
                h.remove()
        gen = {d: torch.stack(v) for d, v in got.items()}   # (B, 128, D): latent frame 0 rows, then frame 1
        return {s: gen[int(s[1:])][:, :n] if s[0] == "O" else gen[int(s[1:])][:, n:2 * n] for s in sites}

    return Encoder("cosmos3_gen", COSMOS3_GEN_SITES, fn, text=True)


ADAPTERS.update({
    "vjepa2_1": _vjepa2_1, "random_vit": lambda d: _vjepa2_1(d, pretrained=False), "vjepa2_l": _vjepa2_l,
    "cradio4": _cradio4, "tipsv2": _tipsv2, "lingbot_vision": _lingbot_vision, "dinov3_l": _dinov3_l, "r3m": _r3m,
    "qwen3vl_2b": _qwen3vl_2b, "cosmos_reason2": _cosmos_reason2, "groot_n17": _groot_n17, "molmoact2": _molmoact2,
    "wan22": _wan22, "wan22_512": _wan22_512, "lingbot_va": _lingbot_va,
    # §1.5 token-budget ablation: 16x16 tokens for two dense ViTs
    "vjepa2_1_t256": lambda d: _vjepa2_1(d, side=16), "cradio4_t256": lambda d: _cradio4(d, side=16),
    # Tier 2
    "vggt_omega": _vggt_omega, "cosmos3_gen": _cosmos3_gen,
    "qwen3vl_4b": lambda d: _qwen_vl(d, "qwen3vl_4b", "Qwen/Qwen3-VL-4B-Instruct", Q36, 36),
    "xiaomi_r1": lambda d: _qwen_vl(d, "xiaomi_r1", "Qwen/Qwen3-VL-4B-Instruct", Q36, 36, state_dict=_xr1_vlm_state_dict),
    "cosmos3_reason": lambda d: _qwen_vl(d, "cosmos3_reason", "Qwen/Qwen3-VL-8B-Instruct", Q36, 36,
                                         state_dict=_cosmos3_und_state_dict),
})


def load(name, device):
    if name not in ADAPTERS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[name](torch.device(device))
