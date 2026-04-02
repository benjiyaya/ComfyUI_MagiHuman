![cover](assets/cover.png)


-----

<div align="center">

# daVinci-MagiHuman

### Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model

<p align="center">
  <a href="https://plms.ai">SII-GAIR</a> &nbsp;&amp;&nbsp; <a href="https://sand.ai">Sand.ai</a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2603.21986-b31b1b.svg)](https://arxiv.org/abs/2603.21986)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-HuggingFace-orange)](https://huggingface.co/spaces/SII-GAIR/daVinci-MagiHuman)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-HuggingFace-yellow)](https://huggingface.co/GAIR/daVinci-MagiHuman)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-ee4c2c.svg)](https://pytorch.org/)

</div>


ComfyUI_MagiHuman
----
[DaVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman): Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model


Updates (this fork)
----

This README documents **our fork** of ComfyUI_MagiHuman. Upstream is [daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman) / the original ComfyUI port. Below is everything we changed in code—**no model checkpoints were modified**.

### 1. Attention backends (Windows-safe, selectable)

Upstream assumed **FlashAttention-2** was always importable, which fails on Windows (e.g. `ModuleNotFoundError: No module named 'flash_attn.flash_attn_interface'`).

| File / area | What we fixed |
|---------------|----------------|
| `inference/model/dit/dit_module.py` | **Fallback order:** FlashAttention-3 (Hopper) → FlashAttention-2 if import works → else **PyTorch SDPA** (`scaled_dot_product_attention`). **GQA:** repeat K/V heads to match Q. **Flex / local attention:** when FA2 is unavailable, `_flash_attn_with_correction` uses an explicit float32 **LSE + softmax** path compatible with the online merge. **Detection:** `_flash_attn2_usable()` uses a real import test so broken/partial installs fall back cleanly. |
| `MagiHuman_node.py` | **`MagiHuman_SM_Model`** adds **`attn_mode`**: `auto` / `sage_attn` / `sdpa` / `flash_attn`. |
| `load_utils.py` | **`load_model(..., attn_mode=...)`** calls `set_attention_mode()` from `dit_module` when the model loads. |

| Mode | Behaviour |
|------|-----------|
| `auto` | FA3 (Hopper) → FA2 → SDPA |
| `sage_attn` | [SageAttention](https://github.com/thu-ml/SageAttention) if installed, else SDPA |
| `sdpa` | Always PyTorch SDPA |
| `flash_attn` | Force FA2/FA3 (fails if not installed) |

Optional: `pip install sageattention` (NVIDIA CUDA; not required on Windows if you use `sdpa` or `auto`).

---

### 2. `MagiHuman_LATENTS` — ComfyUI audio input

Upstream **`load_audio_and_encode`** only called **`whisper.load_audio(path)`** (ffmpeg subprocess). ComfyUI passes **in-memory audio** as `{"waveform": Tensor, "sample_rate": int}`, which caused subprocess / `fsdecode` errors.

| File | What we fixed |
|------|----------------|
| `inference/pipeline/video_process.py` | **Paths:** still use `whisper.load_audio`. **Dict:** `_comfy_audio_to_mono_numpy()` — batch 0, mono (mean channels if stereo), resample to **51200 Hz**, **clamp / peak-limit** to ~whisper-style **[-1, 1]** range before the VAE. |
| `load_utils.py` | Unchanged API; `get_latents` passes Comfy audio through correctly. |

---

### 3. Audio VAE encode dtype and decode quality

| Issue | Fix |
|-------|-----|
| **RuntimeError:** input float32 vs **bfloat16** bias in conv | After `torch.from_numpy`, tensors are cast to the **same device and dtype** as `audio_vae.vae_model` parameters before `encode`. |
| **Stochastic VAE bottleneck** (Gaussian sample every encode → hiss) | **`VAEBottleneck.encode(..., deterministic=True)`** uses the **mean** latent only at inference. `SAAudioFeatureExtractor.encode(..., deterministic=True)` by default; **`load_audio_and_encode`** uses **`audio_vae.encode(..., deterministic=True)`** instead of calling `vae_model.encode` directly. |
| **Playback resampling** | **`resample_waveform_to_playback()`** in `video_process.py`: prefers **`torchaudio.functional.resample`**, falls back to scipy. Uses **`AUDIO_ENCODE_SAMPLE_RATE` (51200)** → **`audio_vae.sample_rate`** (e.g. 44100) instead of a fragile hardcoded `441/512` ratio only. |
| `load_utils.py` **`decoder_audio`** | Uses the new resampler and target rate from the loaded VAE. |
| `inference/pipeline/video_generate.py` **`post_process`** | Same resampling helper when audio is decoded there. |

---

### 4. Layer offload (from upstream port)

Configurable **layer offload** count on the sampler for different VRAM levels (raise until stable; low VRAM start at `1`). **MagiCompiler** remains listed for compatibility even when not strictly needed for inference.

---

1. Installation
-----
In the `./ComfyUI/custom_nodes` directory, run:

```bash
git clone https://github.com/benjiyaya/ComfyUI_MagiHuman
```

Upstream / alternate: `https://github.com/smthemex/ComfyUI_MagiHuman`

2. Requirements
----

```bash
pip install -r requirements.txt

# If your Python version is below 3.12, comment out line 13
# (requires-python = ">=3.12") in MagiCompiler/pyproject.toml before installing.
git clone https://github.com/SandAI-org/MagiCompiler.git
cd MagiCompiler
pip install -r requirements.txt
pip install .
```

> **Windows note:** `flash-attn` does not have pre-built Windows wheels. You do **not** need to install it — the SDPA fallback in this fork handles inference automatically.

3. Checkpoints
----
* DiT and Text Encoder weights: [HuggingFace mirror](https://huggingface.co/smthem/daVinci-MagiHuman-custom-comfyUI) or (China users) [Quark cloud](https://pan.quark.cn/s/26c7d9d39c87)

```
├── ComfyUI/models/
|     ├── diffusion_models/
|        ├── distill-merger_bf16.safetensors    # 28 GB
|        ├── 540p_sr_merge_bf16.safetensors     # 28 GB  (SR / upscale model)
|     ├── vae/
|        ├── sd_audio.safetensors               # 4.7 GB
|        ├── Wan2.2_VAE.pth                     # 2.7 GB
|     ├── gguf/
|        ├── t5gemma-9b-9b-ul2-Q6_K.gguf       # 11 GB
```

4. Example Workflows
----
![](https://github.com/smthemex/ComfyUI_MagiHuman/blob/main/example_workflows/examplei2v.png)

![](https://github.com/smthemex/ComfyUI_MagiHuman/blob/main/example_workflows/examplemagi.png)


## Acknowledgements

We thank the open-source community, and in particular [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [Turbo-VAED](https://github.com/hustvl/Turbo-VAED), for their valuable contributions.

## License

This project is released under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

## Citation

```bibtex
@misc{davinci-magihuman-2026,
  title   = {Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model},
  author  = {SII-GAIR and Sand.ai},
  year    = {2026},
  url     = {https://github.com/GAIR-NLP/daVinci-MagiHuman}
}
```
