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


Updates
----

### [2026-04-03] Attention Mode Selection + Windows / SDPA Compatibility Fork (this edition)

The upstream codebase hard-requires `flash-attn` (FlashAttention-2/3), which **does not build on Windows** and causes the following error at inference time:

```
ModuleNotFoundError: No module named 'flash_attn.flash_attn_interface'
```

This fork adds a selectable **Attention Mode** dropdown to the `MagiHuman_SM_Model` node, replacing the hard `flash_attn` dependency with a four-way backend:

| Mode | Backend | Notes |
|------|---------|-------|
| `auto` *(default)* | FA3 → FA2 → **SDPA** | Best available; safe on Windows |
| `sage_attn` | **SageAttention** → SDPA fallback | Fastest on NVIDIA CUDA (pip install sageattention) |
| `sdpa` | **PyTorch SDPA** | Always works; Windows, macOS, CPU |
| `flash_attn` | FA3 / FA2 (forced) | Errors if flash-attn is not installed |

All paths handle **grouped-query attention (GQA)** automatically. The `flex_flash_attn` online-softmax merge path (`_flash_attn_with_correction`) routes through the same selection.

**How to install SageAttention (optional, Linux/CUDA only):**
```bash
pip install sageattention
```

**No model weights are changed — this is a drop-in fix.**

---

### [earlier] Layer Offload Configuration

Added a configurable layer-offload count to accommodate different VRAM budgets.
- **High VRAM**: increase the offload count until you find your limit.
- **Low VRAM**: start from `1` and increase incrementally.
- The `MagiCompiler` library is not strictly required for inference but is kept as a dependency to avoid refactoring overhead.


1. Installation
-----
In the `./ComfyUI/custom_nodes` directory, run:

```bash
git clone https://github.com/smthemex/ComfyUI_MagiHuman
```

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
