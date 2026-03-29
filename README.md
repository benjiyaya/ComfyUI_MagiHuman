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
[DaVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman):Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model


Update
----
* add layer offload num to fit high vram 大显存 把offload开到你跑得动为止，小显存则从1开始测试，MagiCompiler库不用，但是改了麻烦，还是加了回来


1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_MagiHuman
```
2.requirements  
----

```
pip install -r requirements.txt

#python312以下，注释掉MagiCompiler目录下的pyproject.toml的13行的requires-python = ">=3.12" 再安装
git clone https://github.com/SandAI-org/MagiCompiler.git 
cd MagiCompiler
pip install -r requirements.txt
pip install .
```

3.checkpoints 
----
* dit and TE  [links](https://huggingface.co/smthem/daVinci-MagiHuman-custom-comfyUI) or 国内用户 [夸克](https://pan.quark.cn/s/26c7d9d39c87)

```
├── ComfyUI/models/
|     ├── diffusion_models/
|        ├──distill-merger_bf16.safetensors #28G
|        ├──540p_sr_merge_bf16.safetensors #28g For SR ,放大用开源不下
|     ├── vae/
|        ├──sd_audio.safetensors  #4.7GM
|        ├──Wan2.2_VAE.pth # 2.7G
|     ├── gguf
|        ├──t5gemma-9b-9b-ul2-Q6_K.gguf # 11G

```

4.Example
----
![](https://github.com/smthemex/ComfyUI_MagiHuman/blob/main/example_workflows/examplei2v.png)

![](https://github.com/smthemex/ComfyUI_MagiHuman/blob/main/example_workflows/examplemagi.png)


## 🙏 Acknowledgements

We thank the open-source community, and in particular [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [Turbo-VAED](https://github.com/hustvl/Turbo-VAED), for their valuable contributions.

## 📄 License

This project is released under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

## 📖 Citation

```bibtex
@misc{davinci-magihuman-2026,
  title   = {Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model},
  author  = {SII-GAIR and Sand.ai},
  year    = {2026},
  url     = {https://github.com/GAIR-NLP/daVinci-MagiHuman}
}
```
