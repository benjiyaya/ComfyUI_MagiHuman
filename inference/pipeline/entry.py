# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from ..common.config import MagiPipelineConfig
try:
    from .pipeline import MagiPipeline
except ImportError:
    # Keep compatibility when entry.py is executed as a script path.
    from . import MagiPipeline



def load_magihuman(dit_path,gguf_path,sr_dit_path,sr_gguf_path): # Load MAGIHuman model
    config = MagiPipelineConfig()
    config.engine_config.load=dit_path
    config.evaluation_config.use_sr_model=True if sr_dit_path is not None else False
    config.evaluation_config.sr_model_path=sr_dit_path 
    is_distill=True if "distill" in dit_path  else False
    pipeline = MagiPipeline(None, config.evaluation_config,config,is_distill)
    return pipeline


def infer_magihuman(pipeline,seed,conds,steps,sr_steps,sr_mode=False,offload=False,offload_block_num=1):
    optional_kwargs = {
    "seed": seed,
    "seconds": conds["seconds"],
    "br_width": conds["br_width"],
    "br_height": conds["br_height"],
    "sr_width": conds["sr_width"],
    "sr_height": conds["sr_height"],
    "output_width": None,
    "output_height": None,
    "upsample_mode": "bilinear",
    }

    optional_kwargs = {k: v for k, v in optional_kwargs.items() if v is not None and v is not False}
   
    latent_video, latent_audio,params=pipeline.run_offline(
        prompt=None, image=None, audio=None, save_path_prefix="save_path_prefix",conds=conds,sr_mode=sr_mode,steps=steps,sr_steps=sr_steps,offload=offload,offload_block_num=offload_block_num, **optional_kwargs
    )
    return latent_video, latent_audio,params

