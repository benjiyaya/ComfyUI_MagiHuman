 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
from .load_utils import (load_model, load_vae, load_audio_vae, en_decoder_video, decoder_audio,
                         load_clip, encoder_text, read_lat_emb, save_lat_emb, get_latents)

ATTN_MODE_OPTIONS = ["auto", "sage_attn", "sdpa", "flash_attn"]
from .model_loader_utils import clear_comfyui_cache
from .inference.pipeline.entry import infer_magihuman

MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir


class MagiHuman_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagiHuman_SM_Model",
            display_name="MagiHuman_SM_Model",
            category="MagiHuman_SM",
            inputs=[
                io.Combo.Input("dit", options=["none"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("sr_dit", options=["none"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("gguf", options=["none"] + folder_paths.get_filename_list("gguf")),
                io.Combo.Input("sr_gguf", options=["none"] + folder_paths.get_filename_list("gguf")),
                io.Combo.Input("attn_mode", options=ATTN_MODE_OPTIONS),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, dit, sr_dit, gguf, sr_gguf, attn_mode) -> io.NodeOutput:
        clear_comfyui_cache()
        model = load_model(dit, sr_dit, gguf, sr_gguf, attn_mode=attn_mode)
        return io.NodeOutput(model)

class MagiHuman_SM_VAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="MagiHuman_SM_VAE",
            display_name="MagiHuman_SM_VAE",
            category="MagiHuman_SM",
            inputs=[
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
                io.Combo.Input("turbo_vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
            ],
            outputs=[io.Vae.Output(display_name="vae"),],
            )
    @classmethod
    def execute(cls,vae,turbo_vae ) -> io.NodeOutput:
        clear_comfyui_cache()
        vae=load_vae(vae,turbo_vae,device,torch.bfloat16) 
        return io.NodeOutput(vae)
    
class MagiHuman_SM_Clip(io.ComfyNode):
    @classmethod
    def define_schema(cls):       
        return io.Schema(
            node_id="MagiHuman_SM_Clip",
            display_name="MagiHuman_SM_Clip",
            category="MagiHuman_SM",
            inputs=[
                io.Combo.Input("clip",options= ["none"] + folder_paths.get_filename_list("clip") ),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf") ),
            ],
            outputs=[io.Clip.Output(display_name="clip"),],
            )
    @classmethod
    def execute(cls,clip,gguf ) -> io.NodeOutput:
        clear_comfyui_cache()
        clip=load_clip(clip,gguf,device)     
        return io.NodeOutput(clip)
    
class MagiHuman_SM_AUDIO_VAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):      
        return io.Schema(
            node_id="MagiHuman_SM_AUDIO_VAE",
            display_name="MagiHuman_SM_AUDIO_VAE",
            category="MagiHuman_SM",
            inputs=[
                io.Combo.Input("audio_vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
            ],
            outputs=[io.Vae.Output(display_name="audio_vae"),],
            )
    @classmethod
    def execute(cls,audio_vae, ) -> io.NodeOutput:
        clear_comfyui_cache()
        audio_vae=load_audio_vae(audio_vae,device)
        return io.NodeOutput(audio_vae)

class MagiHuman_EN_DECO_VIDEO(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagiHuman_EN_DECO_VIDEO",
            display_name="MagiHuman_EN_DECO_VIDEO",
            category="MagiHuman_SM",
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                ],
            )
    @classmethod
    def execute(cls,vae,latent,) -> io.NodeOutput:
        clear_comfyui_cache()
        video=en_decoder_video(vae,latent)  
        print(video.shape) #torch.Size([249, 256, 448, 3])

        return io.NodeOutput(video)

class MagiHuman_DECO_AUDIO(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="MagiHuman_DECO_AUDIO",
            display_name="MagiHuman_DECO_AUDIO",
            category="MagiHuman_SM",
            inputs=[
                io.Vae.Input("audio_vae"),
                io.Latent.Input("audio_latents"),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
                ],
            )
    @classmethod
    def execute(cls,audio_vae,audio_latents,) -> io.NodeOutput:
        clear_comfyui_cache()
        audio=decoder_audio(audio_vae,audio_latents,device)
        return io.NodeOutput(audio,None)
      
        
class MagiHuman_LATENTS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagiHuman_LATENTS",
            display_name="MagiHuman_LATENTS",
            category="MagiHuman_SM",
            inputs=[
                io.Int.Input("width", default=448, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=256, min=256, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("sr_width", default=896 , min=0, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("sr_height", default=512, min=0, max=nodes.MAX_RESOLUTION,step=32,display_mode=io.NumberDisplay.number),
                io.Int.Input("seconds", default=10, min=1, max=MAX_SEED,step=1,display_mode=io.NumberDisplay.number),
                io.Vae.Input("vae",optional=True),
                io.Vae.Input("audio_vae",optional=True),
                io.Image.Input("image",optional=True),
                io.Audio.Input("audio",optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                ],
            )
    @classmethod
    def execute(cls,width,height,sr_width,sr_height,seconds,vae=None,audio_vae=None,image=None,audio=None,) -> io.NodeOutput:
        clear_comfyui_cache() 
        # width=(width //32)*32 if width % 32 != 0  else width 
        # height=(height //32)*32 if height % 32 != 0  else height
        output=get_latents(vae,image,audio_vae,audio,width,height,sr_width,sr_height,device,seconds,)
        return io.NodeOutput(output)

    
class MagiHuman_SM_ENCODER(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="MagiHuman_SM_ENCODER",
            display_name="MagiHuman_SM_ENCODER",
            category="MagiHuman_SM",
            inputs=[
                io.Clip.Input("clip"),
                io.Boolean.Input("save_emb",default=False),
                io.String.Input("prompt",multiline=True,default="A close-up of a cheerful girl puppet with curly auburn yarn hair and wide button eyes, " \
                "holding a small red umbrella above her head. Rain falls gently around her. She looks upward and begins to sing with joy in English: It's raining," \
                " it's raining, I love it when its raining. Her fabric mouth opening and closing to a melodic tune. Her hands grip the umbrella handle as she sways slightly from side to side in rhythm. The camera holds steady as the rain sparkles against the soft lighting. Her eyes blink occasionally as she sings."),
                io.String.Input("negative_prompt",multiline=True,default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards," \
                " low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, static, compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy, unclear, echo, reverb, room echo, over-reverberated, hollow sound, distant, washed out, harsh, shrill, piercing, grating, tinny, thin sound, boomy, bass-heavy, flat EQ, over-compressed, abrupt cut, jarring transition, sudden silence, looping artifact, music, instrumental, sirens, alarms, crowd noise, unrelated sound effects, chaotic, disorganized, messy, cheap sound " \
                ", emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, flat intonation, undynamic, boring, reading from a script, AI voice, synthetic, text-to-speech, TTS, insincere, fake emotion, exaggerated, overly dramatic, melodramatic, cheesy, cringey, hesitant, unconfident, tired, weak voice, stuttering, stammering, mumbling, slurred speech, mispronounced, bad articulation, lisp, vocal fry, creaky voice, mouth clicks, lip smacks, wet mouth sounds, heavy breathing, audible inhales, plosives, p-pops, coughing, clearing throat, sneezing, speaking too fast, rushed, speaking too slow, dragged out, unnatural pauses, awkward silence, choppy, disjointed, multiple speakers, two voices, background talking, out of tune, off-key, autotune artifacts"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                ],
            )
    @classmethod
    def execute(cls,clip,save_emb,prompt,negative_prompt, ) -> io.NodeOutput:
        clear_comfyui_cache()
        positive,negative=encoder_text(clip,prompt,negative_prompt,save_emb)
        return io.NodeOutput(positive,negative)

class MagiHuman_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagiHuman_SM_KSampler",
            display_name="MagiHuman_SM_KSampler",
            category="MagiHuman_SM",
            inputs=[
                io.Model.Input("model"),     
                io.Latent.Input("latents",),    
                io.Int.Input("steps", default=8, min=1, max=nodes.MAX_RESOLUTION,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("offload", default=True),
                io.Int.Input("offload_block_num", default=1, min=1, max=40,step=1,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("save_latents", default=True),
                io.Boolean.Input("pass_stage1", default=False),
                io.Conditioning.Input("positive",optional=True),
                io.Conditioning.Input("negative",optional=True),  
            ], 
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Latent.Output(display_name="audio_latents"),
            ],
        )
    @classmethod
    def execute(cls, model,latents,steps,seed,offload,offload_block_num,save_latents,pass_stage1,positive=None,negative=None,) -> io.NodeOutput:
        if positive is None:
            positive,negative=read_lat_emb("embeds", positive, negative,device)
        clear_comfyui_cache()
        if pass_stage1:
            video_latents,audio_latents=read_lat_emb("latents", positive, negative,device)
        else:
            latents["positives"]=positive
            latents["negatives"]=negative
            video_lat, audio_lat,params=infer_magihuman(model,seed,latents,steps,sr_steps=50,offload=offload,offload_block_num=offload_block_num)
            video_latents={"samples":video_lat}
            if params:  
                latents["params"]=params
            latents["samples"]=audio_lat
            latents["seed"]=seed
            audio_latents=latents
            if save_latents:
                save_lat_emb("latents",video_latents,audio_latents,model.infer_mode)
        return io.NodeOutput(video_latents, audio_latents)

class MagiHuman_SM_SRSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagiHuman_SM_SRSampler",
            display_name="MagiHuman_SM_SRSampler",
            category="MagiHuman_SM",
            inputs=[
                io.Model.Input("model"),      
                io.Latent.Input("latents"), 
                io.Latent.Input("audio_latents"),  
                io.Int.Input("sr_steps", default=5, min=1, max=nodes.MAX_RESOLUTION,step=1,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("offload", default=True),
            ], 
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Latent.Output(display_name="audio_latents"),
            ],
        )
    @classmethod
    def execute(cls, model,latents,audio_latents,sr_steps,offload) -> io.NodeOutput:
        clear_comfyui_cache()
        audio_latents["video_latents"]=latents
        video, audio,params=infer_magihuman(model,audio_latents["seed"],audio_latents,steps=8,sr_steps=sr_steps,sr_mode=True,offload=offload)
        latents["samples"]=video
        audio_latents["samples"]= audio
        return io.NodeOutput(latents, audio_latents)

class  MagiHuman_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MagiHuman_SM_Model,
            MagiHuman_SM_VAE,
            MagiHuman_SM_Clip,
            MagiHuman_SM_AUDIO_VAE,
            MagiHuman_EN_DECO_VIDEO,
            MagiHuman_DECO_AUDIO,
            MagiHuman_LATENTS,
            MagiHuman_SM_ENCODER,
            MagiHuman_SM_KSampler,
            MagiHuman_SM_SRSampler,
        ]
async def comfy_entrypoint() -> MagiHuman_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return MagiHuman_SM_Extension()
