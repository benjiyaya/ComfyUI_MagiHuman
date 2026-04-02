 # !/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
import torch
import os
import gc
import folder_paths
from .inference.model.turbo_vaed import TurboVAED, get_turbo_vaed
from .inference.model.vae2_2 import Wan2_2_VAE, get_vae2_2
from .inference.common import CPUOffloadWrapper, get_arch_memory
from .inference.model.sa_audio import SAAudioFeatureExtractor
from .model_loader_utils import nomarl_upscale
from .inference.pipeline.entry import load_magihuman
from .inference.model.dit.dit_module import set_attention_mode

from .inference.pipeline.video_process import (
    AUDIO_ENCODE_SAMPLE_RATE,
    load_audio_and_encode,
    resample_waveform_to_playback,
    resizecrop,
)
from .inference.pipeline.video_generate import encode_image_w,decode_video_w
from .inference.pipeline.prompt_process import pad_or_trim
from .inference.model.t5_gemma.t5_gemma_model import  get_t5_gemma_embedding,get_t5_gemma_encoder
node_cr_path_ = os.path.dirname(os.path.abspath(__file__))
 

def load_model(dit, sr_dit, gguf, sr_gguf, attn_mode="auto"):
    dit_path=folder_paths.get_full_path("diffusion_models", dit) if dit != "none" else None
    gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
    sr_dit_path=folder_paths.get_full_path("diffusion_models", sr_dit) if sr_dit != "none" else None
    sr_gguf_path=folder_paths.get_full_path("gguf", sr_gguf) if sr_gguf != "none" else None
    set_attention_mode(attn_mode)
    model=load_magihuman(dit_path,gguf_path,sr_dit_path,sr_gguf_path)
    model.infer_mode="sr" if sr_gguf_path is not None or sr_dit_path is not None else "base"
    return model
    

def load_clip(clip,gguf,device):
    clip_path=folder_paths.get_full_path("clip", clip) if clip != "none" else None
    gguf_path=folder_paths.get_full_path("gguf", gguf) if gguf != "none" else None
    repo=os.path.join(node_cr_path_,"t5gemma-9b-9b-ul2")
    #assert clip_path is not None,"Please provide a clip_path"
    #clip_path="D:/Downloads/t5gemma-9b-9b-ul2"
    text_encoder=get_t5_gemma_encoder(clip_path,gguf_path,device,torch.bfloat16,repo)
    return text_encoder

def encoder_text(text_encoder,prompt,negative_prompt,save_emb,target_length=640):
   
    with torch.no_grad():
        txt_feat=get_t5_gemma_embedding(prompt, text_encoder) 
        txt_feat, original_len=pad_or_trim(txt_feat, target_size=target_length, dim=1)
        txt_feat=txt_feat.to(torch.float32)
        txt_feat_null=get_t5_gemma_embedding(negative_prompt, text_encoder) 
        txt_feat_null, original_len_null=pad_or_trim(txt_feat_null, target_size=target_length, dim=1)
        txt_feat_null=txt_feat_null.to(torch.float32)
    torch.cuda.empty_cache()
    gc.collect()
    #print(txt_feat.shape,txt_feat_null.shape) #torch.Size([1, 640, 3584]) torch.Size([1, 640, 3584])
    positive=[[txt_feat,{"pooled_output": original_len}]]
    negative=[[txt_feat_null,{"pooled_output": original_len_null}]]
    if save_emb:
        save_lat_emb("embeds",positive,negative)
    return positive, negative


def load_vae(vae,turbo_vae,device,weight_dtype):
    vae_model_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
    student_ckpt_path=folder_paths.get_full_path("vae", turbo_vae) if turbo_vae != "none" else None
    
    if vae_model_path is not None:
        vae = CPUOffloadWrapper(
        get_vae2_2(vae_model_path, device, weight_dtype=weight_dtype), is_cpu_offload=get_arch_memory() <= 48
        )
        vae.model.use_turbo_vae = False
    elif student_ckpt_path is not None:
        student_config_path=os.path.join(node_cr_path_,"example/TurboV3-Wan22-TinyShallow_7_7.json")
        vae = CPUOffloadWrapper(
            get_turbo_vaed(student_config_path, student_ckpt_path, device, weight_dtype=weight_dtype),
            is_cpu_offload=get_arch_memory() <= 48,
        )
        vae.model.use_turbo_vae = True
    else:
        raise ValueError("Please provide a vae_model_path or student_ckpt_path")
    return vae

def load_audio_vae(audio_vae,device):
    #vocoder_path=folder_paths.get_full_path("vae", vocoder) if vocoder != "none" else None    
    vae_path=folder_paths.get_full_path("vae", audio_vae) if audio_vae != "none" else None
    repo=os.path.join(node_cr_path_,"stable-audio-open")
    assert vae_path is not None,"Please provide a vae_path"
    audio_vae = SAAudioFeatureExtractor(device=device, model_path=vae_path,repo=repo)
    return audio_vae



def en_decoder_video(vae,latent):
    lat=latent["samples"]
    with torch.no_grad():
        videos=decode_video_w(vae,lat,torch.bfloat16)
    return videos


def get_latents(vae,image,audio_vae,audio,width,height,sr_width,sr_height,device,seconds,):
    if image is not None and vae is not None:
        br_image=nomarl_upscale(image, width, height)
        br_image = encode_image_w(vae,br_image, height, width,device,torch.bfloat16)
        if sr_width>0 and sr_height>0:
            sr_image = nomarl_upscale(image, sr_width, sr_height)
            sr_image = encode_image_w(vae,sr_image, sr_height, sr_width, device,torch.bfloat16)
    else:
        br_image = None
        sr_image = None

    if audio is not None and audio_vae is not None:
        latent_audio = load_audio_and_encode(audio_vae, audio, seconds)
    else:
        latent_audio=None
           
    output={"latent_audio":latent_audio,"br_image":br_image,"sr_image":sr_image,"br_width":width,"br_height":height,"sr_width":sr_width,"sr_height":sr_height,"seconds":seconds}
    return output


def decoder_audio(audio_vae,audio_latents,device):
    latent_audio=audio_latents["samples"].to(torch.bfloat16)
    latent_audio = latent_audio.squeeze(0)
    audio_output = audio_vae.decode(latent_audio.T)
    audio_output_np = audio_output.squeeze(0).T.cpu().float().numpy()
    target_sr = int(getattr(audio_vae, "sample_rate", 44100))
    audio_output_np = resample_waveform_to_playback(
        audio_output_np, AUDIO_ENCODE_SAMPLE_RATE, target_sr
    )
    torch.cuda.empty_cache()
    gc.collect()
    audio_output_np=torch.from_numpy(audio_output_np).to(device)
    print(audio_output_np.shape) #torch.Size([442764, 2])

    return {"waveform": audio_output_np.permute(1,0).contiguous().reshape(1, -1).cpu().float().unsqueeze(0), "sample_rate": audio_vae.sample_rate}



def read_lat_emb(prefix, positive, negative,device):
    if prefix =="embeds":
        if not  os.path.exists(os.path.join(folder_paths.get_output_directory(),"raw_embeds_MagiHuman_sm.pt")):
            raise Exception("No backup prompt embeddings found. Please run MagiHuman_SM_ENCODER node first.")
        else:
            prompt_embeds=torch.load(os.path.join(folder_paths.get_output_directory(),"raw_embeds_MagiHuman_sm.pt"),weights_only=False)
            if os.path.exists(os.path.join(folder_paths.get_output_directory(),"n_raw_embeds_MagiHuman_sm.pt")):
                negative_prompt_embeds=torch.load(os.path.join(folder_paths.get_output_directory(),"n_raw_embeds_MagiHuman_sm.pt"),weights_only=False)
            else:
                negative_prompt_embeds=[[torch.zeros_like(prompt_embeds[0][0]),prompt_embeds[0][1]]] 
        #print("Loaded backup prompt embeddings",prompt_embeds[0][0].shape) # Loaded backup prompt embeddings torch.Size([1, 640, 3584])
        positive=[[prompt_embeds[0][0].to(device,torch.bfloat16),prompt_embeds[0][1]]]
        negative=[[negative_prompt_embeds[0][0].to(device,torch.bfloat16),negative_prompt_embeds[0][1]]]
        #print(positive[0][0].shape,negative[0][0].shape) #torch.Size([1, 640, 3584]) torch.Size([1, 640, 3584])
        #print(negative[0][1]["pooled_output"],positive[0][1]["pooled_output"]) #392 114
        return positive,negative
    
    elif prefix =="latents":
        if not  os.path.exists(os.path.join(folder_paths.get_output_directory(),"raw_latents_MagiHuman_sm.pt")) or not os.path.exists(os.path.join(folder_paths.get_output_directory(),"raw_audio_latents_MagiHuman_sm.pt")):
            raise Exception("No backup latents found. Please run MagiHuman_SM_KSampler node first.")
        else:
            video_latents=torch.load(os.path.join(folder_paths.get_output_directory(),"raw_latents_MagiHuman_sm.pt"),weights_only=False)
            audio_latents=torch.load(os.path.join(folder_paths.get_output_directory(),"raw_audio_latents_MagiHuman_sm.pt"),weights_only=False)
            #print("Loaded backup latents",video_latents.shape,audio_latents.shape) #([1, 128, 11, 16, 24]) [1, 8, 84, 16] torch.Size([1, 84, 128])

        video_latents["samples"]=video_latents["samples"].to(device,torch.bfloat16)
        audio_lat=audio_latents["samples"].to(device,torch.bfloat16)  # [1, 8, 84, 16]
        print(f"audio shape: {audio_lat.shape}")  #audio shape: torch.Size([1, 84, 8, 16])
        print(f"video shape: {video_latents['samples'].shape}")
        # batch, frames, combined_dim = audio_lat.shape
        # reshaped = audio_lat.view(batch, frames, 8, 16)
        #audio_lat = audio_lat.permute(0, 2, 1, 3)
       
        audio_latents["samples"]=audio_lat
        return video_latents, audio_latents
    
def  save_lat_emb(save_prefix,data1,data2,mode=""):
    data1_prefix, data2_prefix = ("raw_embeds_MagiHuman", "n_raw_embeds_MagiHuman") if save_prefix == "embeds" else ("raw_latents_MagiHuman", "raw_audio_latents_MagiHuman")
    default_data1_path = os.path.join(folder_paths.get_output_directory(),f"{data1_prefix}_sm.pt")
    default_data2_path = os.path.join(folder_paths.get_output_directory(),f"{data2_prefix}_sm.pt")
    prefix = mode+str(int(time.time()))
    if os.path.exists(default_data1_path): # use a different path if the file already exists
        default_data1_path=os.path.join(folder_paths.get_output_directory(),f"{data1_prefix}_sm_{prefix}.pt")
    torch.save(data1,default_data1_path)
    if data2 is not None:
        if os.path.exists(default_data2_path):
            default_data2_path=os.path.join(folder_paths.get_output_directory(),f"{data2_prefix}_sm_{prefix}.pt")
        torch.save(data2,default_data2_path)