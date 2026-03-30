from __future__ import annotations
import gc
from typing import Optional
import os
import torch
from transformers import AutoTokenizer
try :
    from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoderModel,T5GemmaConfig
except:
    from transformers.models.t5gemma import T5GemmaEncoderModel,T5GemmaConfig
from transformers import AutoModel, AutoTokenizer,AutoConfig
from safetensors.torch import load_file
from ...common import CPUOffloadWrapper, get_arch_memory
from ...utils import env_is_true
from contextlib import nullcontext
from accelerate import init_empty_weights
from diffusers.utils import is_accelerate_available

class T5GemmaEncoder:
    def __init__(self, model_path: str,gguf_path, device: str, weight_dtype: torch.dtype,repo):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        self.gguf_mode=False    
        if model_path is not None:
            if os.path.isfile(model_path):
                configs=T5GemmaConfig.from_pretrained(repo,is_encoder_decoder=False,model_type=weight_dtype,ignore_mismatched_sizes=True,low_cpu_mem_usage=False)
                with ctx():
                    model = T5GemmaEncoderModel(configs)
                model_dict=load_file(model_path)
                model_dict={k.replace("model.", ""): v for k, v in model_dict.items()}
                x,y=model.load_state_dict(model_dict,strict=False,assign=True)
                # print(x,"########_missing")
                # print(y,"########_unused")
                del model_dict
                gc.collect()

            else:
                model = T5GemmaEncoderModel.from_pretrained(
                    model_path,
                    is_encoder_decoder=False,
                    dtype=weight_dtype,
                )
            self.model = CPUOffloadWrapper(model, is_cpu_offload=env_is_true("CPU_OFFLOAD") or get_arch_memory() <= 48)
        elif gguf_path is not None:
            configs=T5GemmaConfig.from_pretrained(repo,is_encoder_decoder=False,model_type=weight_dtype,ignore_mismatched_sizes=True,low_cpu_mem_usage=False)
            with ctx():
                self.model = T5GemmaEncoderModel(configs)
            g_dict=load_gguf_checkpoint(gguf_path)
            #print(weight_dtype,"########weight_dtype")
            set_gguf2meta_model(self.model,g_dict,weight_dtype,torch.device("cpu"))
            del g_dict
            gc.collect()
            self.gguf_mode=True
            #self.model = CPUOffloadWrapper(model, is_cpu_offload=env_is_true("CPU_OFFLOAD") or get_arch_memory() <= 48)
    
    def encode(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        if self.gguf_mode:
            self.model.to(self.device)
        outputs = self.model(**inputs)
        if self.gguf_mode:
            self.model.to("cpu")
        return outputs["last_hidden_state"].half()


#_t5_gemma_cache: Optional[T5GemmaEncoder] = None


def get_t5_gemma_encoder(model_path: str,gguf_path, device: str, weight_dtype: torch.dtype,repo) -> T5GemmaEncoder:
    #global _t5_gemma_cache
    #if _t5_gemma_cache is None:
    _t5_gemma_cache = T5GemmaEncoder(model_path=model_path,gguf_path=gguf_path, device=device, weight_dtype=weight_dtype,repo=repo)
    return _t5_gemma_cache


@torch.inference_mode()
def get_t5_gemma_embedding(prompt: str, encoder) -> torch.Tensor:
    #encoder = get_t5_gemma_encoder(model_path=model_path, device=device, weight_dtype=weight_dtype)
    return encoder.encode(prompt)
@torch.inference_mode()
def get_t5_gemma_embedding_(prompt: str, model_path: str, device: str, weight_dtype: torch.dtype) -> torch.Tensor:
    encoder = get_t5_gemma_encoder(model_path=model_path, device=device, weight_dtype=weight_dtype)
    return encoder.encode(prompt)
def load_gguf_checkpoint(gguf_checkpoint_path):

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    from  diffusers.utils  import is_gguf_available, is_torch_available
    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    parsed_parameters = {}
  
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        quant_type = tensor.tensor_type

        
        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )

        weights = torch.from_numpy(tensor.data) #tensor.data.copy()
 
        parsed_parameters[name.replace("model.", "")] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
        del tensor,weights
        if i > 0 and i % 1000 == 0:  # 每1000个tensor执行一次gc
            logger.info(f"Processed {i}tensors...")
            gc.collect()
    del reader
    gc.collect()
    return parsed_parameters

def set_gguf2meta_model(meta_model,model_state_dict,dtype,device):
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer
    g_config = GGUFQuantizationConfig(compute_dtype=dtype or torch.bfloat16)
    hf_quantizer = GGUFQuantizer(quantization_config=g_config)
    hf_quantizer.pre_quantized = True


    hf_quantizer._process_model_before_weight_loading(
        meta_model,
        device_map={"": device} if device else None,
        state_dict=model_state_dict
    )
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
    x,y=load_model_dict_into_meta(
        meta_model, 
        model_state_dict, 
        hf_quantizer=hf_quantizer,
        device_map={"": device} if device else None,
        dtype=dtype
    )
    print(x,"offload_index")
    print(y,"state_dict_index")

    hf_quantizer._process_model_after_weight_loading(meta_model)

    
    del model_state_dict
    gc.collect()
    return meta_model.to(dtype=dtype)
