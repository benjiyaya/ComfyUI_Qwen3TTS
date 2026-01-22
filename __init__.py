import os
import gc
import torch
import soundfile as sf
import numpy as np
import folder_paths
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel

# Register the "Qwen3TTS" folder type
folder_paths.add_model_folder_path("Qwen3TTS", os.path.join(folder_paths.base_path, "models", "Qwen3TTS"))

def get_model_base_dir():
    paths = folder_paths.get_folder_paths("Qwen3TTS")
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

def convert_audio(wav, sr):
    
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    
    if wav.dim() == 1:
        wav = wav.unsqueeze(0) # (1, samples) (channels=1)
    

    if wav.shape[0] > wav.shape[1]: 
        # assume (samples, channels) - verify this assumption
        wav = wav.transpose(0, 1)
        

    wav = wav.unsqueeze(0) # (1, channels, samples)
    
    return {"waveform": wav, "sample_rate": sr}

def load_audio_input(audio_input):

    if audio_input is None:
        return None
        
    waveform = audio_input["waveform"]
    sr = audio_input["sample_rate"]
    
    wav = waveform[0] # (channels, samples)

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0) # Mix to mono
    else:
        wav = wav.squeeze(0) # (samples,)
        
    return (wav.numpy(), sr)


class Qwen3Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": ([
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
                ], {"default": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("QWEN3_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3-TTS"

    def load_model(self, repo_id, precision, attention):
        # Use the custom path support logic
        base_dir = get_model_base_dir()
        local_model_path = os.path.join(base_dir, repo_id)

        # Check existence and Download if necessary
        config_file = os.path.join(local_model_path, "config.json")
        
        if not os.path.exists(config_file):
            print(f"[Qwen3-TTS] Model not found at {local_model_path}. Downloading from HuggingFace...")
            try:
                os.makedirs(local_model_path, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_model_path,
                    local_dir_use_symlinks=False, 
                    resume_download=True
                )
                print(f"[Qwen3-TTS] Download complete: {local_model_path}")
            except Exception as e:
                raise Exception(f"[Qwen3-TTS] Failed to download model: {e}")
        else:
            print(f"[Qwen3-TTS] Loading existing model from: {local_model_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        dtype = torch.float32
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
            
        # Determine attention implementation
        attn_impl = "sdpa" 
        
        if attention == "flash_attention_2":
            # Explicit check for flash_attn installation
            try:
                import flash_attn
                import importlib.metadata
                importlib.metadata.version("flash_attn")
                attn_impl = "flash_attention_2"
            except ImportError:
                raise ModuleNotFoundError(
                    print(f"Flash Attention 2 is selected, but it is not installed.")
                )
        elif attention == "auto":
            # Auto-detect
            try:
                import flash_attn
                import importlib.metadata
                importlib.metadata.version("flash_attn")
                attn_impl = "flash_attention_2"
            except Exception:
                # Fallback to sdpa if flash_attn missing or metadata broken
                print(f"Flash Attention is selected, but it is not installed.")
                attn_impl = "sdpa"
        else:
            # Use whatever user specified (sdpa or eager)
            attn_impl = attention

        print(f"[Qwen3-TTS] Device: {device}, Precision: {dtype}, Attention: {attn_impl}")

        model = Qwen3TTSModel.from_pretrained(
            local_model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl
        )
        
        return (model,)


class Qwen3CustomVoice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "text": ("STRING", {"multiline": True}),
                "language": ([
                    "Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                    "French", "Russian", "Portuguese", "Spanish", "Italian"
                ], {"default": "Auto"}),
                "speaker": ([
                    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", 
                    "Ryan", "Aiden", "Ono_Anna", "Sohee"
                ], {"default": "Vivian"}),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, model, text, language, speaker, instruct=""):
        lang = language if language != "Auto" else None
        inst = instruct if instruct.strip() != "" else None
        
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=lang,
            speaker=speaker,
            instruct=inst
        )
        
        result = convert_audio(wavs[0], sr)

        # Cleanup memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return (result,)


class Qwen3VoiceDesign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "text": ("STRING", {"multiline": True}),
                "instruct": ("STRING", {"multiline": True}),
                "language": ([
                    "Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                    "French", "Russian", "Portuguese", "Spanish", "Italian"
                ], {"default": "Auto"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, model, text, instruct, language):
        lang = language if language != "Auto" else None
        
        wavs, sr = model.generate_voice_design(
            text=text,
            language=lang,
            instruct=instruct
        )

        result = convert_audio(wavs[0], sr)

        # Cleanup memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (result,)


class Qwen3TTSVoiceClonePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("QWEN3_PROMPT",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3-TTS"

    def create_prompt(self, model, ref_audio, ref_text):
        audio_tuple = load_audio_input(ref_audio)
        
        prompt = model.create_voice_clone_prompt(
            ref_audio=audio_tuple,
            ref_text=ref_text
        )

        # Cleanup memory (clears the processed reference audio tensors)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (prompt,)


class Qwen3VoiceClone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_MODEL",),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "language": ([
                    "Auto", "Chinese", "English", "Japanese", "Korean", "German", 
                    "French", "Russian", "Portuguese", "Spanish", "Italian"
                ], {"default": "Auto"}),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True}),
                "prompt": ("QWEN3_PROMPT",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, model, text, language="Auto", ref_audio=None, ref_text=None, prompt=None):
        lang = language if language != "Auto" else None
        
        wavs = None
        sr = 0
        
        if prompt is not None:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=lang,
                voice_clone_prompt=prompt
            )
        elif ref_audio is not None and ref_text is not None and ref_text.strip() != "":
            audio_tuple = load_audio_input(ref_audio)
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=lang,
                ref_audio=audio_tuple,
                ref_text=ref_text
            )
        else:
             raise ValueError("For Voice Clone, you must provide either 'prompt' OR ('ref_audio' AND 'ref_text').")
             
        result = convert_audio(wavs[0], sr)

        # Cleanup memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (result,)
    


NODE_CLASS_MAPPINGS = {
    "Qwen3Loader": Qwen3Loader,
    "Qwen3CustomVoice": Qwen3CustomVoice,
    "Qwen3VoiceDesign": Qwen3VoiceDesign,
    "Qwen3VoiceClone": Qwen3VoiceClone,
    "Qwen3TTSVoiceClonePrompt": Qwen3TTSVoiceClonePrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3Loader": "Qwen3-TTS Loader",
    "Qwen3CustomVoice": "Qwen3-TTS Custom Voice",
    "Qwen3VoiceDesign": "Qwen3-TTS Voice Design",
    "Qwen3VoiceClone": "Qwen3-TTS Voice Clone",
    "Qwen3TTSVoiceClonePrompt": "Qwen3-TTS Voice Clone Prompt"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]