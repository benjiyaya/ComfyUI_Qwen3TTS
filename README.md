# Development In Progress.....



# ComfyUI-Qwen3TTS

A ComfyUI custom node implementation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). This node allows you to generate high-quality text-to-speech audio directly within ComfyUI, featuring support for Voice Design, Custom Voice, and advanced Audio Tokenization.

![ComfyUI-Qwen3TTS](https://img.shields.io/badge/ComfyUI-Custom_Node-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

## Features

*   **Multiple Model Support**: Compatible with all 5 released Qwen3-TTS variants (1.7B and 0.6B).
*   **Extreme Memory Optimization**:
    *   **4-bit Quantization**: Run 1.7B models on GPUs with <8GB VRAM using BitsAndBytes.
    *   **CPU Offload**: Dynamic model splitting for GPUs with limited memory.
*   **NVIDIA Blackwell Support**: Optimized for the latest architectures with `bfloat16` compute dtype support.
*   **Audio Tokenization Pipeline**: Separate nodes for encoding (Audio $\to$ Codes) and decoding (Codes $\to$ Audio), enabling latent audio manipulation.
*   **Custom Path Support**: Fully supports `extra_model_paths.yaml` for custom model directories.
*   **Advanced Inference**:
    *   **VoiceDesign**: Modify voice characteristics using text instructions (e.g., "A deep male voice").
    *   **CustomVoice**: Clone voice timbre using a reference audio clip.

## Installation

1.  **Navigate to your ComfyUI `custom_nodes` directory:**
    ```bash
    cd ComfyUI/custom_nodes
    ```

2.  **Clone this repository:**
    ```bash
    git clone https://github.com/benjiyaya/ComfyUI_Qwen3TTS.git
    ```

3.  **Install Dependencies:**
    Navigate into the cloned folder and install the required Python libraries:
    ```bash
    cd ComfyUI_Qwen3TTS
    pip install -r requirements.txt
    ```

    *Note: `bitsandbytes` is required for 4-bit quantization. Ensure your PyTorch and CUDA versions are compatible.*

4.  **Restart ComfyUI.**

## Requirements

The `requirements.txt` includes the following:
*   `transformers`
*   `accelerate` (Required for offloading)
*   `bitsandbytes` (Required for 4-bit quantization)
*   `huggingface_hub`
*   `torchaudio`
*   `sentencepiece`

## Nodes

### 1. Qwen3-TTS Model Loader

Loads the Qwen3-TTS model and processor from HuggingFace or local disk. Supports standard loading and 4-bit quantization.

**Inputs:**
*   **model_name**: Select from the supported variants (1.7B or 0.6B).
*   **dtype**: Precision for non-quantized weights (`float16`, `bfloat16`, etc.).
*   **quantization**: 
    *   `none`: Load full weights.
    *   `4-bit`: Load weights in 4-bit (NF4) to drastically reduce VRAM usage (requires `bitsandbytes`).
*   **bnb_compute_dtype**: When using 4-bit, select the computation precision.
    *   `bfloat16`: Recommended for NVIDIA Blackwell, Hopper, and Ampere GPUs.
    *   `float16`: Fallback for older GPUs.
*   **offload** (Boolean): Splits model between GPU and CPU. *Note: Automatically enabled when using 4-bit.*

**Outputs:**
*   `model`: The loaded model object.

### 2. Qwen3-TTS Tokenizer (New)

Converts raw audio waveforms into discrete audio codes (Tokens).

**Inputs:**
*   **model**: Connected from Model Loader.
*   **audio**: Input audio (e.g., a voice sample).

**Outputs:**
*   `codes`: Discrete tensor representation of the audio.

**Use Case:**
*   Extracting features from a voice sample before generation.
*   Inspecting the latent representation of audio.
*   Can be piped into the Reconstruction node to test the encode/decode loop.

### 3. Qwen3-TTS Reconstruction (New)

Reconstructs audio waveforms from discrete codes (Vocoder step).

**Inputs:**
*   **model**: Connected from Model Loader.
*   **codes**: Discrete tensor codes (from Tokenizer or manual generation).

**Outputs:**
*   `AUDIO`: Standard ComfyUI audio dictionary.

**Use Case:**
*   Re-synthesizing audio from tokens.
*   Verifying the quality of the audio encoder/decoder loop.

### 4. Qwen3-TTS Sampler

Generates audio from the loaded model using Text, Instructions, or Reference Audio.

**Inputs:**
*   **model**: Connected from the Model Loader.
*   **text**: The text content to synthesize.
*   **instruction** (Optional): Text prompt for Voice Design (e.g., "A soft whisper").
*   **audio_prompt** (Optional): Reference audio for Custom Voice cloning.
*   **seed**: Random seed.
*   **cleanup_memory** (Boolean): Clears CUDA cache after generation.

**Outputs:**
*   `AUDIO`: Standard ComfyUI audio dictionary.

## Usage Examples

### Example 1: Low VRAM Workflow (4-bit Quantization)
*Best for running the 1.7B model on GPUs with 8GB - 12GB VRAM.*

1.  **Qwen3-TTS Model Loader**:
    *   `model_name`: `Qwen3-TTS-12Hz-1.7B-Base`
    *   `quantization`: `4-bit`
    *   `bnb_compute_dtype`: `bfloat16` (or `float16` if on older cards)
2.  **Qwen3-TTS Sampler**:
    *   Connect Model.
    *   Enter text.
    *   Enable `cleanup_memory`.

### Example 2: Voice Cloning (Custom Voice)
1.  **Load Audio**: Use a standard ComfyUI "Load Audio" node.
2.  **Qwen3-TTS Model Loader**:
    *   `model_name`: `Qwen3-TTS-12Hz-1.7B-CustomVoice`
3.  **Qwen3-TTS Sampler**:
    *   Connect Audio to `audio_prompt`.
    *   Enter text to be spoken in the cloned voice.

### Example 3: Audio Token Manipulation
*Testing the tokenizer and vocoder pipeline.*

1.  **Load Audio**: Load a short voice sample.
2.  **Qwen3-TTS Tokenizer**: Pass the audio through to get `codes`.
3.  **Qwen3-TTS Reconstruction**: Pass the `codes` through to get `AUDIO`.
4.  **Save Audio**: Listen to the result. This tests the quality of the audio codec used by Qwen.

## Performance & Optimization

### 4-bit Quantization
To save memory, enable `4-bit` in the Model Loader. This reduces the model size by ~75% with minimal quality loss.
*   **Requirement**: Must have `bitsandbytes` installed.
*   **Behavior**: Automatically forces `device_map="auto"` (CPU Offload) to manage the layers.

### Blackwell & Bfloat16
If you are using an NVIDIA RTX 50 series (Blackwell), 40 series, or 30 series:
*   Set `bnb_compute_dtype` to `bfloat16`.
*   This provides better numerical stability and utilizes the Tensor Cores efficiently on modern hardware.

## Troubleshooting

*   **Out Of Memory (OOM)**: 
    *   Enable **4-bit quantization** in the Model Loader.
    *   Ensure **offload** is checked.
*   **`bitsandbytes` Import Error**: 
    *   Ensure you have a CUDA-compatible version of PyTorch installed. 
    *   On Windows, you may need to install a precompiled wheel: `pip install bitsandbytes-windows`.
*   **Slow Generation**: 
    *   If you have >24GB VRAM, disable `offload` and set `quantization` to `none` for maximum speed.

## Model Storage

Models are automatically downloaded to:
*   `ComfyUI/models/Qwen3TTS/`

If you use `extra_model_paths.yaml`, models are saved to your custom defined path.

## Credits

*   **Qwen Team / Alibaba Cloud** for the original [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) models.
*   **ComfyUI** for the node-based interface.
*   **BitsAndBytes** for the quantization technology.

## License

This project is licensed under the Apache License 2.0. Please note that the Qwen3-TTS models themselves have their own specific licenses found on HuggingFace.
