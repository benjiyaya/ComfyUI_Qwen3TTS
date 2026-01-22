# ComfyUI_Qwen3TTS
ComfyUI custom node for Qwen3 TTS


Custom Node for Text To Speech AI using Qwen 3 TTS series (0.6B/1.7B) based on Qwen3-TTS-Tokenizer-12Hz.
https://github.com/QwenLM/Qwen3-TTS





Here is the README.md document for your GitHub project.

```markdown
# ComfyUI-Qwen3TTS

A ComfyUI custom node implementation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). This node allows you to generate high-quality text-to-speech audio directly within ComfyUI, featuring support for Voice Design (text-based styling) and Custom Voice (audio cloning).

![ComfyUI-Qwen3TTS](https://img.shields.io/badge/ComfyUI-Custom_Node-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

*   **Multiple Model Support**: Compatible with all 5 released Qwen3-TTS variants (1.7B and 0.6B).
*   **Auto-Download**: Automatically downloads model weights from HuggingFace if they are missing.
*   **Low VRAM Mode**: Built-in CPU offloading support to run large models on GPUs with limited VRAM.
*   **Memory Management**: Optional memory cleanup after generation to prevent Out-Of-Memory errors during long workflows.
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

    *Note: Ensure your PyTorch installation is compatible with your GPU drivers.*

4.  **Restart ComfyUI.**

## Requirements

The `requirements.txt` includes the following:
*   `transformers`
*   `accelerate` (Required for low VRAM offloading)
*   `huggingface_hub`
*   `torchaudio`
*   `sentencepiece`

## Nodes

### 1. Qwen3-TTS Model Loader

Loads the Qwen3-TTS model and processor from HuggingFace or local disk.

**Inputs:**
*   **model_name**: Select from the supported variants:
    *   `Qwen3-TTS-12Hz-1.7B-VoiceDesign`
    *   `Qwen3-TTS-12Hz-1.7B-CustomVoice`
    *   `Qwen3-TTS-12Hz-1.7B-Base`
    *   `Qwen3-TTS-12Hz-0.6B-CustomVoice`
    *   `Qwen3-TTS-12Hz-0.6B-Base`
*   **dtype**: Precision for the model weights (`float16` recommended for most GPUs).
*   **offload** (Boolean):
    *   `True` (Enable): Splits model between GPU and CPU (Best for Low VRAM).
    *   `False` (Disable): Loads entire model to GPU (Faster, requires more VRAM).

**Outputs:**
*   `model`: The loaded model object passed to the Sampler.

### 2. Qwen3-TTS Sampler

Generates audio from the loaded model.

**Inputs:**
*   **model**: Connected from the Model Loader.
*   **text**: The text content to synthesize.
*   **instruction** (Optional): Text prompt to describe the voice style (Used for **VoiceDesign** models). e.g., "Speak in an excited tone."
*   **audio_prompt** (Optional): Connect an Audio node to use as a reference for cloning (Used for **CustomVoice** models).
*   **seed**: Random seed for generation.
*   **cleanup_memory** (Boolean): If `True`, runs garbage collection and clears CUDA cache after generation to free up VRAM.

**Outputs:**
*   `AUDIO`: Standard ComfyUI audio dictionary. Connect to `Save Audio` or `Preview Audio`.

## Usage Example

1.  Add **Qwen3-TTS Model Loader**.
    *   Select `Qwen3-TTS-12Hz-0.6B-Base` for faster testing or `1.7B-VoiceDesign` for high quality.
    *   Keep **offload** checked if you have <12GB VRAM.
2.  Add **Qwen3-TTS Sampler**.
    *   Connect the model.
    *   Enter text: "Hello, this is a test of Qwen TTS in ComfyUI."
    *   (Optional) Enter instruction: "A soft female voice."
3.  Add **Save Audio** node (default ComfyUI node).
4.  Connect the **AUDIO** output from Sampler to **Save Audio**.
5.  Execute the workflow.

## Model Storage

Models are automatically downloaded to the following location by default:
*   `ComfyUI/models/Qwen3TTS/`

If you have configured `extra_model_paths.yaml`, the models will be saved to the custom path specified under `Qwen3TTS`.

## Troubleshooting

*   **Out Of Memory (OOM)**: Enable the **offload** option in the Model Loader. If that fails, try switching from the 1.7B model to the 0.6B model.
*   **Slow Generation**: Ensure `offload` is **unchecked** if you have sufficient VRAM (>16GB) to keep the model fully on the GPU.
*   **Import Errors**: Make sure you ran `pip install -r requirements.txt` inside the custom node folder or your main python environment.

## Credits

*   **Qwen Team / Alibaba Cloud** for the original [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) models and inference code.
*   **ComfyUI** for the powerful node-based interface.

## License

This project is licensed under the MIT License. Please note that the Qwen3-TTS models themselves have their own specific licenses which can be found on the HuggingFace model cards.
```
