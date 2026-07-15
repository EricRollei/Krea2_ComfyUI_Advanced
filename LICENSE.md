# Dual License

This software is available under a dual licensing model.

## For Non-Commercial Use

This work is licensed under a modified Creative Commons Attribution-NonCommercial 4.0 International License, with additional requirements:

### Attribution Requirements

1. **Mandatory Credit**: You must give appropriate credit to Eric Hiss as the original author by:
   - Including a link to the original repository (<https://github.com/EricRollei/Krea2_ComfyUI_Advanced>) in any documentation
   - Maintaining the original copyright and license notices in all copies or substantial portions of the software
   - Including attribution comments in your code that clearly state "Based on work by Eric Hiss (<https://github.com/EricRollei/Krea2_ComfyUI_Advanced>)"

2. **Modification Notice**: If you modify the software, you must clearly indicate what changes were made and attribute the original work to Eric Hiss.

3. **No Endorsement**: You may not use the name of the author or contributors to endorse or promote products derived from this software without specific prior written permission.

### Additional Restrictions

- **NonCommercial** - You may not use the material for commercial purposes.
- **No Redistribution Without Attribution** - You may not redistribute this code without the required attribution as specified above.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

### For the full text of the CC BY-NC 4.0 License

Visit <http://creativecommons.org/licenses/by-nc/4.0/> or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

## For Commercial Use

For commercial use of this software, a separate commercial license is required.

Please contact Eric Hiss at <eric@rollei.us> or <eric@historic.camera> for licensing options and terms.

### Commercial License Terms (General Overview)

1. **Definition of Commercial Use**: Any use of the software that is primarily intended for or directed toward commercial advantage or monetary compensation.

2. **License Grant**: Subject to payment of applicable fees and compliance with the terms of the commercial license agreement, licensee will be granted a non-exclusive, non-transferable license to use the software for commercial purposes.

3. **Mandatory Attribution**: All commercial uses must include appropriate attribution to Eric Hiss and a link to the original repository in a manner appropriate to the medium, visible to end users.

4. **Restrictions**: The commercial license does not allow redistribution, modification, or creation of derivative works without explicit permission and additional licensing.

5. **Support**: Commercial licenses may include support options depending on the specific agreement.

6. **Liability**: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For detailed commercial license terms and pricing, please contact the author directly.

## Third-Party Libraries and Dependencies

This project includes or depends on several third-party libraries and code, each with its own license:

### Models

- **Krea 2 (Raw / Turbo)**: The Krea 2 image model was created by Krea AI and its weights are released under the Krea 2 Community License. These nodes load the model via HuggingFace Diffusers and do not redistribute the weights; consult the official Krea 2 model card / repository for the full, authoritative license terms before any use.
  - Author: Krea AI
  - License: Krea 2 Community License
  - Repository: <https://github.com/krea-ai/krea-2>

- **Krea2Pipeline / Krea2Transformer2DModel**: The Krea 2 pipeline and transformer code (`diffusers/pipelines/krea2/pipeline_krea2.py`, `diffusers/models/transformers/transformer_krea2.py`) is part of HuggingFace Diffusers, licensed under Apache 2.0.
  - Repository: <https://github.com/huggingface/diffusers>

- **AutoencoderKLQwenImage (Krea 2 VAE)**: The f8 VAE used by Krea 2 for latent decode, provided through HuggingFace Diffusers (Apache 2.0); the underlying Qwen-Image VAE was developed by the Qwen Team (Alibaba).
  - Repository: <https://github.com/QwenLM/Qwen-Image>

### Wan2.1-VAE-upscale2x (2× Super-Resolution VAE) and base Wan 2.1 VAE

- **Model Weights**: Licensed under Apache 2.0
  - Author: spacepxl
  - HuggingFace: <https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x>
  - A decoder-only finetune of the Wan2.1 VAE with 2× upscaling integrated into the decoder. Used here for the optional between-stage and final upscale-decode paths, and (base Wan 2.1 VAE) as an alternate decode VAE.

- **ComfyUI-VAE-Utils** (reference implementation): Licensed under MIT License
  - Author: spacepxl
  - Repository: <https://github.com/spacepxl/ComfyUI-VAE-Utils>

- The upscale-VAE code in `_upscale_vae.py` is an independent implementation that loads and uses the above model weights via the HuggingFace Diffusers `AutoencoderKLWan` class. It does not contain code from ComfyUI-VAE-Utils, but was inspired by its approach.

### LoRA loading utilities

- The LoRA helpers in `_lora_utils.py` (standard LoRA / LoKR / LoHa loading, key normalization, PEFT injection with a direct-merge fallback, and compatibility diagnostics) are the author's own work, ported from the Eric_Qwen_Edit LoRA node and vendored here so this package remains self-contained.

### Python Libraries

- PyTorch (torch): BSD 3-Clause License
- HuggingFace Diffusers: Apache 2.0 License
- HuggingFace Transformers: Apache 2.0 License
- HuggingFace Accelerate: Apache 2.0 License
- PEFT: Apache 2.0 License
- NumPy: BSD 3-Clause License
- SciPy: BSD 3-Clause License
- PIL/Pillow: HPND License
- safetensors: Apache 2.0 License

### ComfyUI

- **ComfyUI**: GPL-3.0 License
  - Repository: <https://github.com/comfyanonymous/ComfyUI>
  - These nodes are designed to run as custom nodes within ComfyUI.

---

Copyright (c) 2026 Eric Hiss. All rights reserved.
